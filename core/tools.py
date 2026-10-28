"""工具注册表：Agent 可调用的原子能力"""
import inspect
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

from core.log import get_logger, brief
from core.workspace import MAX_READ_BYTES, WorkspaceError

log = get_logger("core.tools")

TOOLS = {}


class ConfirmRequired(Exception):
    """高危操作需用户确认（V23）。confirm_dangerous 开启时由 execute_tool
    走确认流；handler 收到 _confirmed=True 后放行。"""


# 同一任务内「只确认一次」的工具：用户点头后，本任务内后续调用直接放行。
# 只放 run_python——修脚本天然是「写→跑→改→再跑」的循环，每次都弹窗太吵；
# 递归删除、shell 命令的内容每次都不同（确认的就是那条具体命令），仍逐次确认。
ONCE_PER_TASK = frozenset({"run_python"})


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable
    schema: dict = None      # OpenAI function schema（原生 function calling 用）


def tool(name: str, description: str, parameters: dict, schema: dict = None):
    def deco(fn):
        TOOLS[name] = Tool(name, description, parameters, fn, schema)
        return fn
    return deco


def _schema(props: dict, required=()) -> dict:
    """构造 JSON Schema：props = {参数名: (类型, 说明)}"""
    return {
        "type": "object",
        "properties": {k: {"type": t, "description": d}
                       for k, (t, d) in props.items()},
        "required": list(required),
    }


# ------------------------------------------------------------------
# V28 原生 function calling：各工具的 JSON Schema
# （集中在此，与装饰器解耦；openai_tools() 导出）
# ------------------------------------------------------------------
TOOL_SCHEMAS = {
    "list_files": _schema({
        "path": ("string", "相对路径，默认 '.'"),
        "recursive": ("boolean", "是否递归子目录，默认 false")}),
    "read_file": _schema({
        "path": ("string", "工作区内文件相对路径")}, ["path"]),
    "write_file": _schema({
        "path": ("string", "工作区内文件相对路径"),
        "content": ("string", "要写入的完整文本内容")}, ["path", "content"]),
    "append_file": _schema({
        "path": ("string", "工作区内文件相对路径"),
        "content": ("string", "要追加的文本内容")}, ["path", "content"]),
    "create_dir": _schema({
        "path": ("string", "目录相对路径")}, ["path"]),
    "delete_file": _schema({
        "path": ("string", "相对路径"),
        "recursive": ("boolean", "递归删除整个目录树，默认 false")}, ["path"]),
    "search_files": _schema({
        "pattern": ("string", "glob 通配符，如 *.py")}, ["pattern"]),
    "run_command": _schema({
        "command": ("string", "要执行的 shell 命令")}, ["command"]),
    "web_collect": _schema({
        "query": ("string", "搜索关键词"),
        "criteria": ("string", "结果筛选标准，如'必须是2024年以后的新闻'"),
        "count": ("integer", "需要的有效数据条数，默认 10")}),
    "kb_search": _schema({
        "query": ("string", "要检索的问题或关键词"),
        "top_n": ("integer", "最多返回的相关文件数，默认 3")}),
    "run_python": _schema({
        "path": ("string", "工作区内 .py 脚本相对路径，如 calc.py"),
        "args": ("string", "传给脚本的命令行参数，空格分隔，可省略")}),
    "load_skill": _schema({
        "name": ("string", "技能名，如 'pdf-report'"),
        "path": ("string", "可选：技能自带参考文件的相对路径")}),
    "install_skill": _schema({
        "url": ("string", "git 仓库地址"),
        "subdir": ("string", "可选：仓库内技能子目录，留空自动查找")}),
}


def openai_tools(include: list = None, cfg=None, ws=None, task: str = "") -> list:
    """生成 OpenAI function calling 的 tools 参数（V28 原生协议路径）

    include: 仅导出指定工具（默认全部）。
    cfg/ws: 提供时追加「技能」工具——每个已发现技能暴露为一个独立工具，
    模型像挑工具一样挑技能（原生协议下模型对 tools 的选择明显更可靠）。
    task: 当前任务描述——非空时按关键词预筛技能，避免工具列表被撑爆。"""
    out = []
    for name, t in TOOLS.items():
        if include is not None and name not in include:
            continue
        schema = t.schema or TOOL_SCHEMAS.get(name)
        if not schema:
            continue
        out.append({"type": "function",
                    "function": {"name": t.name,
                                 "description": t.description,
                                 "parameters": schema}})
    # V30 技能：每个技能一个工具（skill_<slug>）
    try:
        from core.skills import tool_entries
        for e in tool_entries(cfg, ws, task=task):
            out.append({k: v for k, v in e.items() if k != "_skill"})
    except Exception as e:                      # 技能不可用不应影响主流程
        log.warning("技能工具导出失败: %s", e)
    return out


def _need_ws(ws):
    if ws is None:
        raise WorkspaceError("未挂载工作区，请先挂载（控制台 /mount 或界面「挂载工作区」按钮）")


@tool("list_files", "列出工作区目录下的文件和子目录",
      {"path": "相对路径，默认 '.'", "recursive": "是否递归子目录，默认 false"})
def _list_files(ws, cfg, path=".", recursive=False):
    _need_ws(ws)
    return ws.list_files(path, recursive)


@tool("read_file", "读取工作区内文本文件内容(限200KB)", {"path": "文件相对路径"})
def _read_file(ws, cfg, path):
    _need_ws(ws)
    return ws.read_file(path)


@tool("write_file", "创建或覆盖文本文件(自动创建父目录)",
      {"path": "文件相对路径", "content": "写入的文本内容"})
def _write_file(ws, cfg, path, content=""):
    _need_ws(ws)
    return ws.write_file(path, content)


@tool("append_file", "向文件末尾追加内容",
      {"path": "文件相对路径", "content": "追加的文本内容"})
def _append_file(ws, cfg, path, content=""):
    _need_ws(ws)
    return ws.append_file(path, content)


@tool("create_dir", "创建目录(支持多级路径)", {"path": "目录相对路径"})
def _create_dir(ws, cfg, path):
    _need_ws(ws)
    return ws.create_dir(path)


@tool("delete_file", "删除文件或目录(非空目录需 recursive=true)",
      {"path": "相对路径", "recursive": "递归删除，默认 false"})
def _delete_file(ws, cfg, path, recursive=False, _confirmed=False):
    _need_ws(ws)
    # 高危：递归删除一次性抹掉整个目录树，先要用户确认（V23）
    if recursive and not _confirmed:
        raise ConfirmRequired(f"递归删除目录 '{path}' 及其全部内容（不可恢复）")
    return ws.delete_file(path, recursive)


@tool("search_files", "按 glob 通配符递归搜索工作区内文件名", {"pattern": "如 *.py"})
def _search_files(ws, cfg, pattern):
    _need_ws(ws)
    return ws.search_files(pattern)


@tool("run_command", "在工作区目录内执行 shell 命令(需 allow_shell=true)",
      {"command": "命令字符串"})
def _run_command(ws, cfg, command="", _confirmed=False):
    _need_ws(ws)
    if not cfg.allow_shell:
        return "错误: 出于安全考虑 shell 工具默认关闭。确需启用请在配置/设置中打开 allow_shell。"
    # 高危：shell 是通用逃逸通道，执行前需用户确认（V23）
    if not _confirmed:
        raise ConfirmRequired(f"执行 shell 命令: {str(command)[:120]}")
    try:
        r = subprocess.run(command, shell=True, cwd=str(ws.root),
                           capture_output=True, text=True, timeout=30)
        out = ((r.stdout or "") + (r.stderr or ""))[:4000]
        return f"exit_code={r.returncode}\n{out or '(无输出)'}"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时(30s)"


@tool("web_collect",
      "联网采集：多引擎搜索（百度/必应/搜狗/360，风控自动切换）并用「筛选模型」"
      "自动过滤，直到凑满指定条数的有效数据（返回含标题/摘要/来源，可直接分析）。"
      "涉及最新资讯/网络信息时优先使用本工具",
      {"query": "搜索关键词。中文专有名词过长搜不到时可尝试加引号或缩短",
       "criteria": "结果筛选标准，强烈建议填写：描述想要什么样的结果"
                   "（如'必须是2024年以后的卫星发射新闻'）；留空则仅按与查询的相关性过滤",
       "count": "需要的有效数据条数，默认 10"})
def _web_collect(ws, cfg, query, criteria="", count=10, _progress=None,
                 _should_stop=None):
    if not getattr(cfg, "web_enabled", True):
        return ("错误: 联网采集已在设置中关闭。请直接基于已有知识回答，"
                "或提示用户在设置中开启联网。")
    from core.web_search import collect_until          # 延迟导入，未联网不受影响
    try:
        n = max(1, min(50, int(count)))
    except (TypeError, ValueError):
        n = 10
    return collect_until(query, criteria, n, cfg,
                         progress=_progress, should_stop=_should_stop)


# ------------------------------------------------------------------
# 知识库检索：工作区文件 × 「知识库检索模型」相关性判定
# ------------------------------------------------------------------
_KB_TEXT_EXT = {".txt", ".md", ".py", ".json", ".c", ".h", ".cpp", ".log",
                ".csv", ".yaml", ".yml", ".xml", ".ini", ".cfg", ".rst",
                ".java", ".js", ".ts", ".sh", ".bat", ".sql"}
_KB_MAX_FILES = 12          # 送模型的文件数上限
_KB_PREVIEW = 1200          # 每文件给模型看的字符数


@tool("kb_search",
      "知识库检索：在挂载的工作区文件中查找与问题相关的知识，由「知识库检索模型」"
      "判断相关性并摘录关键内容。工作区内有资料时优先使用本工具（先于联网搜索）",
      {"query": "要检索的问题或关键词",
       "top_n": "最多返回的相关文件数，默认 3"})
def _kb_search(ws, cfg, query, top_n=3):
    if ws is None:
        return "错误: 未挂载工作区。请先挂载包含资料文件的工作区。"
    try:
        n = max(1, min(8, int(top_n)))
    except (TypeError, ValueError):
        n = 3

    # 1. 收集文本文件（<200KB）
    candidates = []
    for p in sorted(ws.root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _KB_TEXT_EXT:
            continue
        try:
            if p.stat().st_size > MAX_READ_BYTES:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        candidates.append((p.relative_to(ws.root).as_posix(), text))
        if len(candidates) >= 60:            # 总量上限，防大目录
            break
    if not candidates:
        return "工作区内没有可检索的文本文件。"

    # 2. 文件过多 → 关键词预筛（查: 角色模型暂不可用时的兜底路径）
    kws = [w for w in _kb_split(query) if len(w) >= 2]
    if len(candidates) > _KB_MAX_FILES:
        def kw_hit(item):
            _, text = item
            return sum(1 for k in kws if k in text)
        candidates = sorted(candidates, key=kw_hit, reverse=True)[:_KB_MAX_FILES]

    # 3. 知识库检索模型判定（失败 → 关键词兜底，保证工具可用）
    from core.llm_client import LLMClient, is_embedding_model
    llm = LLMClient(cfg, profile="kb")
    verdicts = {}
    kb_note = ""
    if is_embedding_model(llm.model):
        # embedding 模型（如 bge-m3）没有对话能力，调 chat 必然失败——
        # 直接走关键词兜底，并给出明确配置指引（V23，修 kb 角色误配静默降级）
        kb_note = (f"（注意：kb 角色模型 {llm.model} 为 embedding 类模型，无对话能力，"
                   "本次为关键词兜底检索；请在设置中把「知识库检索」角色换成"
                   "支持 chat 的轻量模型）")
        log.warning("kb 角色模型 '%s' 是 embedding 类模型，无对话能力，"
                    "跳过模型判定走关键词检索", llm.model)
    else:
        try:
            lines = [f"{i}. 「{name}」: {text[:_KB_PREVIEW]}"
                     for i, (name, text) in enumerate(candidates, 1)]
            prompt = (f"用户问题：{query}\n下面是工作区内 {len(candidates)} 个文件的片段。"
                      "判断每个文件与问题的相关性并摘录最相关的原文关键句。\n"
                      '只输出 JSON 数组: [{"i": 1, "rel": true, "excerpt": "关键原文…"}]，'
                      "rel=false 表示不相关（excerpt 可为空）。\n\n" + "\n".join(lines))
            full = ""
            for d in llm.chat_stream([{"role": "user", "content": prompt}]):
                full += d
            m = re.search(r"\[.*\]", full, re.S)
            verdicts = {int(v["i"]): v for v in json.loads(m.group(0))
                        if isinstance(v, dict) and "i" in v} if m else {}
        except Exception as e:
            kb_note = f"（kb 角色模型调用失败，本次为关键词兜底检索：{brief(e, 120)}）"
            log.warning("kb 角色模型调用失败(%s)，走关键词兜底", brief(e, 200))
            verdicts = {}

    out = [f"知识库检索完成（问题「{query}」，检索范围 {len(candidates)} 个文件）{kb_note}"]
    hits = 0
    for i, (name, text) in enumerate(candidates, 1):
        v = verdicts.get(i, {})
        excerpt = str(v.get("excerpt", "")).strip()
        if v.get("rel") and (excerpt or any(k in text for k in kws)):
            hits += 1
            out.append(f"[{hits}] {name}\n    摘录: {excerpt[:200]}")
            if hits >= n:
                break
    if hits == 0:
        # 模型判定无果 → 关键词直出（工具永远可用）
        for name, text in candidates:
            for k in kws:
                idx = text.find(k)
                if idx >= 0:
                    hits += 1
                    out.append(f"[{hits}] {name}\n    关键词「{k}」上下文: "
                               f"{text[max(0, idx - 60):idx + 120]}")
                    break
            if hits >= n:
                break
    if hits == 0:
        out.append("未找到相关内容。可尝试更换关键词，或检查工作区文件内容。")
    return "\n".join(out)


# ------------------------------------------------------------------
# Python 脚本执行：直接调用本地 Python（sys.executable），默认启用
# ------------------------------------------------------------------
@tool("run_python",
      "执行 Python：用本地 Python 解释器运行工作区内的 .py 脚本并返回输出"
      "（stdout/stderr）。需要运行/验证 Python 代码时必须使用本工具直接执行，"
      "不要让用户自己运行。写好代码后：先 write_file 保存为 .py，再 run_python 执行",
      {"path": "工作区内 .py 脚本相对路径，如 calc.py",
       "args": "传给脚本的命令行参数，空格分隔，可省略"})
def _run_python(ws, cfg, path, args="", _progress=None, _confirmed=False):
    if not getattr(cfg, "python_enabled", True):
        return ("错误: Python 执行已在设置中关闭。请提示用户在设置中开启"
                "「允许执行 Python 脚本」。")
    if ws is None:
        return "错误: 未挂载工作区。请先挂载工作区（脚本须在工作区内）。"
    # 高危：本地 Python 同样能删工作区外文件/联网，与 shell 同级风险，
    # 执行前需用户确认（对齐 run_command 的安全模型）。但它属于
    # ONCE_PER_TASK：同一任务内确认过一次后不再重复询问。
    if not _confirmed:
        raise ConfirmRequired(
            f"执行 Python 脚本: {path}（本地解释器，无沙箱；"
            "同一任务内后续执行不再询问）")
    try:
        target = ws.resolve(path, must_exist=True)
        if target.suffix != ".py":
            return f"错误: '{path}' 不是 Python 脚本（.py）。"
    except WorkspaceError as e:
        return f"错误: {e}"
    # 交互式脚本预检：input()/stdin 会永远阻塞等不到人，与其挂满超时，
    # 不如立即拒绝并教模型怎么改（V17 闭环：失败原因要可行动）
    try:
        src = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"错误: 无法读取脚本: {e}"
    interactive = []
    if re.search(r"\binput\s*\(", src):
        interactive.append("input()")
    if re.search(r"\bsys\.stdin\b", src):
        interactive.append("sys.stdin")
    if interactive:
        return ("错误: 脚本包含交互式输入（" + "、".join(interactive) + "），"
                "Agent 在后台执行时无人输入会一直阻塞直到超时。\n"
                "请改写脚本后再运行：用命令行参数（sys.argv）或固定变量代替输入，"
                "结果用 print() 输出；例如把 while True+input 循环改为直接处理"
                "预设的一条指令并打印结果，作为可自动验证的版本。")
    # 分词：Windows 模型常给反斜杠路径（data\x.txt），shlex 会把 \ 当
    # 转义符吃掉 → 先归一为正斜杠再分词，两侧写法都能对上同一文件
    try:
        argv = [sys.executable, str(target)] + shlex.split(
            str(args or "").replace("\\", "/"))
    except ValueError:
        return "错误: args 参数格式非法（无法分词）"
    # Popen + 轮询：长跑时持续上报进度（防 UI 看门狗误报超时）
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            cwd=str(ws.root))
    t0 = time.monotonic()
    while proc.poll() is None:
        if time.monotonic() - t0 > 60:
            proc.kill()
            proc.wait()
            return ("错误: 脚本执行超时(60s)（脚本可能存在死循环或长时间计算，"
                    "请检查 while 循环的退出条件，或拆小计算量）")
        if _progress:
            _progress(f"脚本运行中… {int(time.monotonic() - t0)}s")
        time.sleep(5)
    out = proc.stdout.read() or ""
    err = proc.stderr.read() or ""
    code = proc.returncode
    parts = []
    if out:
        parts.append(f"[stdout]\n{out}")
    if err:
        parts.append(f"[stderr]\n{err}")
    if code != 0:
        parts.append(f"[退出码 {code}]")
    return "\n".join(parts) or "(脚本无输出，退出码 0)"


def _kb_split(query):
    return [w for w in re.split(r"[\s，,。；;、！!？?/\\]+", str(query)) if w]


@tool("load_skill",
      "加载技能（可复用的操作规范）全文。当任务涉及某个技能的专业流程/规范时"
      "先加载它，再按规范执行。可用技能会在本工具描述中列出。",
      {"name": "技能名，如 'pdf-report'",
       "path": "可选：读取技能自带的参考文件相对路径"})
def _load_skill(ws, cfg, name="", path=""):
    from core.skills import dispatch               # 延迟导入
    return dispatch(cfg, ws, name=name, path=path)


@tool("install_skill",
      "从 git 仓库安装技能（仓库内须含 SKILL.md）。用于获取他人分享的技能包。",
      {"url": "git 仓库地址，如 https://github.com/user/repo.git",
       "subdir": "可选：仓库内技能子目录，留空自动查找"})
def _install_skill(ws, cfg, url="", subdir=""):
    from core.skills import install_from_git        # 延迟导入
    r = install_from_git(url, subdir)
    return r["message"] + (f"（路径：{r['path']}）" if r.get("path") else "")


def execute_tool(name: str, ws, cfg, arguments: dict, progress=None,
                 confirm=None, should_stop=None, approved=None) -> str:
    """统一执行入口：永远返回字符串观察结果，不向 Agent 抛异常

    progress: 可选进度回调（callable(str)）。仅支持进度的工具
    （web_collect，参数名 _progress）会收到；其他工具自动忽略。
    confirm: 可选回调 confirm(name, reason) -> bool，用于高危操作
    （递归删除 / shell 命令 / 本地 Python 执行）执行前的用户确认。
    优先级：confirm_dangerous=false（免确认）> confirm 回调 > 无回调默认拒绝
    （--task 单次等非交互模式不传 confirm，保证安全）。
    should_stop: 可选停止信号（callable()->bool）。长任务工具
    （web_collect，参数名 _should_stop）会收到，使「停止」按钮能真正中断
    原本同步阻塞、看门狗触发也无法打断的采集。
    approved: 可选 dict，**任务级确认结果**记录 {工具名: 允许与否}。
    ONCE_PER_TASK 里的工具（run_python）用户确认过一次后写入 True，同一任务
    内的后续调用直接放行、不再弹窗——修脚本常需反复执行，每次都确认会很吵；
    用户拒绝则写入 False，本任务内后续调用直接拒绝、也不再弹窗（避免反复
    骚扰）。每次 Agent.run() 新建一个 dict 传入，任务结束即失效。"""
    # V30 技能工具：skill_<slug> 动态生成，不在 TOOLS 注册表里
    if name not in TOOLS:
        try:
            from core.skills import resolve_tool, load_skill_text
            skill = resolve_tool(name, cfg, ws)
            if skill is not None:
                log.info("执行技能工具: %s（%s）", name, skill.path)
                return load_skill_text(skill)
        except Exception as e:
            log.warning("技能工具执行失败 %s: %s", name, e)
    t = TOOLS.get(name)
    if t is None:
        return f"错误: 未知工具 '{name}'。可用工具: {', '.join(TOOLS)}"
    kwargs = dict(arguments or {})
    params = _sig_params(t.handler)
    if progress is not None and "_progress" in params:
        kwargs["_progress"] = progress
    if should_stop is not None and "_should_stop" in params:
        kwargs["_should_stop"] = should_stop
    _coerce_args(kwargs, params)                  # 模型常把 true/10 写成字符串
    try:
        # 参数形状前置校验：把「模型给了错参数」与「handler 内部 bug」分开
        inspect.signature(t.handler).bind(ws, cfg, **kwargs)
    except TypeError as e:
        log.warning("工具 %s 参数不匹配: %s", name, e)
        return f"错误: 参数不匹配 - {e}"
    except ValueError:
        pass                                      # 签名不可检（极少见），放行
    # 任务级确认结果命中：本任务内已对 run_python 做过决定 → 直接用结论
    if (approved is not None and name in ONCE_PER_TASK
            and name in approved and "_confirmed" in params):
        if not approved[name]:
            log.info("工具 %s 本任务内已被拒绝，直接拒绝（不再询问）", name)
            return ("错误: 用户已拒绝执行高危操作（本任务内已拒绝过该工具）。"
                    "请与用户确认意图，或换一种非破坏性方式，不要重复尝试同一操作。")
        log.info("工具 %s 本任务内已确认过，直接执行", name)
        try:
            return str(t.handler(ws, cfg, **dict(kwargs, _confirmed=True)))
        except WorkspaceError as e:
            log.warning("工具 %s 拒绝: %s", name, e)
            return f"错误: {e}"
        except Exception as e:
            log.exception("工具 %s 执行异常", name)
            return f"错误: 工具内部错误({type(e).__name__}): {e}"
    try:
        return str(t.handler(ws, cfg, **kwargs))
    except ConfirmRequired as e:
        return _run_confirmed(name, t, ws, cfg, kwargs, str(e), confirm, approved)
    except WorkspaceError as e:
        log.warning("工具 %s 拒绝: %s", name, e)
        return f"错误: {e}"
    except Exception as e:
        # 走到这里的是 handler 内部 bug（不再是模型参数问题）——
        # 分开回报，避免模型对着一个真 bug 反复改参数（V23）
        log.exception("工具 %s 执行异常", name)
        return f"错误: 工具内部错误({type(e).__name__}): {e}"


def _run_confirmed(name, tool_obj, ws, cfg, kwargs, reason, confirm,
                   approved=None) -> str:
    """高危操作确认流：免确认配置 > 用户回调 > 非交互默认拒绝

    用户给出结论后写入 approved（ONCE_PER_TASK 工具）：允许 → True，
    拒绝 → False。同一任务内的后续调用直接按该结论处理，不再弹窗。"""
    log.warning("高危操作待确认: %s | %s", name, reason)

    def _remember(ok: bool):
        if approved is not None and name in ONCE_PER_TASK:
            approved[name] = ok

    if not getattr(cfg, "confirm_dangerous", True):
        log.info("confirm_dangerous=false，免确认执行: %s", name)
        _remember(True)
    elif confirm is None:
        return ("错误: 该操作需要用户确认，但当前模式不支持交互确认，已拒绝执行。"
                f"（{reason}）如为受信自动化，可在配置中关闭 confirm_dangerous。")
    else:
        try:
            ok = bool(confirm(name, reason))
        except Exception as e:
            log.warning("确认回调异常，按拒绝处理: %s", e)
            ok = False
        _remember(ok)
        if not ok:
            return (f"错误: 用户已拒绝执行高危操作（{reason}）。"
                    "请与用户确认意图，或换一种非破坏性方式，不要重复尝试同一操作。")
    kwargs["_confirmed"] = True
    if "_confirmed" not in _sig_params(tool_obj.handler):
        return "错误: 该工具不支持确认后执行。"
    try:
        return str(tool_obj.handler(ws, cfg, **kwargs))
    except Exception as e:
        log.exception("工具 %s 确认后执行异常", name)
        return f"错误: 工具内部错误({type(e).__name__}): {e}"


def _coerce_args(kwargs: dict, params) -> None:
    """模型常把布尔/数字参数写成字符串（recursive:"true"、count:"10"），
    按 handler 签名默认值类型就地归一——否则 "false" 也是真值，
    recursive 删除会误触发（V23）"""
    for k, v in list(kwargs.items()):
        p = params.get(k)
        if p is None or not isinstance(v, str):
            continue
        if isinstance(p.default, bool):
            kwargs[k] = v.strip().lower() in ("true", "1", "yes", "on")
        elif isinstance(p.default, int) and not isinstance(p.default, bool):
            s = v.strip()
            if s.lstrip("-").isdigit():
                kwargs[k] = int(s)


def _sig_params(handler):
    try:
        return inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return {}


def tools_description(cfg=None, ws=None, task: str = "") -> str:
    """生成工具清单；关闭联网时隐藏 web_collect（模型不会尝试调用）

    V30：技能在此以「skill_<slug>」形式列出（文本协议路径没有原生 tools
    参数，靠这份清单让模型知道有哪些技能可用）。task 非空时按关键词预筛，
    只列相关技能——技能一多，全量清单会挤占模型的注意力。"""
    lines = []
    for t in TOOLS.values():
        if (cfg is not None and t.name == "web_collect"
                and not getattr(cfg, "web_enabled", True)):
            continue
        params = ", ".join(f"{k}({v})" for k, v in t.parameters.items()) or "无参数"
        lines.append(f"- {t.name}: {t.description}\n  参数: {params}")
    # V30 技能清单（渐进式披露第一层：只给名字+描述）
    try:
        from core.skills import discover, preselect, tool_name
        skills = discover(cfg, ws)
        if task:
            skills = preselect(skills, task)
        if skills:
            lines.append("- 可用技能（调用对应工具加载完整规范）：")
            for s in skills:
                lines.append(f"  - {tool_name(s)}: {s.description or '(无描述)'}")
    except Exception as e:
        log.warning("技能清单生成失败: %s", e)
    return "\n".join(lines)
