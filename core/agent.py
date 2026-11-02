"""Agent 核心：ReAct 循环（思考 → 工具调用 → 观察 → … → 回答）"""
import json
import re
import time
from typing import Callable, Iterator, Optional

from core.config import Config
from core.llm_client import LLMClient, LLMError
from core.log import get_logger, brief, oneline_json
from core.tools import execute_tool, openai_tools, tools_description
from core.workspace import Workspace, WorkspaceError

MAX_OBS_LENGTH = 4000

log = get_logger("core.agent")


def _parse_tool_args(raw) -> Optional[dict]:
    """解析原生工具 arguments；非法 JSON 或非对象返回 None，禁止空参数误执行。"""
    try:
        d = json.loads(raw or "{}")
        return d if isinstance(d, dict) else None
    except (json.JSONDecodeError, TypeError):
        log.warning("工具参数 JSON 解析失败: %s", brief(raw, 200))
        return None


def _is_skill_tool(name: str) -> bool:
    """是否技能工具（skill_<slug> 或 load_skill）"""
    n = str(name or "")
    return n.startswith("skill_") or n == "load_skill"

SYSTEM_PROMPT = """你是一个可以操作"工作区"目录的智能助手（Agent）。

{workspace_info}

可用工具：
{tools_description}

交互协议（严格遵守，每次回复只能是以下两种格式之一）：
1. 需要调用工具时（JSON 必须合法，用双引号）：
ACTION: {{"name": "工具名", "arguments": {{"参数": "值"}}}}
2. 给出最终答案时：
ANSWER: 最终答案内容

回复格式示例（模仿这种简洁格式，不要加解释文字）：
ACTION: {{"name": "list_files", "arguments": {{"path": "", "recursive": false}}}}
ANSWER: 工作区内共有 3 个文件：a.txt、b.py、notes.md。

工作准则：
- 面对陌生工作区先调用 list_files 观察，再决定行动
- 每次只调用一个工具，依据 OBSERVATION 结果调整下一步
- 文件路径一律使用工作区内相对路径，禁止越界
- 涉及资料/知识查询时优先 kb_search 检索工作区，工作区没有再联网
- 涉及网络/最新信息时优先使用可用的联网采集工具，不要凭记忆作答
- 需要运行或验证 Python 代码时：write_file 写入 .py 后立即 run_python 执行，
  拿到输出继续推理——绝不把"请用户自己运行"当作回答
- 不要编造工具结果或文件内容
- 任务完成后必须以 ANSWER 结束
- 使用中文交流"""

# V25「仅提问」模式系统提示：纯对话，不引导也不执行工具
SYSTEM_PROMPT_QA = """你是一个可以操作"工作区"目录的智能助手（Agent），当前为「仅提问」模式。

{workspace_info}

在本模式下你只进行对话与答疑：
- 直接像聊天一样回答用户的问题，不要输出 ACTION/ANSWER 协议
- 用户需要执行文件操作/联网采集/运行代码时，明确提示「请切换到 Agent 模式」
- 使用中文交流"""

# V28 原生 function calling 模式：system 保持极短。
# 实测（gpt-oss-20B 经 llama.cpp 网关）：system 里的长中文指令会让模型把
# 输出卡在 analysis 通道、final 通道为空；短 system + tools 参数则一切正常。
# V31 实测（gpt-oss-20B @ Ollama）：此前"不确定的事实先用 web_collect 核实"
# 一句会让模型对工作区内的代码符号去联网搜索（8 轮 web_collect 烧光迭代，
# 日志 17:29 实证）——本地任务必须显式规定"工作区优先"路由。
SYSTEM_PROMPT_NATIVE = (
    "你是可以操作工作区目录的智能助手。工作区内的文件、代码、变量/函数名"
    "一律先用本地工具处理：目录结构未知时才 list_files；已知目标直接读写。找代码符号用 "
    "search_files（会自动搜索文件内容）；知识问答用 kb_search。"
    "只有用户明确需要互联网上的时效信息（新闻/价格/软件版本等）时才调用 "
    "web_collect，绝不联网搜索工作区内的变量名或代码。"
    "每轮只调用一个工具，拿到结果后再决定下一步；不要并行发起多个工具调用。"
    "调用前检查必填参数；参数不确定时先澄清，不要猜测或传空参数。"
    "工具报错时先根据错误修正参数或更换方法，不要原样重复调用。"
    "需要文件信息或执行动作时调用工具，拿到结果后用中文回答，不要编造工具结果。"
    "用户要求编写并测试程序时，必须在最后一次修改代码后真实运行测试，"
    "若测试报错必须修改代码并重跑；未实际验证不得宣称测试通过。"
    "TCP 服务器调试用 start_process 或 tcp_echo_test，不要用同步 run_python 等待常驻服务退出。"
    "最终回答要完整给出结论，不要写“我需要先…”这类自述。"
    "若任务匹配某个 skill_* 技能工具，先调用它获取操作规范再执行。")


class Agent:
    # 教程式回答判定：长文+代码块+本任务还没用过工具（7B 模型常见病——
    # 用户要"创建并执行"，模型输出"打开编辑器粘贴代码"的教程）
    TUTORIAL_MIN_CHARS = 400
    HISTORY_TRUNCATE = 1200        # 长直答存入历史的截断长度（UI 显示全文）
    KEEP_CHARS = 16000             # 上下文字符级上限（超出先裁老消息）
    MAX_EMPTY_RETRIES = 3          # 连续空回复上限（reasoning 模型高发，V23）
    MAX_PROTOCOL_NUDGES = 2        # 思考含工具意图时的协议提醒次数上限（V26）
    REPEAT_NUDGE = 2               # 同一调用连续出现该次数起注入换思路警告（V31）
    TOOL_STREAK_NUDGE = 3          # 同一工具（无论参数）连续调用该次数起警告（V31）
    REASONING_BEAT = 1.0           # 思考期心跳间隔（秒）——喂 UI 看门狗

    def __init__(self, llm: LLMClient, workspace: Optional[Workspace] = None,
                 max_iterations: int = 8, qa_mode: bool = False):
        self.llm = llm
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.qa_mode = qa_mode      # V25 仅提问模式：纯对话，不执行工具
        self.messages = []          # 对话历史，跨 run 保持（多轮上下文）
        self._force_text = False    # 端点拒绝 tools 参数后置位，永久回退文本协议
        self._reuse_task_on_fallback = False  # native→text 回退时复用已入历史的任务消息
        self._last_call_sig = None  # 上一轮工具调用签名（V31 重复调用熔断）
        self._repeat_count = 0      # 当前签名连续重复次数
        self._last_tool_name = None # 上一轮工具名（V31 同工具刷屏熔断）
        self._tool_streak = 0       # 同一工具连续调用次数

    # ---------- 外部接口 ----------
    def set_workspace(self, ws: Optional[Workspace]):
        self.workspace = ws
        self._refresh_system_prompt()

    def set_llm(self, llm: LLMClient):
        self.llm = llm
        self._refresh_system_prompt()

    def set_qa_mode(self, on: bool):
        """切换「仅提问」模式（V25）并重建系统提示"""
        self.qa_mode = bool(on)
        self._refresh_system_prompt()

    def reset(self):
        self.messages.clear()

    def run(self, task: str,
            should_stop: Optional[Callable[[], bool]] = None,
            progress: Optional[Callable[[str], None]] = None,
            confirm: Optional[Callable[[str, str], bool]] = None) -> Iterator[dict]:
        """执行一次任务，生成器产出事件流:
        delta / action / observation / answer / format_error / cancelled / error

        progress: 可选进度回调（长任务工具如 web_collect 执行期间实时推送，
        绕过事件流直接到达前端——生成器在阻塞工具调用期间无法 yield，
        由调用方提供线程安全的通道，如 GUI 的 Qt 信号）
        confirm: 可选高危操作确认回调 confirm(name, reason) -> bool，
        透传给 execute_tool（递归删除 / shell / 本地 Python 执行前弹确认）"""
        self._last_call_sig, self._repeat_count = None, 0   # 重复熔断按任务重置
        self._last_tool_name, self._tool_streak = None, 0
        # V28 协议分流：native = 原生 function calling（gpt-oss 等思考模型
        # 唯一可靠路径）；text = 原有 ACTION/ANSWER 文本协议
        if self._native_mode():
            yield from self._run_native(task, should_stop, progress, confirm)
            return
        self._refresh_system_prompt(task)   # 技能清单按任务预筛（V30）
        skill_hint = self._auto_skill_hint(task)   # 强匹配技能自动注入（V30）
        if self._reuse_task_on_fallback:
            # 原生协议入口已经把当前任务写入历史；回退时复用该消息，
            # 避免重复追加任务导致小模型把同一请求理解成两个任务。
            self._reuse_task_on_fallback = False
        else:
            self.messages.append({"role": "user", "content": task + skill_hint})
        self._trim_history()
        used_tools = False          # 本任务是否已成功执行过工具
        redirected = False         # 教程式纠偏只用一次（防循环）
        fail_streak = 0            # 连续执行失败计数（闭环校验）
        empty_streak = 0           # 连续空回复计数（V23）
        protocol_nudge = 0         # 协议提醒次数（V26 思考含工具意图时追问）
        written = {}               # 本任务成功写入的文件 path 集合（V21 落盘追踪）
        approved = {}              # 本任务内各工具的确认结论 {name: 允许与否}
        log.info("任务开始: %s | 历史消息数=%d workspace=%s",
                 brief(task, 120), len(self.messages),
                 self.workspace.root if self.workspace else "未挂载")

        for step in range(self.max_iterations):
            if should_stop and should_stop():
                log.info("任务取消（外部停止）")
                yield {"type": "cancelled"}
                return

            full = ""
            chunk_n = 0
            t0 = time.monotonic()
            # 喂看门狗：覆盖「组消息 → 首 token」静默窗口，避免慢端点被误报超时
            yield {"type": "keepalive"}
            # V31.2：思考期心跳——reasoning 分片不产生正文事件，长时间静默
            # 会让 UI 看门狗误判超时（gpt-oss 推理模型高发）
            act = {"n": 0}
            last_beat = t0

            def _act():
                act["n"] += 1

            # 桩/自定义 LLM 可能没有 on_activity 形参 → 兼容调用
            try:
                stream = self.llm.chat_stream(self.messages, on_activity=_act)
            except TypeError:
                stream = self.llm.chat_stream(self.messages)
            try:
                for delta in stream:
                    full += delta
                    chunk_n += 1
                    yield {"type": "delta", "content": delta}
                    if act["n"]:
                        now = time.monotonic()
                        if now - last_beat >= self.REASONING_BEAT:
                            last_beat = now
                            yield {"type": "keepalive"}
                    if should_stop and should_stop():
                        break
            except LLMError as e:
                log.error("任务终止(LLM错误): %s", brief(e, 200))
                yield {"type": "error", "content": str(e)}
                return

            if should_stop and should_stop():
                if full:
                    self.messages.append({"role": "assistant", "content": full})
                log.info("任务取消（流式中断）")
                yield {"type": "cancelled"}
                return
            if not full.strip():
                # V23：空回复不再静默转圈。① yield 事件让 GUI 有反馈并喂
                # 看门狗（此前零事件 → 45s 后误报「对话超时」）；② 连续
                # 空回复带原因终止，不再耗尽 max_iterations
                empty_streak += 1
                reasoning = getattr(self.llm, "last_reasoning", "") or ""
                if reasoning:
                    log.warning("模型第 %d 次空回复：仅返回思考过程(%d字)无正文",
                                empty_streak, len(reasoning))
                else:
                    log.warning("模型回复为空，要求重答（第 %d 次）", empty_streak)
                if empty_streak >= self.MAX_EMPTY_RETRIES:
                    why = ("模型流式返回了思考过程但正文为空（llm_client 已将思考过程"
                           "作为回复输出，仍出现此错误说明端点响应异常，建议更换模型"
                           "或调整角色配置）。"
                           if reasoning else
                           "请检查模型端点是否正常工作（上下文过长、服务端限流、"
                           "模型未加载等都可能导致空响应）。")
                    log.error("连续 %d 次空回复，任务终止", empty_streak)
                    yield {"type": "error",
                           "content": f"模型连续 {empty_streak} 次返回空内容，任务终止。{why}"}
                    return
                yield {"type": "format_error",
                       "content": (f"模型回复为空（第 {empty_streak}/"
                                   f"{self.MAX_EMPTY_RETRIES} 次），已要求重试")}
                self.messages.append({"role": "user",
                                      "content": "（你的回复为空，请严格按 ACTION/ANSWER 协议回复）"})
                continue
            empty_streak = 0        # 正常回复，清零

            # 长直答截断入历史：UI/answer 事件显示全文，上下文只存摘要——
            # 防教程式长文逐轮堆积稀释协议引导（实测 7B 上下文惯性显著）
            self._append_assistant_history(full)
            log.info("模型回复(%d字): %s", len(full), brief(full, 600))
            t_llm = time.monotonic() - t0
            ctx_msgs = len(self.messages)
            ctx_chars = sum(len(m.get("content", "")) for m in self.messages)
            parsed = self._parse(full)
            # V22 调试事件：模型原文 + 解析判定（UI 调试卡展示，正常模式收起）
            yield {"type": "debug", "round": step + 1,
                   "llm": {"secs": round(t_llm, 1), "chunks": chunk_n,
                           "ctx_msgs": ctx_msgs, "ctx_chars": ctx_chars},
                   "raw": full,
                   "parse": ({"type": parsed.get("type"), "name": parsed.get("name"),
                              "arguments": parsed.get("arguments"),
                              "skipped": parsed.get("skipped", 0),
                              "note": parsed.get("note", "")} if parsed
                             else {"type": "直答", "note": "未检测到 ANSWER/ACTION 协议"})}

            if self.qa_mode:
                # 仅提问模式：任何输出都按直答处理——不执行工具、不要求重试
                log.info("仅提问模式：按直答处理（%s）",
                         parsed.get("type") if parsed else "无协议")
                yield {"type": "answer", "content": full.strip()}
                return

            # V26：兜底输出（端点未分离思考/正文）里若表达了工具意图、但协议
            # 不可解析 → 追问一次协议。模型下一轮若把 "ACTION: {...}" 写进
            # 思考通道，_parse 扫全文照样能抓到 → 工具真正跑起来
            if (getattr(self.llm, "reasoning_fallback_used", False)
                    and (parsed is None or parsed.get("type") == "bad_action")
                    and self._reasoning_needs_protocol(full)
                    and protocol_nudge < self.MAX_PROTOCOL_NUDGES):
                protocol_nudge += 1
                log.info("兜底输出含工具意图，注入协议提醒（第 %d 次）",
                         protocol_nudge)
                self.messages.append({
                    "role": "user",
                    "content": ("协议提醒：上一轮只输出了思考过程，没有按协议给出"
                                "可执行动作。现在请只输出下面两种格式之一，不要写"
                                "思考过程：\n"
                                "1) 需要调用工具：ACTION: {\"name\": \"工具名\", "
                                "\"arguments\": {...}}\n"
                                "2) 直接作答：ANSWER: 最终答案")})
                yield {"type": "redirect",
                       "content": "模型只输出了思考过程（该端点未分离正文），"
                                  "已要求它按协议重新输出"}
                continue

            if parsed is None:                      # 宽容策略：无 ACTION 视为直接回答
                if (not used_tools and not redirected
                        and len(full) > self.TUTORIAL_MIN_CHARS
                        and "```" in full):
                    # 教程式回答纠偏：用户要"执行"而模型在"教用户怎么做"。
                    # 注入明确指令要求改用工具，给一次重试机会
                    redirected = True
                    log.warning("检测到教程式回答(%d字含代码块)，注入纠偏", len(full))
                    self.messages.append({
                        "role": "user",
                        "content": ("纠偏：你刚才输出的是给用户看的操作教程，"
                                    "而不是执行任务。请立即改用工具实际完成："
                                    "先用 ACTION 调 write_file 把代码写入工作区文件，"
                                    "再用 ACTION 调 run_python 执行它并读取输出。"
                                    "严格按 ACTION 协议回复，不要输出教程。")})
                    yield {"type": "redirect",
                           "content": "检测到教程式回答（模型在教你做而不是自己做），"
                                      "已要求模型改用工具实际执行"}
                    continue
                log.info("无 ACTION 协议 → 按直答处理")
                yield {"type": "answer", "content": self._clean_answer(full)}
                return
            if parsed["type"] == "answer":
                log.info("ANSWER: %s", brief(parsed["content"], 600))
                yield {"type": "answer", "content": parsed["content"]}
                return
            if parsed["type"] == "bad_action":      # 有 ACTION 意图但格式坏，要求重试
                log.warning("ACTION 格式错误，已要求重试: %s", brief(full, 300))
                yield {"type": "format_error",
                       "content": "ACTION 格式错误，已要求模型重试"}
                self.messages.append({
                    "role": "user",
                    "content": "格式错误：ACTION 的 JSON 不合法。请严格按协议重新回复。"})
                continue

            name, arguments = parsed["name"], parsed["arguments"]
            # 逐段提取跳过了非法段 → 告知模型（否则它以为前面的 write 成功了，
            # 原样重发 → 死循环：跳过的 write + 执行失败的 run）
            if parsed.get("skipped"):
                yield {"type": "format_error",
                       "content": (f"你的前 {parsed['skipped']} 个 ACTION 段 JSON 非法已跳过"
                                   "（常见原因：content 用了 Python 三引号 \"\"\"），"
                                   "本次执行了后一个合法段。请检查 JSON 字符串转义")}
            log.info("ACTION: %s %s", name, oneline_json(arguments))
            yield {"type": "action", "name": name, "arguments": arguments}
            result = self._execute_and_observe(
                name, arguments, step, task, written,
                should_stop=should_stop, progress=progress, confirm=confirm,
                approved=approved,
                repeat=self._track_repeat(name, arguments)[0],
                streak=self._tool_streak)
            yield {"type": "observation", "content": result}
            used_tools = True        # 本任务已实际执行过工具

        log.error("达到最大迭代次数(%d)，任务终止", self.max_iterations)
        yield {"type": "error",
               "content": f"已达到最大迭代次数({self.max_iterations})，任务终止。"}

    # ---------- V28 原生 function calling 协议 ----------
    def _native_mode(self) -> bool:
        """是否走原生 function calling 协议

        config.protocol：auto（默认，优先 native，端点不支持时自动回退 text）
        / native（强制）/ text（强制走原文本协议）"""
        if getattr(self, "_force_text", False):
            return False
        if self.qa_mode:                 # 仅提问模式：一律走文本路径（不调工具）
            return False
        mode = (getattr(self._config(), "protocol", "auto") or "auto").lower()
        return mode != "text"

    def _run_native(self, task, should_stop, progress, confirm):
        """原生 function calling 循环（V28）

        模型用 OpenAI 原生 function calling 表达工具调用 → 执行 → 以
        role=tool 回传结果 → 模型据此给出最终回答。gpt-oss 这类用原生
        tools 训练过的模型在这条路上才稳定（文本协议会卡在 analysis 通道）。"""
        cfg = self._config()
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = {"role": "system", "content": SYSTEM_PROMPT_NATIVE}
        else:
            self.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_NATIVE})
        ws = self.workspace
        ws_hint = (f"当前工作区（绝对路径）：{ws.root}，文件操作限定在此目录内。"
                   if ws else "当前未挂载工作区，文件类工具不可用。")
        # V30 技能自动匹配（参考 Jarvis「固定流程筛选必要规则」阶段）：
        # 任务与某技能强匹配时，直接把规范注入任务消息——小模型常漏掉
        # 「该调哪个技能工具」，主动注入比指望它自己发现可靠得多。
        skill_hint = self._auto_skill_hint(task, cfg)
        # 小模型编码测试规则：解决多次真实评测里覆盖 TestCase.run 导致的
        # TextTestResult 参数错误，以及跨测试共享数据库造成的假失败。
        coding_hint = ""
        if re.search(r"unittest|单元测试", task, re.I):
            coding_hint = ("\n\n编码验收要点：unittest.TestCase 的 run(self, result) 是框架保留入口，"
                           "绝不能用 def run(self, *args) 定义命令辅助函数；请命名 run_cli。"
                           "每个测试用临时独立 TASK_DB，调用子进程用 sys.executable，"
                           "并将 UTF-8 stdout/stderr 错误定位到产生异常的测试文件。"
                           "README 必须给出可复制的 add/list/done 示例，不写占位文字。")
        # TCP 联调需要实际套接字证据，不允许仅向用户描述手工运行步骤。
        network_debug_task = bool(ws and
            re.search(r"TCP|socket|服务端|服务器|客户端|网络通信", task, re.I) and
            re.search(r"调试|测试|联调|发送|接收|连通", task, re.I) and
            re.search(r"你自己|自己|实际|执行|进行|帮我|请", task, re.I))
        if network_debug_task:
            coding_hint += ("\n\n这是必须执行的 TCP 联调任务，不能只说明如何手工运行。"
                "先检查并修复服务端、客户端代码，再调用 tcp_echo_test，"
                "设置 server_path、client_path、port 和 message='hello'。"
                "该工具会启动服务端、发送消息、检查客户端，并清理进程。"
                "若失败，阅读错误、修改代码后重试；测试通过才能宣称完成。"
                "若需手动逐步调试，使用 start_process/process_status/stop_process 管理服务。")
        self.messages.append({"role": "user",
                              "content": f"{ws_hint}\n\n用户任务：{task}{skill_hint}{coding_hint}"})
        self._trim_history()
        # Python 小项目只暴露编码所需的少量工具，降低小模型在 13+ 工具间
        # 选择的负担与 Ollama 上下文占用；其他项目仍保留完整工具集。
        python_project = bool(ws and re.search(r"Python|unittest|pytest|\.py", task, re.I)
                              and re.search(r"开发|创建|实现|编写|修改|项目|测试|test", task, re.I))
        tools_include = (["list_files", "read_file", "write_file", "append_file",
                          "create_dir", "search_files", "run_python",
                          "start_process", "process_status", "stop_process", "tcp_echo_test"]
                         if network_debug_task else
                         (["list_files", "read_file", "write_file", "append_file",
                           "create_dir", "search_files", "run_python"]
                          if python_project else None))
        tools = openai_tools(include=tools_include, cfg=cfg, ws=self.workspace, task=task)
        empty_streak = 0
        loaded_skills = []           # 本任务已加载的技能（V30）
        written = {}                 # 本任务写入的文件（落盘追踪，原生路径同样适用）
        approved = {}                # 本任务内各工具的确认结论 {name: 允许与否}
        # 编码任务以实际测试为完成条件，不能把 write_file 成功当作项目完成。
        coding_task = bool(ws and re.search(r"测试|运行|执行|验证|test|项目|project", task, re.I)
                           and re.search(r"代码|脚本|程序|创建|编写|实现|修改|开发|app\.py|\.py", task, re.I))
        # TCP 联调以 tcp_echo_test 的真实发送/接收为验收标准；不能因工具名含
        # test 或任务要求「测试结果」，误要求另外生成 unittest 脚本。
        explicit_unit_tests = bool(re.search(
            r"单元测试|unittest|pytest|测试脚本|测试文件|test[_-][\w-]*\.py", task, re.I))
        tests_required = bool(coding_task and
            (explicit_unit_tests or (not network_debug_task and
             re.search(r"测试|test", task, re.I))))
        code_revision, verified_revision, tested_revision = 0, -1, -1
        network_verified = False
        completion_nudges = 0
        doc_nudges = 0
        # 同一版本的脚本运行失败后必须先修改代码，再允许重试。
        failed_exec_revision = {}
        # 在模型端点连续 5xx 时允许有限次数的上下文恢复，不丢失工作区产物。
        model_recoveries = 0
        reasoning_recoveries = 0
        last_execution_failure = ""
        if "gpt-oss" in str(getattr(self.llm, "model", "")).lower():
            self.KEEP_CHARS = min(self.KEEP_CHARS, 5200)
        log.info("原生协议启动: model=%s 工具数=%d",
                 getattr(self.llm, "model", "?"), len(tools))
        # V31：手动计数替代 for-range——空回合重试不消耗迭代配额。
        # gpt-oss-20B 实测偶发"全空回合"（无正文/思考/工具调用），此前每次
        # 重试烧掉 1/8 迭代，8 上限经不起浪费；由 MAX_EMPTY_RETRIES 封顶防死循环。
        step = 0
        while step < self.max_iterations:
            if should_stop and should_stop():
                log.info("任务取消（外部停止）")
                yield {"type": "cancelled"}
                return
            content_parts, tool_calls = [], []
            yield {"type": "keepalive"}        # 喂看门狗：覆盖「组消息 → 首 token」静默窗口
            try:
                for ev in self.llm.chat_turn(self.messages, tools=tools):
                    if ev["type"] == "content":
                        content_parts.append(ev["text"])
                        yield {"type": "delta", "content": ev["text"]}
                    elif ev["type"] == "tool_call":
                        tool_calls.append(ev)
                    elif ev["type"] == "keepalive":
                        # V31.2：思考期心跳，透传给 UI 喂看门狗（防误报超时）
                        yield {"type": "keepalive"}
            except LLMError as e:
                # 端点不认 tools 参数 → 自动回退文本协议
                if ("HTTP 400" in str(e)
                        and not getattr(self, "_force_text", False)):
                    self._force_text = True
                    self._reuse_task_on_fallback = True
                    log.warning("端点拒绝 tools 参数（HTTP 400），自动回退文本协议")
                    yield {"type": "redirect",
                           "content": "该端点不支持原生工具调用，已自动切换到文本协议"}
                    # _run_native 已经把当前任务追加到历史。移除这一条，
                    # 让文本协议入口重新追加一次；保留此前完整对话上下文。
                    yield from self.run(task, should_stop, progress, confirm)
                    return
                recoverable = any(f"HTTP {code}" in str(e) for code in (500, 502, 503, 504))
                if (recoverable and not content_parts and not tool_calls
                        and model_recoveries < 2 and ws is not None):
                    model_recoveries += 1
                    try:
                        listing = str(ws.list_files(".", recursive=True))[:1600]
                    except Exception:
                        listing = "（无法自动列目录，请使用 list_files）"
                    system_msg, user_msg = self.messages[:2]
                    self.messages = [system_msg, user_msg, {"role": "user",
                        "content": (f"继续此前任务，不要重头创建已经存在的文件。"
                                    f"模型服务发生 HTTP 5xx，已压缩过长的工具历史（恢复第{model_recoveries}/2次）。"
                                    f"目前工作区文件：\n{listing}\n"
                                    "请先 read_file 检查现有代码/测试，修复未完成工作。"
                                    "之前的模型回答不能证明测试通过，必须真实运行并核对退出码。") }]
                    log.warning("模型 HTTP 5xx 后压缩上下文恢复，第 %d 次", model_recoveries)
                    yield {"type": "redirect", "content": "模型服务暂时出错，压缩上下文后继续当前任务"}
                    continue
                log.error("任务终止(LLM错误): %s", brief(e, 200))
                yield {"type": "error", "content": str(e)}
                return

            text = "".join(content_parts)
            if tool_calls:
                tool_msgs, results = [], []
                # 大块 write_file 参数只在实际请求中需要一次；下一轮历史只保留
                # 文件路径与内容摘要，减少上下文膨胀（必要时模型可 read_file）。
                history_calls = []
                for tc in tool_calls:
                    raw_args = tc["arguments"]
                    if tc["name"] in ("write_file", "append_file"):
                        parsed_args = _parse_tool_args(raw_args)
                        if parsed_args and len(str(parsed_args.get("content", ""))) > 400:
                            source = str(parsed_args["content"])
                            parsed_args["content"] = (source[:200] +
                                f"\n...[已实际写入文件，共{len(source)}字符；需要全文请 read_file]...")
                            raw_args = json.dumps(parsed_args, ensure_ascii=False)
                    history_calls.append({"id": tc["id"], "type": "function",
                                          "function": {"name": tc["name"],
                                                       "arguments": raw_args}})
                self.messages.append({"role": "assistant", "content": text or None,
                                      "tool_calls": history_calls})
                for tc in tool_calls:
                    args = _parse_tool_args(tc["arguments"])
                    log.info("ACTION: %s %s", tc["name"], oneline_json(args))
                    if args is None:
                        result = ("错误: 工具参数不是合法 JSON 对象。请重新生成参数，"
                                  "必须使用双引号并提供 object 格式；本次未执行工具。")
                        yield {"type": "format_error", "content": result}
                    else:
                        yield {"type": "action", "name": tc["name"], "arguments": args}
                        call_key = (tc["name"], json.dumps(args, ensure_ascii=False,
                                                            sort_keys=True))
                        repeat = self._track_repeat(tc["name"], args)[0]
                        if (tc["name"] in ("run_python", "run_command") and
                                failed_exec_revision.get(call_key) == code_revision):
                            result = ("错误: 相同代码版本上次运行已失败，本次禁止无修改重复运行。"
                                      "先 read_file 查看错误文件，使用 write_file 修复代码后，"
                                      "才能重新调用 run_python；原样重跑不能解决问题。")
                        else:
                            result = self._execute_and_observe(
                                tc["name"], args, step, task, written,
                                should_stop=should_stop, progress=progress,
                                confirm=confirm, approved=approved,
                                repeat=repeat, streak=self._tool_streak)
                        if (tc["name"] in ("run_python", "run_command") and
                                self._execution_failed(tc["name"], result)):
                            failed_exec_revision[call_key] = code_revision
                            last_execution_failure = "\n".join(result.splitlines()[-14:])[-1200:]
                        elif tc["name"] in ("run_python", "run_command"):
                            failed_exec_revision.pop(call_key, None)
                    yield {"type": "observation", "content": result}
                    tool_msgs.append({"role": "tool", "tool_call_id": tc["id"],
                                      "name": tc["name"], "content": result})
                    results.append(result)
                    if args is not None:
                        ok = not self._execution_failed(tc["name"], result)
                        file_path = str(args.get("path", "")).replace("\\\\", "/").lower()
                        if ok and tc["name"] in ("write_file", "append_file") and file_path.endswith(".py"):
                            code_revision += 1
                            if network_debug_task:
                                network_verified = False
                        if ok and tc["name"] in ("run_python", "run_command"):
                            verified_revision = code_revision
                            if (tc["name"] == "run_python" and
                                    re.search(r"(^|/)test[^/]*\.py$", file_path) and
                                    re.search(r"Ran\s+[1-9]\d*\s+tests?", result) and
                                    re.search(r"(?m)^OK\s*$", result)):
                                tested_revision = code_revision
                        if (ok and tc["name"] == "tcp_echo_test"
                                and "TCP联调通过" in result):
                            network_verified = True
                            verified_revision = code_revision
                    # V30 技能加载：规范是「接下来怎么做」的指令，不是任务结果——
                    # 不能参与目标核对闭环，否则模型会以为加载完就算完成
                    if _is_skill_tool(tc["name"]):
                        loaded_skills.append(tc["name"])
                # 目标核对闭环（对应文本协议 V17）：工具全成功时提醒模型自检——
                # 实测缺这一步会出现"工具全成功却回答『无法完成』"（模型自评失败）
                if (tool_msgs and not loaded_skills
                        and not any(self._execution_failed("", str(r)) for r in results)):
                    tool_msgs[-1]["content"] += (
                        "\n\n（本工具操作成功，但不等于整个任务完成。若用户要求运行/测试，"
                        "请先执行测试并确认退出码为零，修复失败再重跑；全部步骤完成后才给最终答案。）")
                # 已加载技能：显式要求按规范执行（不依赖模型自发理解）
                if loaded_skills:
                    tool_msgs[-1]["content"] += (
                        "\n\n（已加载技能规范：" + "、".join(loaded_skills)
                        + "。请严格按照上面技能规范中的步骤执行本任务，"
                          "规范要求调用的工具要照常调用；完成后再给出最终答案。）")
                # 小项目交付检查点：实际单元测试成功且文档齐全后直接收尾。
                # 避免小模型在已完成任务后继续重写通过的代码，引入回归。
                if (python_project and tests_required and code_revision > 0
                        and tested_revision == code_revision and ws is not None):
                    try:
                        readme = ws.resolve("README.md", must_exist=True)
                        docs = readme.read_text(encoding="utf-8")
                        docs_ready = (len(docs.strip()) >= 120 and
                                      all(cmd in docs for cmd in ("add", "list", "done")))
                    except (OSError, ValueError, WorkspaceError):
                        docs_ready = False
                    if docs_ready:
                        self.messages.extend(tool_msgs)
                        self._trim_history()
                        final_text = ("已完成 Python 项目文件与 README.md；"
                                      "最后一次代码修改后，run_python 执行的 unittest "
                                      "测试报告通过。请以工作区文件和实际测试输出为准。")
                        self._append_assistant_history(final_text)
                        log.info("项目验证检查点已通过，阻止后续无意义重写")
                        yield {"type": "answer", "content": final_text}
                        return
                self.messages.extend(tool_msgs)
                self._trim_history()
                step += 1               # 只有真实执行过工具才消耗迭代配额
                continue

            if text.strip():
                text = self._clean_answer(text)
                if network_debug_task and not network_verified:
                    missing = "尚未使用 tcp_echo_test 完成真实 TCP 发送、回显及客户端验证"
                    if completion_nudges < 3 and step < self.max_iterations - 1:
                        completion_nudges += 1
                        self._append_assistant_history(text)
                        self.messages.append({"role": "user", "content":
                            f"执行任务未通过：{missing}。不能仅输出手工运行说明。"
                            "请直接调用 tcp_echo_test；若失败就修复对应代码并重新测试。"})
                        yield {"type": "redirect", "content": missing}
                        step += 1
                        continue
                    yield {"type": "error", "content": missing + "，不能宣称任务已完成。"}
                    return
                if coding_task and code_revision and (verified_revision < code_revision or
                        (tests_required and tested_revision < code_revision)):
                    missing = ("执行 test*.py 并确认 unittest 测试通过"
                               if tests_required and tested_revision < code_revision
                               else "用 run_python 实际运行新修改的程序并确认结果")
                    if completion_nudges < 2 and step < self.max_iterations - 1:
                        completion_nudges += 1
                        self._append_assistant_history(text)
                        self.messages.append({"role": "user", "content":
                            f"验收未通过：最后一次代码修改后尚未{missing}。"
                            "你不能宣称项目完成。立即调用真实工具进行验证；失败就修改后重试。"})
                        log.warning("阻止未验证的代码任务提前回答：%s", missing)
                        yield {"type": "redirect", "content": f"尚未{missing}，继续验证"}
                        step += 1
                        continue
                    yield {"type": "error", "content": f"代码已写入，但尚未{missing}；"
                           "不能报告项目完成。请检查测试输出和迭代上限。"}
                    return
                # 编码任务文档完整性：明确要求 README.md 时，不允许交付空文档、
                # 占位文档或缺少用户指定的命令示例。
                if coding_task and ws is not None and re.search(r"README\.md", task, re.I):
                    readme_path = ws.root / "README.md"
                    try:
                        readme = readme_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeError):
                        readme = ""
                    expected_commands = set(re.findall(
                        r"python\s+\S+\.py\s+([a-z][\w-]*)", task, re.I))
                    missing_commands = [cmd for cmd in sorted(expected_commands)
                                        if not re.search(r"\b" + re.escape(cmd) + r"\b",
                                                         readme, re.I)]
                    doc_issue = (not readme_path.is_file() or len(readme.strip()) < 180
                                 or bool(missing_commands) or "此处" in readme
                                 or "待补充" in readme)
                    if doc_issue:
                        reason = ("README.md 不存在或过于简略，必须写清使用步骤。"
                                  + (f"缺少这些命令：{', '.join(missing_commands)}。" if missing_commands else ""))
                        if doc_nudges < 3 and step < self.max_iterations - 1:
                            doc_nudges += 1
                            self._append_assistant_history(text)
                            self.messages.append({"role": "user",
                                "content": (f"交付验收未通过：{reason}"
                                            "现在使用 write_file 补全 README.md，"
                                            "包括项目说明、安装/运行方法和用户要求的所有 CLI 示例。"
                                            "只修改文档即可；如果也修改 Python 代码，则必须重跑测试。")})
                            log.warning("阻止缺失/不完整文档交付: %s", reason)
                            yield {"type": "redirect", "content": reason}
                            step += 1
                            continue
                        yield {"type": "error",
                               "content": f"项目单元测试已完成，但{reason}无法宣称全部交付完成。"}
                        return
                self._append_assistant_history(text)
                log.info("ANSWER: %s", brief(text, 600))
                yield {"type": "answer", "content": text}
                return
            # 空回合：不消耗迭代配额（V31），MAX_EMPTY_RETRIES 封顶
            empty_streak += 1
            if empty_streak >= self.MAX_EMPTY_RETRIES:
                if coding_task and ws is not None and reasoning_recoveries < 2:
                    reasoning_recoveries += 1
                    empty_streak = 0
                    try:
                        listing = str(ws.list_files(".", recursive=True))[:1300]
                    except Exception:
                        listing = "请使用 list_files 检查当前文件"
                    self.messages = [self.messages[0], self.messages[1], {"role": "user",
                        "content": ("上一轮推理耗尽生成预算却没有可执行工具调用。"
                                    "继续原有项目，不要重建现有文件，不要空回复。"
                                    f"当前文件：{listing}\n"
                                    f"最近测试失败摘要：{last_execution_failure[-800:]}\n"
                                    "请先 read_file 查看相关文件，再用 write_file 修复失败原因，"
                                    "最后 run_python 真实验证。必须测试通过后才能完成任务。")}]
                    yield {"type": "redirect", "content": "连续只返回推理，压缩上下文继续修复"}
                    log.warning("原生空回复耗尽后恢复任务 %d/2", reasoning_recoveries)
                    continue
                log.error("连续 %d 次空回复，任务终止", empty_streak)
                yield {"type": "error",
                       "content": f"模型连续 {empty_streak} 次既无正文也无工具调用，"
                                  "任务终止。请检查模型端点。"}
                return
            yield {"type": "format_error",
                   "content": f"模型本轮既无正文也无工具调用（第 {empty_streak}/"
                              f"{self.MAX_EMPTY_RETRIES} 次），已要求重试（不消耗迭代）"}
            self.messages.append({"role": "user",
                                  "content": "（上一轮输出为空。请调用工具继续任务，"
                                             "或直接给出最终中文回答。）"})
        log.error("达到最大迭代次数(%d)，任务终止", self.max_iterations)
        yield {"type": "error",
               "content": f"已达到最大迭代次数({self.max_iterations})，任务终止。"}

    # ---------- 工具执行 + 闭环校验（文本/原生协议共用） ----------
    def _track_repeat(self, name: str, arguments: dict):
        """记录本任务的工具调用签名，返回 (同签名连续次数, 同工具连续次数)

        小模型两种高发病（日志实证，V31）：
        ① 完全相同的调用连发（search_files 相同 pattern 连调 5 次）；
        ② 同一工具换参数刷屏（web_collect 换 8 种关键词烧光迭代）。
        前者按「工具+参数」签名计数，后者只按工具名计数。"""
        try:
            sig = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        except (TypeError, ValueError):
            sig = (name, str(arguments))
        if sig == self._last_call_sig:
            self._repeat_count += 1
        else:
            self._last_call_sig = sig
            self._repeat_count = 1
        if name == self._last_tool_name:
            self._tool_streak += 1
        else:
            self._last_tool_name = name
            self._tool_streak = 1
        return self._repeat_count, self._tool_streak

    def _execute_and_observe(self, name, arguments, step, task, written,
                             should_stop=None, progress=None, confirm=None,
                             approved=None, repeat=1, streak=1) -> str:
        """执行工具并生成观察结果（含日志/截断/闭环提示/重复调用警告）。

        文本协议与原生协议原本各写一份几乎相同的「执行→日志→截断→提醒」
        流程，现抽为共用方法——修 bug 只需改一处。
        approved: 本任务的一次性授权集合（run_python 确认过一次后免再询问）。
        repeat: 该「工具+参数」签名连续重复的次数；streak: 同一工具连续调用
        的次数（无论参数）。超过阈值时在观察结果里注入换思路警告（V31）。"""
        t0 = time.monotonic()
        result = execute_tool(name, self.workspace, self._config(),
                              arguments, progress=progress, confirm=confirm,
                              should_stop=should_stop, approved=approved)
        log.info("OBSERVATION(%d字, 耗时%.1fs): %s",
                 len(result), time.monotonic() - t0, brief(result, 400))
        if len(result) > MAX_OBS_LENGTH:
            half = MAX_OBS_LENGTH // 2
            result = (result[:half] + "\n...(中间内容过长已截断，保留末尾错误原因)...\n"
                      + result[-half:])
        # ---------- 执行结果闭环校验（V17） ----------
        failed = self._execution_failed(name, result)
        if not failed and name == "write_file" and "path" in arguments:
            written[str(arguments["path"]).replace("\\", "/")] = True
        if failed:
            obs = f"OBSERVATION: {result}"
            obs += (f"\n\n（执行失败。请仔细阅读上方报错，分析根本原因，"
                    "修改代码/参数后**重新执行**——不要只输出代码让用户"
                    "自己运行，不要放弃。）")
            if name in ("run_python", "run_command"):
                # Python/unittest traceback 的真正错误通常在最后几行。
                lines = [line.strip() for line in result.splitlines() if line.strip()]
                summary = " | ".join(lines[-4:])[:650]
                if ("TextTestResult" in result and "test" in str(arguments.get("path", "")).lower()):
                    obs += ("\n（错误定位：测试脚本很可能在 unittest.TestCase 子类中"
                            "覆盖了框架保留的 run(self, result) 方法。必须修改 test*.py："
                            "将命令执行辅助函数 def run(...) 改名为 def run_cli(...)，"
                            "同步替换 self.run(...) 调用，然后重新执行测试。"
                            "这不是 app.py 业务逻辑错误。）")
                obs += (f"\n（末尾关键错误：{summary}。"
                        "请定位并改动相应文件，严禁重复运行未修改的失败脚本。）")
            # V21 落盘追踪：失败工具的目标文件本任务从未写入 → 修改没落盘。
            # 7B 高发病：回复里贴出改好的代码却直接重跑旧文件（展示≠写入）
            stale = None
            if name in ("run_python", "read_file") and "path" in arguments:
                p = str(arguments["path"]).replace("\\", "/")
                if p not in written:
                    stale = p
            if stale is not None:
                obs += (f"\n（关键问题：文件「{stale}」在本次任务中从未被 write_file "
                        "写入或更新——你现在运行的仍是旧版本文件。你在回复里展示的"
                        "修改代码并没有落盘。请先 ACTION write_file 把完整新代码写入"
                        f"「{stale}」，再 ACTION run_python 重新执行。）")
            elif (isinstance(arguments.get("content"), str)
                  and "```" in arguments["content"] and name != "write_file"):
                pass        # content 参数里带代码块属正常调用，不算未落盘
                # （必须 isinstance 判型：search_files 也有 content 参数，是布尔）
            log.warning("执行失败: %s %s", name, brief(result, 120))
        else:
            # 成功：技能规范以「指令」呈现；普通工具附目标核对提醒
            if _is_skill_tool(name):
                obs = (f"SKILL（操作规范，请按它执行本任务）：\n{result}"
                       "\n\n（已加载技能规范。请严格按照上面技能规范中的步骤执行"
                       "本任务，规范要求调用的工具要照常调用；完成后再给出 ANSWER。）")
            else:
                obs = f"OBSERVATION: {result}"
                if name in ("run_python", "run_command"):
                    # 执行类工具成功 → 提醒对照任务目标自检
                    obs += ("\n\n（本次执行成功。请对照用户任务目标检查输出："
                            "已达成 → ANSWER 总结结果；未达成 → 继续下一步。）")
                if step >= 1:
                    # 小模型补偿：多步推理时注入任务提醒，防止跑偏/重复调用/忘了 ANSWER
                    obs += (f"\n\n（提醒：用户任务是「{brief(task, 60)}」。"
                            "若已收集到足够信息，请直接 ANSWER 总结；"
                            "不要重复调用相同工具。）")
        # ---------- 重复调用熔断（V31） ----------
        if repeat >= self.REPEAT_NUDGE:
            obs += (f"\n\n⚠️ 重复调用警告：这是与上一轮完全相同的调用（已连续第 "
                    f"{repeat} 次），同样的调用只会得到同样的结果。请立即改变策略："
                    "检查参数是否用对、换用其他工具，或直接 ANSWER 总结已知信息——"
                    "不要再重复这个调用。")
            log.warning("重复调用熔断: %s 连续第 %d 次", name, repeat)
        elif streak >= self.TOOL_STREAK_NUDGE and self._result_unproductive(result):
            hint = ("联网已多次尝试仍无有效结果，请改用本地工具"
                    "（search_files/kb_search/read_file）或基于已知信息直接 "
                    "ANSWER，不要继续联网。" if name == "web_collect" else
                    "这个方向看起来没有产出，请换用其他工具或换一种思路。")
            obs += (f"\n\n⚠️ 方向警告：你已连续 {streak} 次调用同一工具「{name}」"
                    f"（即使每次参数不同），且结果都没有产出。{hint}")
            log.warning("同工具刷屏熔断: %s 连续第 %d 次", name, streak)
        return obs

    # ---------- 内部 ----------
    def _auto_skill_hint(self, task: str, cfg=None) -> str:
        """V30 技能自动匹配：任务与技能强匹配时返回规范正文（否则空串）

        参考 Jarvis「固定流程筛选必要规则」阶段：小模型常漏掉「该调哪个
        技能工具」，主动把规范注入任务消息比指望它自己发现可靠得多。
        """
        try:
            from core.skills import auto_inject_text, auto_match, discover
            matched = auto_match(discover(cfg or self._config(), self.workspace),
                                 task)
            if not matched:
                return ""
            log.info("技能自动匹配: %s", ", ".join(s.name for s in matched))
            return "\n\n" + auto_inject_text(matched)
        except Exception as e:
            log.warning("技能自动匹配失败: %s", e)
            return ""

    @staticmethod
    def _reasoning_needs_protocol(text: str) -> bool:
        """兜底得到的思考过程里是否表达了「要调工具/要按协议输出」的意图。

        用于判断"思考里其实是想执行动作"——这种情况该追问协议而不是把
        思考原样当答案给用户（gpt-oss 思考通道未分离时的典型表现）。"""
        t = str(text).lower()
        if "action:" in t or "answer:" in t:
            return True                    # 有协议痕迹却没解析成功 → 再问一次
        tools = ("list_files", "read_file", "write_file", "kb_search",
                 "web_collect", "run_python", "run_command", "search_files",
                 "create_dir", "delete_file", "append_file")
        if any(w in t for w in tools):
            return True                    # 明确提到要调某个工具
        intents = ("我应该", "让我", "接下来", "先查", "需要调用", "下一步",
                   "let's ", "i should", "i will", "use ", "need to")
        return any(k in t for k in intents)

    # 答案前缀残留：模型常加 "ANSWER:" / "**答：**" / "**ANSWER**" 等协议标记
    _ANS_PREFIX_RE = re.compile(
        r"^\s*\*{2}\s*(?:ANSWER|Answer|answer|最终答案|答案|答)\s*(?:[:：]\s*)?\*{2}\s*"
        r"|^\s*\*{0,2}\s*(?:ANSWER|Answer|answer|最终答案|答案|答)\s*\*{0,2}\s*[:：]\s*")

    @staticmethod
    def _clean_answer(text: str) -> str:
        """清洗答案前缀：模型常在正文前加 ANSWER: / **答：** 等协议残留"""
        t = str(text).strip()
        t = Agent._ANS_PREFIX_RE.sub("", t, count=1).strip()
        return t or str(text).strip()

    # 结果"没有产出"的标记（V31.2）：连续用同一工具但每次都空手而归时
    # 才提示换方向——有命中的连续搜索（如换关键词找符号）是正常策略，不打断
    _UNPRODUCTIVE_MARKERS = ("未找到", "未获取到", "命中 0", "无匹配",
                             "(空目录)", "(空文件)", "没有找到")

    @staticmethod
    def _result_unproductive(result: str) -> bool:
        """工具结果是否"没有产出"（空结果/未命中）"""
        r = str(result or "")
        return any(m in r for m in Agent._UNPRODUCTIVE_MARKERS)

    @staticmethod
    def _execution_failed(name: str, result: str) -> bool:
        """执行结果成败判定（确定性规则，不依赖模型智力）：
        工具层错误 / 脚本非零退出 / Traceback = 失败"""
        if result.startswith("错误: 用户已拒绝"):
            return True         # 拒绝执行不能计为操作成功，更不能计为测试通过
        if result.startswith("错误") or result.startswith("OBSERVATION: 错误"):
            return True
        if name == "run_command":
            match = re.search(r"(?:^|\n)exit_code=(-?\d+)", result)
            if match and int(match.group(1)) != 0:
                return True
        if name in ("run_python", "run_command"):
            if "[退出码" in result and "退出码 0]" not in result:
                return True
            if "Traceback" in result:
                return True
        return False

    def _config(self) -> Config:
        cfg = getattr(self.llm, "cfg", None)
        return cfg if cfg is not None else Config()

    def _refresh_system_prompt(self, task: str = ""):
        if self.workspace is not None:
            ws_info = (f"当前已挂载工作区(绝对路径): {self.workspace.root}\n"
                       "所有文件操作都被限制在该目录内。")
        else:
            ws_info = ("当前未挂载工作区，文件类工具不可用(会返回错误)。"
                       "可直接回答用户，或提示用户挂载工作区。")
        if self.qa_mode:                       # V25 仅提问模式：无工具清单、无协议
            prompt = SYSTEM_PROMPT_QA.format(workspace_info=ws_info)
        else:
            prompt = SYSTEM_PROMPT.format(
                workspace_info=ws_info,
                tools_description=tools_description(self._config(),
                                                    self.workspace, task))
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = {"role": "system", "content": prompt}
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def _append_assistant_history(self, full: str):
        """助手回复存入历史：超长内容截断（UI 与事件流仍显示全文）。

        实测 7B 模型上下文惯性显著——历史里堆积数千字教程式长文会稀释
        ACTION 协议引导，让模型越来越倾向直答而非调工具。"""
        if len(full) > self.HISTORY_TRUNCATE:
            self.messages.append({"role": "assistant",
                                  "content": full[:self.HISTORY_TRUNCATE]
                                  + "\n...(长回复已截断存储)"})
        else:
            self.messages.append({"role": "assistant", "content": full})

    def _trim_history(self, keep: int = 24):
        """按完整对话单元裁剪历史，绝不拆开 assistant.tool_calls/tool 结果组。

        system 与当前会话首条 user 上下文保留；其余历史从最旧单元开始淘汰。
        原生工具调用的 assistant 消息及其对应 tool 回复是一个不可拆分单元。
        """
        if len(self.messages) <= 2:
            return

        def _size(msg):
            return (len(msg.get("content") or "")
                    + len(json.dumps(msg.get("tool_calls") or [], ensure_ascii=False)))

        def _drop_oldest_unit():
            if len(self.messages) <= 2:
                return False
            first = self.messages[2]
            calls = first.get("tool_calls") or []
            if first.get("role") == "assistant" and calls:
                ids = {c.get("id") for c in calls if c.get("id")}
                end = 3
                seen = set()
                while end < len(self.messages) and self.messages[end].get("role") == "tool":
                    tid = self.messages[end].get("tool_call_id")
                    if ids and tid not in ids:
                        break
                    if tid:
                        seen.add(tid)
                    end += 1
                    if ids and ids.issubset(seen):
                        break
                del self.messages[2:end]
            elif first.get("role") == "tool":
                # 旧数据或异常历史中的孤立 tool 消息，按连续 tool 块整体清除。
                while len(self.messages) > 2 and self.messages[2].get("role") == "tool":
                    self.messages.pop(2)
            else:
                self.messages.pop(2)
            return True

        total = sum(_size(m) for m in self.messages)
        while (len(self.messages) > keep + 1 or total > self.KEEP_CHARS):
            before = len(self.messages)
            removed = self.messages[2:]
            if not _drop_oldest_unit():
                break
            total -= sum(_size(m) for m in removed[:before - len(self.messages)])

    # ---------- 解析（小模型友好：本地修复常见 JSON 病，避免浪费重试轮次） ----------
    @staticmethod
    def _repair_json(s: str):
        """修复小模型常见的 JSON 病态：围栏/单引号/尾逗号/未闭合括号"""
        s = re.sub(r"```(?:json)?", "", s).strip()        # 代码围栏
        s = re.sub(r",\s*([}\]])", r"\1", s)              # 尾逗号
        # 单引号字符串 → 双引号（loads 失败后的兜底路径才走到这里）
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r"'([^'\\\n]*)'", r'"\1"', s)
        if fixed != s:
            s = fixed
            try:
                json.loads(s)
                return s
            except json.JSONDecodeError:
                s = re.sub(r",\s*([}\]])", r"\1", s)      # 单引号修复后再补刀尾逗号
        # 未闭合大括号：按数量补齐
        opens, closes = s.count("{"), s.count("}")
        if opens > closes:
            s += "}" * (opens - closes)
        return s

    @staticmethod
    def _balanced_json(seg: str):
        """从 seg 中提取第一个括号平衡的 JSON 对象（字符串感知，content
        含 {} 的代码写入不误切）。返回 (obj, end) 或 None"""
        i = seg.find("{")
        if i < 0:
            return None
        depth, in_str, escape = 0, False, False
        for j in range(i, len(seg)):
            c = seg[j]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    cand = seg[i:j + 1]
                    try:
                        return json.loads(cand), j + 1
                    except json.JSONDecodeError:
                        return None      # 平衡但非法 → 交给上层试下一段
        return None

    @classmethod
    def _extract_action_json(cls, text: str):
        """逐段提取：回复可能平铺多个 ACTION（7B 常见——多步计划一条吐完），
        从第一个 ACTION: 起逐段解析，第一个合法的执行；每段还试本地修复。
        返回 (parsed, skipped, note)——skipped = 前面解析失败被跳过的段数，
        note = 解析路径说明（调试层展示）"""
        skipped = 0
        for m in re.finditer(r"ACTION\s*:", text):
            seg = text[m.end():]
            for variant, tag in cls._repair_variants(seg):
                try:
                    obj = json.loads(variant)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    name, arguments = obj.get("name"), obj.get("arguments") or {}
                    if isinstance(name, str) and name and isinstance(arguments, dict):
                        note = f"第 {skipped + 1} 个 ACTION 段解析成功（{tag}）"
                        return {"type": "action", "name": name.strip(),
                                "arguments": arguments}, skipped, note
            skipped += 1
        return None, skipped, f"共 {skipped} 个 ACTION 段均解析失败"

    @classmethod
    def _repair_variants(cls, seg):
        """对片段的 {…} 候选做修复变体 → [(候选, 说明标签), …]，
        边界取括号平衡处（第一个 } 截断会切碎含代码的 content）"""
        i = seg.find("{")
        if i < 0:
            return []
        got = cls._balanced_json(seg)
        if got:
            _, end = got
            raw = seg[i:end]
        else:
            # 未闭合：取段尾（到下一个 ACTION: 或文本末）再补括号
            nxt = seg.find("ACTION:", i)
            raw = seg[i:nxt] if nxt > 0 else seg[i:]
            raw = raw.rstrip().rstrip(",")
        out = [(raw, "原样合法")]
        de_triple = cls._de_triple_quote(raw)
        if de_triple != raw:
            out.append((de_triple, "三引号修复"))
            out.append((cls._repair_json(de_triple), "三引号修复+JSON修复"))
        out.append((cls._repair_json(raw), "JSON修复"))
        return out

    @staticmethod
    def _de_triple_quote(raw: str):
        """Python 三引号 content 修复：7B 写大段代码时习惯用 \"\"\"…\"\"\"
        包裹 JSON 字符串值（JSON 里非法）——把三引号体按纯文本重新转义"""
        def repl(m):
            return f'{m.group(1)}: {json.dumps(m.group(2))}'
        return re.sub(r'("[^"]*")\s*:\s*"""(.*?)"""',
                      repl, raw, flags=re.S)

    @staticmethod
    def _parse(text: str):
        # 1. 开头（剥围栏后）即 ANSWER: → 看两种情况：
        #    a. 长总结（前缀长）→ 最终回答（内文引用 ACTION 示例不影响）
        #    b. 短过渡语 + 紧随 ACTION（"没有文件，接下来我们将创建…ACTION:"）
        #       → 实为行动计划，执行第一个 ACTION
        head = text.lstrip("`\n \t")
        if re.match(r"```[a-zA-Z]*\s*ANSWER\s*:", head) or head.startswith("ANSWER"):
            m = re.search(r"ANSWER:\s*(.*?)ACTION\s*:", text, re.S)
            prefix = m.group(1).strip() if m else None
            if prefix is not None and len(prefix) < 200:
                parsed, skipped, note = Agent._extract_action_json(text)
                if parsed:
                    parsed["skipped"] = skipped
                    parsed["note"] = (f"ANSWER 短前缀({len(prefix)}字)→判定为行动计划；"
                                      + note)
                    return parsed                      # 短过渡语 → 执行行动
            m2 = re.search(r"ANSWER:\s*(.*)", text, re.S)
            if m2:
                return {"type": "answer", "content": m2.group(1).strip(),
                        "note": "ANSWER: 开头 → 最终回答"}
        # 2. 宽松定位 ACTION:（可能夹解释文字/围栏/多个并排）→ 逐段提取
        if re.search(r"\bACTION\s*:", text):
            parsed, skipped, note = Agent._extract_action_json(text)
            if parsed:
                parsed["skipped"] = skipped
                parsed["note"] = note
                return parsed
            return {"type": "bad_action",
                    "note": "检测到 ACTION: 但所有段 JSON 解析失败"}
        m = re.search(r"ANSWER:\s*(.*)", text, re.S)
        if m:
            return {"type": "answer", "content": m.group(1).strip(),
                    "note": "内文 ANSWER: → 最终回答"}
        return None
