"""技能（Skill）系统：可复用的操作规范包

一个技能 = 一个目录，内含 ``SKILL.md``：

    skills/pdf-report/
        SKILL.md          # 元数据(frontmatter) + 操作规范正文
        references/...    # 可选：附带资料（load_skill 可按需读取）
        scripts/...       # 可选：附带脚本

格式沿用主流 Agent Skills 约定（与 Claude / WorkBuddy / Jarvis 的 SKILL.md
同款），关键设计是**渐进式披露（progressive disclosure）**：

  1. 常驻上下文的只有「技能名 + 一句话描述」——几十个技能也只占几百字；
  2. 按任务**预筛**出候选技能（关键词命中分），只把它们暴露成工具，
     避免工具列表被几十个技能淹没；
  3. 模型判断需要时才调用工具 / load_skill 取回完整规范；
  4. 规范正文以「技能指令」形式注入对话，模型据此执行（而非当作观察数据）。

发现位置（参考 Jarvis 的多源 + 生态兼容目录设计）：

  | 来源前缀    | 路径                                       |
  |------------|--------------------------------------------|
  | project    | ``<工程>/skills/``                          |
  | project    | ``<工程>/.agents/skills/``（agentskills 标准）|
  | project    | ``<工程>/.claude/skills/``（Claude 兼容）   |
  | workspace  | ``<工作区>/.pyagent/skills/``                |
  | global     | ``~/.pyagent/skills/``                      |
  | global     | ``~/.agents/skills/``（agentskills 标准）    |

同名技能按上表顺序后者覆盖前者（工程 > 工作区 > 用户）。

发现结果按「技能目录 mtime 签名」缓存：每轮对话都会调 discover()，
无节制地重读所有 SKILL.md 是纯 IO 浪费；签名变了才重新扫描。

无第三方依赖：frontmatter 用内置极简解析器（仅需 key: value / 行内列表）。
"""
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.log import get_logger

log = get_logger("core.skills")

SKILL_FILENAME = "SKILL.md"
MAX_BODY_CHARS = 12_000      # 规范正文上限（超长截断，防上下文爆炸）
MAX_FILE_CHARS = 40_000      # 附属文件读取上限
MAX_LIST_CHARS = 4_000       # 技能清单上限（注入提示词的部分）
MAX_PRESELECT = 8            # 预筛后暴露成工具的技能上限（Jarvis 同思路：控制工具列表规模）

# 行内列表 [a, b, c]
_INLINE_LIST = re.compile(r"^\[(.*)\]$", re.S)
# 简易 frontmatter 块：--- 起止
_FM_BLOCK = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


@dataclass
class Skill:
    """一个已发现的技能"""
    name: str
    description: str
    path: Path                 # SKILL.md 完整路径
    root: Path                 # 技能目录
    version: str = ""
    meta: dict = field(default_factory=dict)   # frontmatter 全量键值
    body: str = ""             # SKILL.md 正文（不含 frontmatter）
    source: str = ""           # 来源前缀：project / workspace / global

    @property
    def label(self) -> str:
        """带来源前缀的标识（同名多源时便于区分）"""
        return f"{self.source}:{self.name}" if self.source else self.name

    def allowed_tools(self) -> list:
        """技能声明的 allowed-tools（frontmatter），空表示不限制"""
        v = self.meta.get("allowed-tools") or self.meta.get("allowed_tools") or []
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        return [str(x) for x in v] if isinstance(v, list) else []


# ------------------------------------------------------------------
# frontmatter 极简解析（无 PyYAML 依赖）
# ------------------------------------------------------------------
def _scalar(v: str):
    """标量解析：去引号 / 行内列表 / 布尔"""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    m = _INLINE_LIST.match(v)
    if m:
        inner = m.group(1).strip()
        if not inner:
            return []
        return [_scalar(x) for x in inner.split(",") if x.strip()]
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    return v


def parse_frontmatter(text: str):
    """拆出 frontmatter 键值与正文

    支持：key: value / key: "value" / key: [a, b] / key: 后跟 `  - item` 块列表。
    无法解析时返回 ({}, 原文)——正文仍可当普通规范用。"""
    if not text:
        return {}, ""
    m = _FM_BLOCK.match(text)
    if not m:
        return {}, text
    raw, body = m.group(1), text[m.end():]
    meta, cur_key = {}, None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and cur_key:
            # 块列表项：追加到当前 key
            cur = meta.get(cur_key)
            if not isinstance(cur, list):
                cur = []
            cur.append(_scalar(stripped[2:]))
            meta[cur_key] = cur
            continue
        if ":" not in stripped:
            continue
        k, _, v = stripped.partition(":")
        k = k.strip().lower()
        if not k:
            continue
        if v.strip() == "":
            meta[k] = []          # 可能是块列表，等下一行 `- `
            cur_key = k
        else:
            meta[k] = _scalar(v)
            cur_key = None
    # 空的块列表键没等到任何 item → 视为无值
    for k, v in list(meta.items()):
        if isinstance(v, list) and not v:
            meta[k] = ""
    return meta, body


# ------------------------------------------------------------------
# 发现
# ------------------------------------------------------------------
def skill_roots(cfg=None, ws=None) -> list:
    """技能搜索根目录 [(路径, 来源前缀)]

    参考 Jarvis 的多源发现 + 生态兼容目录设计：agentskills（.agents/skills）
    与 Claude（.claude/skills）生态产出的技能可直接复用，无需转换格式。

    顺序即优先级（后者覆盖同名者）：project > workspace > global
    """
    project = Path(__file__).resolve().parent.parent
    roots = [
        (project / "skills", "project"),
        (project / ".agents" / "skills", "project"),
        (project / ".claude" / "skills", "project"),
    ]
    if ws is not None:
        try:
            roots.append((Path(ws.root) / ".pyagent" / "skills", "workspace"))
            roots.append((Path(ws.root) / ".agents" / "skills", "workspace"))
        except Exception:
            pass
    home = os.path.expanduser("~")
    if home and home != "~":
        roots.append((Path(home) / ".pyagent" / "skills", "global"))
        roots.append((Path(home) / ".agents" / "skills", "global"))
    return roots


def _read_skill_md(path: Path, root: Path, source: str = "") -> Skill:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta, body = parse_frontmatter(text)
    name = str(meta.get("name") or path.parent.name).strip()
    desc = str(meta.get("description") or "").strip()
    if not desc:
        # 没写 description：取正文首个非空行（Markdown 标题/句子）当摘要
        for line in body.splitlines():
            s = line.strip().lstrip("#").strip()
            if s:
                desc = s[:120]
                break
    desc = desc.replace("\n", " ").strip()
    return Skill(name=name, description=desc, path=path, root=path.parent,
                 version=str(meta.get("version") or ""), meta=meta, body=body,
                 source=source)


# 发现结果缓存：(根目录签名) → Skill 列表。每轮 run() 都会调 discover()，
# 命中缓存可省掉重读所有 SKILL.md 的磁盘 IO。
_DISCOVER_CACHE = {"key": None, "skills": []}


def _roots_signature(roots: list):
    """各根目录下「技能名 + SKILL.md mtime」的签名，用于判断是否需要重扫"""
    sig = []
    for root, source in roots:
        try:
            if not root.is_dir():
                sig.append((str(root), source, None))
                continue
            entries = []
            for d in sorted(root.iterdir()):
                if not d.is_dir():
                    continue
                for fname in (SKILL_FILENAME, "skill.md"):
                    cand = d / fname
                    if cand.is_file():
                        try:
                            mtime = cand.stat().st_mtime_ns
                        except OSError:
                            mtime = 0
                        entries.append((d.name, fname, mtime))
                        break
            sig.append((str(root), source, tuple(entries)))
        except OSError:
            sig.append((str(root), source, "error"))
    return tuple(sig)


def discover(cfg=None, ws=None) -> list:
    """扫描所有技能目录，返回 Skill 列表（同名后者覆盖前者）

    结果按根目录签名缓存：技能目录没变化时直接复用，不再重读 SKILL.md。"""
    if cfg is not None and not getattr(cfg, "skills_enabled", True):
        return []
    roots = skill_roots(cfg, ws)
    key = _roots_signature(roots)
    if _DISCOVER_CACHE["key"] == key:
        return list(_DISCOVER_CACHE["skills"])
    found = {}
    for root, source in roots:
        try:
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                if not d.is_dir():
                    continue
                md = None
                for fname in (SKILL_FILENAME, "skill.md"):
                    cand = d / fname
                    if cand.is_file():
                        md = cand
                        break
                if md is None:
                    continue
                try:
                    sk = _read_skill_md(md, root, source)
                except Exception as e:
                    log.warning("技能读取失败 %s: %s", md, e)
                    continue
                if sk.name in found:
                    log.info("技能同名覆盖: %s（%s → %s）", sk.name,
                             found[sk.name].path, md)
                found[sk.name] = sk
        except Exception as e:
            log.warning("技能目录扫描失败 %s: %s", root, e)
    skills = list(found.values())
    _DISCOVER_CACHE["key"], _DISCOVER_CACHE["skills"] = key, skills
    log.info("发现技能 %d 个: %s", len(skills),
             ", ".join(s.label for s in skills) or "(无)")
    return list(skills)


# ------------------------------------------------------------------
# 提示词/工具用：技能清单（渐进式披露的第一层）
# ------------------------------------------------------------------
def catalog(skills: list, limit: int = MAX_LIST_CHARS) -> str:
    """技能清单：名字 + 一句话描述。注入提示词/工具描述用。"""
    if not skills:
        return ""
    lines = []
    for s in skills:
        d = s.description or "(无描述)"
        lines.append(f"- {s.name}: {d}")
        if sum(len(x) for x in lines) > limit:
            break
    return "\n".join(lines)


def catalog_text(skills: list, limit: int = MAX_LIST_CHARS) -> str:
    """带表头的技能清单文本"""
    body = catalog(skills, limit)
    if not body:
        return ""
    return f"可用技能：\n{body}"


# ------------------------------------------------------------------
# 载入规范正文（第二层）
# ------------------------------------------------------------------
def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…（技能内容过长已截断，原文 {len(text)} 字）"


def load_body(skill: Skill) -> str:
    """技能规范正文（frontmatter 之后的 markdown）"""
    return _clip(skill.body.strip(), MAX_BODY_CHARS)


def read_bundle_file(skill: Skill, rel: str) -> str:
    """读取技能目录内的附属文件（references/、scripts/ 等），带越界保护"""
    rel = (rel or "").strip().replace("\\", "/")
    if not rel:
        raise ValueError("未指定文件路径")
    if rel.startswith("/") or ".." in rel.split("/"):
        raise ValueError("非法路径（仅允许技能目录内的相对路径）")
    target = (skill.root / rel).resolve()
    try:
        target.relative_to(skill.root.resolve())
    except ValueError:
        raise ValueError("非法路径（越出技能目录）")
    if not target.is_file():
        raise ValueError(f"技能 {skill.name} 中不存在文件: {rel}")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise ValueError(f"读取失败: {e}")
    return _clip(text, MAX_FILE_CHARS)


def bundle_files(skill: Skill, limit: int = 20) -> list:
    """列出技能目录内的附属文件（相对路径），供模型知道还能读什么"""
    out = []
    try:
        for p in sorted(skill.root.rglob("*")):
            if p.is_file() and p.name.lower() != "skill.md":
                out.append(str(p.relative_to(skill.root)).replace("\\", "/"))
                if len(out) >= limit:
                    break
    except Exception:
        pass
    return out


def find(skills: list, name: str):
    """按名称查技能（不区分大小写，允许省略连字符/下划线差异）"""
    n = (name or "").strip()
    if not n:
        return None
    for s in skills:
        if s.name == n:
            return s
    key = re.sub(r"[\s_\-]+", "", n.lower())
    for s in skills:
        if re.sub(r"[\s_\-]+", "", s.name.lower()) == key:
            return s
    return None


# ------------------------------------------------------------------
# 工具接入：load_skill
# ------------------------------------------------------------------
LOAD_SKILL_DESC = (
    "加载技能（可复用的操作规范）全文。当任务涉及某个技能的专业流程/规范"
    "（如特定文档格式、报告模板、既定工作流程）时，先加载对应技能，再按其"
    "规范执行。传入技能名；也可传 path 读取技能自带的参考文件。"
)


def catalog_for_prompt(skills: list) -> str:
    """注入提示词/工具描述的技能清单（第一层披露：只有名字+描述）"""
    body = catalog(skills)
    if not body:
        return ""
    return ("可用技能（load_skill 加载全文）：\n" + body)


# ------------------------------------------------------------------
# 任务预筛（参考 Jarvis 的关键词粗筛 + 模型选择两阶段）
# ------------------------------------------------------------------
# 中文按 2-gram 切分（无需 jieba 依赖），英文/数字按词切
_CJK = re.compile(r"[\u4e00-\u9fff]")
_SPLIT = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_STOP = frozenset("""
的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这
the a an of to in and or is are for with on at by be as it this that from not can will
用 把 被 让 给 从 到 做 使 对 而 但 并 及其 等 中 内 外 时 后 前
""".split())


def _tokens(text: str) -> list:
    """任务文本 → 关键词列表（中文 2-gram + 英文词）"""
    out = []
    for seg in _SPLIT.split(str(text or "")):
        if not seg:
            continue
        if _CJK.search(seg):
            # 连续中文段：滑窗 2-gram（单字段除外）
            if len(seg) == 1:
                out.append(seg)
            else:
                out.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            low = seg.lower()
            if len(low) >= 3 and low not in _STOP:
                out.append(low)
    return [t for t in out if t and t not in _STOP]


def score_skill(skill: Skill, tokens: list) -> int:
    """技能与任务的关键词命中分（Jarvis 同思路：命中即计分）"""
    if not tokens:
        return 0
    hay = f"{skill.name} {skill.description} {' '.join(skill.meta.get('tags') or [])}"
    hay = hay.lower()
    name_low = skill.name.lower()
    score = 0
    for t in tokens:
        if t in name_low:
            score += 3          # 命中技能名权重更高
        elif t in hay:
            score += 1
    return score


def preselect(skills: list, task: str, limit: int = MAX_PRESELECT) -> list:
    """按任务预筛技能（返回最相关的若干个，按分降序）

    Jarvis 的做法是「关键词粗筛 → 模型选择」两阶段。本实现把粗筛用于
    **控制暴露给模型的工具数量**（避免几十个技能把工具列表撑爆），
    模型仍可在候选中自行挑选，也可以用 load_skill 取任意技能。
    """
    if not skills:
        return []
    tokens = _tokens(task)
    if not tokens:
        return list(skills)[:limit]
    scored = [(score_skill(s, tokens), s) for s in skills]
    scored.sort(key=lambda x: x[0], reverse=True)
    # 只保留命中的（与 Jarvis 粗筛同语义：score>0 才进候选），
    # 再按上限截断——无关技能不进模型视野，避免干扰选型
    picked = [s for sc, s in scored if sc > 0]
    if not picked:
        # 全不命中：宁可少暴露也别给错技能（模型仍可用 load_skill 兜底）
        return list(skills)[:limit]
    return picked[:limit]


AUTO_SCORE_THRESHOLD = 3       # 自动注入阈值（命中分 ≥ 该值才主动加载）

def auto_match(skills: list, task: str, threshold: int = AUTO_SCORE_THRESHOLD) -> list:
    """强匹配技能（分数达阈值）——Jarvis「固定流程筛选必要规则」的等价物

    与 preselect 的区别：preselect 是「缩小候选」，auto_match 是「认准了
    就直接用」。命中技能时把规范正文主动注入上下文，模型无需自己发现
    该调用哪个工具——小模型尤其容易漏掉工具。
    """
    if not skills:
        return []
    tokens = _tokens(task)
    if not tokens:
        return []
    scored = [(score_skill(s, tokens), s) for s in skills]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for sc, s in scored if sc >= threshold]


def auto_inject_text(skills: list) -> str:
    """自动注入的技能规范文本（强制语气——它不是「参考」，是「必须照做」）"""
    if not skills:
        return ""
    names = "、".join(s.name for s in skills)
    parts = [f"【本次任务已自动匹配技能规范：{names}】\n"
             f"以下规范由系统按任务内容自动匹配，**必须严格遵守**，"
             f"不要按自己的习惯另写一套流程：\n"]
    for s in skills:
        parts.append(f"\n--- 技能「{s.name}」规范 ---\n{load_body(s)}")
    parts.append(f"\n（执行完上述规范要求的工作后，再给出最终答案。）")
    return "\n".join(parts)


def load_skill_text(skill: Skill, path: str = "") -> str:
    """技能载入结果文本

    明确框成「操作规范」而非普通观察数据——模型据此改变执行方式。
    """
    if path:
        try:
            content = read_bundle_file(skill, path)
        except ValueError as e:
            return f"错误: {e}"
        return (f"【技能附件：{skill.name}/{path}】\n{content}\n"
                f"（以上是技能 {skill.name} 的附带资料，继续按该技能规范执行）")
    files = bundle_files(skill)
    tail = ""
    if files:
        tail = ("\n\n（该技能附带以下文件，需要时可用 load_skill 传 path 读取："
                + "、".join(files) + "）")
    allowed = skill.allowed_tools()
    limit = ""
    if allowed:
        limit = ("\n（本技能限定使用以下工具：" + "、".join(allowed) + "）")
    return (f"【技能规范：{skill.name}】\n"
            f"以下是「{skill.name}」的完整操作规范，请严格按它执行本任务：\n\n"
            f"{load_body(skill)}{limit}{tail}")


def dispatch(cfg=None, ws=None, name: str = "", path: str = "") -> str:
    """load_skill 工具的落地实现（工具层调用）"""
    skills = discover(cfg, ws)
    if not skills:
        return ("错误: 未发现任何技能。技能是放在 <工程>/skills/<技能名>/SKILL.md "
                "或 ~/.pyagent/skills/ 下的目录（含 SKILL.md 的文件夹）。")
    if not (name or "").strip():
        return ("错误: 未指定技能名。可用技能：\n"
                + catalog(skills))
    skill = find(skills, name)
    if skill is None:
        return (f"错误: 未找到技能「{name}」。可用技能：\n" + catalog(skills))
    log.info("加载技能: %s（来自 %s）", skill.name, skill.path)
    return load_skill_text(skill, path)


# ------------------------------------------------------------------
# 技能 → 工具暴露（每个技能一个工具，模型可直接选用）
# ------------------------------------------------------------------
TOOL_PREFIX = "skill_"
_SLUG_BAD = re.compile(r"[^0-9a-zA-Z_]+")


def slug(name: str) -> str:
    """技能名 → 合法工具名片段（小写字母/数字/下划线）"""
    s = _SLUG_BAD.sub("_", str(name or "").strip().lower()).strip("_")
    return s or "unnamed"


def tool_name(skill) -> str:
    """技能对应的工具名：skill_<slug>"""
    return TOOL_PREFIX + slug(skill.name)


def tool_entries(cfg=None, ws=None, task: str = "") -> list:
    """已发现技能 → 工具条目列表（供原生 function calling 用）

    每个技能暴露为一个独立工具：模型像挑工具一样挑技能，比「读一段目录
    再调用通用加载器」更稳（实测原生协议下模型对 tools 的选择明显更可靠）。

    task: 当前任务描述——非空时按关键词预筛，只暴露最相关的若干个，
    避免技能一多就把模型的工具列表撑爆（参考 Jarvis 的两阶段选择）。
    """
    skills = discover(cfg, ws)
    if task:
        skills = preselect(skills, task)
    out = []
    for s in skills:
        desc = s.description or f"加载「{s.name}」技能的操作规范"
        out.append({
            "type": "function",
            "function": {
                "name": tool_name(s),
                "description": f"[技能] {desc}",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            "_skill": s,
        })
    return out


def resolve_tool(tool_nm: str, cfg=None, ws=None):
    """工具名 → 技能对象（非技能工具返回 None）"""
    if not tool_nm or not tool_nm.startswith(TOOL_PREFIX):
        return None
    skills = discover(cfg, ws)
    want = slug(tool_nm[len(TOOL_PREFIX):])
    for s in skills:
        if slug(s.name) == want:
            return s
    return None


# ------------------------------------------------------------------
# 技能安装（参考 Jarvis 的 git clone 安装器）
# ------------------------------------------------------------------
def install_from_git(url: str, subdir: str = "", dest_root=None,
                      timeout: int = 120) -> dict:
    """从 git 仓库安装技能（clone 后定位 SKILL.md 复制到技能目录）

    流程（对齐 Jarvis SkillInstaller）：
      clone --depth 1 → 定位 <subdir>/SKILL.md → 复制整个技能目录到
      目标根目录 → 返回安装信息。

    dest_root: 安装目标根目录（默认 <工程>/skills）。
    返回 {"ok": bool, "message": str, "path": str, "name": str}
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "message": "未提供仓库地址", "path": "", "name": ""}
    project = Path(__file__).resolve().parent.parent
    dest_root = Path(dest_root) if dest_root else project / "skills"
    dest_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="skill_install_"))
    try:
        log.info("安装技能: git clone %s", url)
        r = subprocess.run(["git", "clone", "--depth", "1", url, str(tmp / "repo")],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            err = (r.stderr or "").strip()[-300:]
            return {"ok": False, "message": f"git clone 失败: {err}",
                    "path": "", "name": ""}
        repo = tmp / "repo"
        # 定位 SKILL.md：优先 subdir，其次自动搜索（限深度 3）
        skill_md = None
        if subdir:
            cand = repo / subdir / "SKILL.md"
            if cand.is_file():
                skill_md = cand
        if skill_md is None:
            cands = [p for p in repo.rglob("SKILL.md")
                     if len(p.relative_to(repo).parts) <= 4]
            if not cands:
                cands = [p for p in repo.rglob("skill.md")
                         if len(p.relative_to(repo).parts) <= 4]
            if cands:
                skill_md = sorted(cands, key=lambda p: len(p.parts))[0]
        if skill_md is None:
            return {"ok": False,
                    "message": "仓库中未找到 SKILL.md（技能需以 SKILL.md 为入口）",
                    "path": "", "name": ""}
        src_dir = skill_md.parent
        meta, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8",
                                                        errors="replace"))
        name = str(meta.get("name") or src_dir.name).strip()
        name = re.sub(r"[^\w\-.]+", "-", name).strip("-") or src_dir.name
        target = dest_root / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(src_dir, target)
        _DISCOVER_CACHE["key"] = None       # 安装后失效缓存，下次 discover 重扫
        log.info("技能安装完成: %s → %s", name, target)
        return {"ok": True, "message": f"已安装技能「{name}」",
                "path": str(target), "name": name}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": f"git clone 超时({timeout}s)",
                "path": "", "name": ""}
    except Exception as e:
        log.warning("技能安装异常: %s", e)
        return {"ok": False, "message": f"安装异常: {e}", "path": "", "name": ""}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
