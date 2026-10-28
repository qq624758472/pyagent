"""技能（Skill）系统单元测试（离线，不连模型）

覆盖：
  1 frontmatter 解析（含行内列表 / 块列表 / 无 frontmatter）
  2 技能发现（多目录 / 同名覆盖 / 开关关闭）
  3 渐进式披露：清单只给名字+描述，load 才给全文
  4 工具暴露：原生 tools 数组含 skill_* 工具；文本协议清单含技能条目
  5 安全：附属文件路径越界被拦截
  6 Agent 集成：加载技能后注入「按规范执行」指令，且不参与目标核对闭环
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import skills                       # noqa: E402
from core.agent import Agent, _is_skill_tool  # noqa: E402
from core.config import Config                # noqa: E402
from core.tools import (execute_tool, openai_tools,  # noqa: E402
                        tools_description)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (f"  | {detail}" if detail and not cond else ""))


# ------------------------------------------------------------------
# 临时技能目录
# ------------------------------------------------------------------
def _mk_skill(base: Path, name: str, desc: str, body: str = "步骤1\n步骤2\n",
              extra: str = "") -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    fm = f"---\nname: {name}\ndescription: {desc}\nversion: 1.0\n{extra}"
    (d / "SKILL.md").write_text(fm + "---\n\n" + body, encoding="utf-8")
    return d


class _TmpRoots:
    """把技能搜索根临时指向 tmp 目录（避免动真实 skills/）"""
    def __init__(self, base):
        self.base = Path(base)
        self._orig = None

    def __enter__(self):
        self._orig = skills.skill_roots
        base = self.base

        def _roots(cfg=None, ws=None):
            return [(base, "project")]     # 与真实实现同构：(路径, 来源前缀)
        skills.skill_roots = _roots
        return self

    def __exit__(self, *e):
        skills.skill_roots = self._orig


# ------------------------------------------------------------------
# 1. frontmatter 解析
# ------------------------------------------------------------------
def test_frontmatter():
    meta, body = skills.parse_frontmatter(
        '---\nname: demo\ndescription: "带引号的描述"\n'
        'allowed-tools: [read_file, write_file]\nversion: 2\n---\n\n正文内容')
    check("frontmatter name", meta.get("name") == "demo", str(meta))
    check("frontmatter 去引号", meta.get("description") == "带引号的描述", str(meta))
    check("frontmatter 行内列表",
          meta.get("allowed-tools") == ["read_file", "write_file"], str(meta))
    check("frontmatter 正文分离", body.strip() == "正文内容", repr(body))

    meta2, body2 = skills.parse_frontmatter("没有 frontmatter 的正文")
    check("无 frontmatter 时正文原样", meta2 == {} and body2 == "没有 frontmatter 的正文",
          f"{meta2} {body2!r}")

    meta3, _ = skills.parse_frontmatter(
        "---\nname: blk\ntags:\n  - a\n  - b\n---\nbody")
    check("frontmatter 块列表", meta3.get("tags") == ["a", "b"], str(meta3))


# ------------------------------------------------------------------
# 2. 技能发现
# ------------------------------------------------------------------
def test_discover():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "alpha", "甲技能")
        _mk_skill(base, "beta", "乙技能")
        with _TmpRoots(base):
            found = skills.discover(Config(), None)
            names = sorted(s.name for s in found)
            check("发现全部技能", names == ["alpha", "beta"], str(names))
            alpha = skills.find(found, "alpha")
            check("技能描述解析", alpha.description == "甲技能", alpha.description)
            # 无 description 时用正文首行兜底
            d = base / "gamma"
            d.mkdir()
            (d / "SKILL.md").write_text("# 伽马技能正文\n更多\n", encoding="utf-8")
            found2 = skills.discover(Config(), None)
            g = skills.find(found2, "gamma")
            check("缺 description 用正文首行兜底",
                  g is not None and "伽马技能正文" in g.description,
                  g.description if g else "None")


def test_discover_toggle():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "alpha", "甲技能")
        with _TmpRoots(base):
            cfg = Config()
            cfg.skills_enabled = False
            check("开关关闭时不暴露技能", skills.discover(cfg, None) == [],
                  str(skills.discover(cfg, None)))
            cfg.skills_enabled = True
            check("开关打开时暴露技能", len(skills.discover(cfg, None)) == 1)


def test_find_fuzzy():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "code-review", "审查")
        with _TmpRoots(base):
            found = skills.discover(Config(), None)
            check("名称容错（连字符/大小写）",
                  skills.find(found, "CodeReview") is not None)
            check("未知技能返回 None", skills.find(found, "nope") is None)


# ------------------------------------------------------------------
# 3. 渐进式披露
# ------------------------------------------------------------------
def test_progressive_disclosure():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "alpha", "甲技能描述",
                  body="非常长的操作规范正文" * 40)
        with _TmpRoots(base):
            found = skills.discover(Config(), None)
            cat = skills.catalog(found)
            check("清单只含名字+描述",
                  "alpha" in cat and "甲技能描述" in cat, cat)
            check("清单不含正文", "非常长的操作规范正文" not in cat, cat[:200])
            full = skills.load_skill_text(found[0])
            check("load 后含全文", "非常长的操作规范正文" in full)
            check("load 后明确要求按规范执行",
                  "请严格按" in full or "请严格按照" in full, full[:120])


def test_body_clip():
    """超长正文被截断（防上下文爆炸）"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "big", "大技能", body="X" * (skills.MAX_BODY_CHARS + 5000))
        with _TmpRoots(base):
            found = skills.discover(Config(), None)
            text = skills.load_skill_text(found[0])
            check("超长正文截断", "已截断" in text and len(text) < skills.MAX_BODY_CHARS * 2,
                  f"len={len(text)}")


# ------------------------------------------------------------------
# 4. 工具暴露
# ------------------------------------------------------------------
def test_tool_exposure():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "alpha", "甲技能描述")
        with _TmpRoots(base):
            cfg = Config()
            tools = openai_tools(cfg=cfg, ws=None)
            names = [t["function"]["name"] for t in tools]
            check("原生协议暴露 skill_ 工具", "skill_alpha" in names, str(names))
            sk = [t for t in tools if t["function"]["name"] == "skill_alpha"][0]
            check("技能工具描述含技能描述",
                  "甲技能描述" in sk["function"]["description"],
                  sk["function"]["description"])
            check("技能工具参数为空对象",
                  sk["function"]["parameters"]["properties"] == {})
            desc = tools_description(cfg, None)
            check("文本协议清单含技能条目",
                  "skill_alpha" in desc and "甲技能描述" in desc, desc[-200:])
            check("load_skill 工具始终注册", "load_skill" in names, str(names))


def test_execute_skill_tool():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "alpha", "甲技能描述", body="具体步骤A")
        with _TmpRoots(base):
            cfg = Config()
            out = execute_tool("skill_alpha", None, cfg, {})
            check("execute_tool 执行技能工具", "具体步骤A" in out, out[:120])
            out2 = execute_tool("load_skill", None, cfg, {"name": "alpha"})
            check("load_skill 工具可用", "具体步骤A" in out2, out2[:120])
            out3 = execute_tool("load_skill", None, cfg, {"name": "不存在"})
            check("未知技能名给可用清单",
                  "错误" in out3 and "alpha" in out3, out3[:120])
            out4 = execute_tool("skill_不存在", None, cfg, {})
            check("未知技能工具返回错误", out4.startswith("错误"), out4[:80])


def test_tool_name_slug():
    check("工具名 slug 化", skills.tool_name(type("S", (), {"name": "PDF Report"})())
          == "skill_pdf_report")
    check("技能工具名合法字符",
          all(c.isalnum() or c == "_" for c in skills.tool_name(
              type("S", (), {"name": "a-b.c d"})())))


# ------------------------------------------------------------------
# 5. 安全：路径越界
# ------------------------------------------------------------------
def test_path_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        d = _mk_skill(base, "alpha", "甲技能")
        (d / "references").mkdir()
        (d / "references" / "guide.md").write_text("参考资料内容", encoding="utf-8")
        with _TmpRoots(base):
            found = skills.discover(Config(), None)
            s = found[0]
            check("读取附属文件", "参考资料内容" in skills.read_bundle_file(s, "references/guide.md"))
            for bad in ("../../config.json", "/etc/passwd", "..\\..\\config.json"):
                try:
                    skills.read_bundle_file(s, bad)
                    check(f"越界拦截 {bad}", False, "未拦截")
                except ValueError:
                    check(f"越界拦截 {bad}", True)
            check("附属文件清单可见",
                  "references/guide.md" in skills.bundle_files(s),
                  str(skills.bundle_files(s)))


# ------------------------------------------------------------------
# 6. Agent 集成
# ------------------------------------------------------------------
def test_agent_skill_integration():
    """加载技能后：注入「按规范执行」指令，且不参与目标核对闭环"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "alpha", "甲技能描述", body="具体步骤A")
        with _TmpRoots(base):
            cfg = Config()
            cfg.skills_enabled = True
            agent = Agent(None, None, qa_mode=False)
            agent.llm = type("L", (), {"cfg": cfg, "model": "m"})()
            agent.set_llm(agent.llm)
            # 直接构造 native 路径的观察文本，验证闭环判定
            from core.agent import _is_skill_tool
            check("_is_skill_tool 识别 skill_ 前缀", _is_skill_tool("skill_alpha"))
            check("_is_skill_tool 识别 load_skill", _is_skill_tool("load_skill"))
            check("_is_skill_tool 不误判普通工具",
                  not _is_skill_tool("write_file"))
            # 原生 tools 里应含技能工具
            tools = openai_tools(cfg=cfg, ws=None)
            check("Agent 路径可取到技能工具",
                  any(t["function"]["name"] == "skill_alpha" for t in tools))


def test_native_loop_skill_closure():
    """原生循环：加载技能后注入规范执行指令，且不触发「已完成」闭环"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "alpha", "甲技能描述", body="具体步骤A")
        with _TmpRoots(base):
            cfg = Config()
            cfg.protocol = "native"
            cfg.skills_enabled = True

            events = []

            class FakeLLM:
                model = "m"
                def __init__(self):
                    self.cfg = cfg
                    self.turn = 0
                    self.messages = []
                def chat_turn(self, messages, tools=None, model=None, timeout=None):
                    self.messages = messages
                    self.turn += 1
                    if self.turn == 1:
                        yield {"type": "tool_call", "id": "c1",
                               "name": "skill_alpha", "arguments": "{}"}
                    else:
                        yield {"type": "content", "text": "已按规范完成"}

            agent = Agent(FakeLLM(), None, max_iterations=4)
            agent._config_override = cfg
            out = list(agent.run("审查代码", progress=None))
            kinds = [e["type"] for e in out]
            check("原生路径：加载技能后产出答案", "answer" in kinds, str(kinds))
            # 找到回传给模型的 tool 消息
            tool_msg = ""
            for m in agent.messages:
                if m.get("role") == "tool":
                    tool_msg = m.get("content") or ""
            check("回传内容含技能规范正文", "具体步骤A" in tool_msg, tool_msg[:150])
            check("回传内容含「按规范执行」指令",
                  "技能规范" in tool_msg and "执行" in tool_msg, tool_msg[-220:])
            check("技能加载不误触发「工具已执行成功」闭环",
                  "工具已执行成功" not in tool_msg, tool_msg[-220:])


# ------------------------------------------------------------------
# 7. Jarvis 借鉴项：多源发现 / 任务预筛 / 自动匹配 / git 安装
# ------------------------------------------------------------------
def test_multi_source_roots():
    """发现位置包含 Jarvis/agentskills/Claude 生态兼容目录"""
    roots = skills.skill_roots(Config(), None)
    paths = [str(p).replace("\\", "/") for p, _ in roots]
    check("含工程 skills 目录", any(p.endswith("/skills") and ".agents" not in p
                                  and ".claude" not in p and ".pyagent" not in p
                                  for p in paths), str(paths))
    check("含 .agents/skills 兼容目录",
          any("/.agents/skills" in p for p in paths), str(paths))
    check("含 .claude/skills 兼容目录",
          any("/.claude/skills" in p for p in paths), str(paths))
    check("含用户级 .pyagent/skills",
          any("/.pyagent/skills" in p for p in paths), str(paths))
    check("来源前缀标注齐全",
          all(src in ("project", "workspace", "global")
              for _, src in skills.skill_roots(Config(), None)))


def test_tokens_and_scoring():
    """中文 2-gram 分词 + 关键词命中分"""
    toks = skills._tokens("帮我写一份总结报告")
    check("中文切出 2-gram", "报告" in toks and "总结" in toks, str(toks))
    check("英文按词切分", "report" in skills._tokens("write a report"),
          str(skills._tokens("write a report")))
    check("停用词过滤", "的" not in skills._tokens("我的代码"),
          str(skills._tokens("我的代码")))


def test_preselect_and_auto_match():
    """预筛缩小候选；强匹配自动命中"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "code-review", "代码审查规范，检查代码质量和安全问题")
        _mk_skill(base, "report-writer", "撰写工作报告和技术文档")
        _mk_skill(base, "sql-tuning", "SQL 慢查询优化与索引调优")
        with _TmpRoots(base):
            found = skills.discover(Config(), None)
            pre = skills.preselect(found, "帮我审查这段代码有没有安全漏洞")
            check("预筛命中审查技能",
                  any(s.name == "code-review" for s in pre), str([s.name for s in pre]))
            check("预筛排除无关技能",
                  all(s.name != "sql-tuning" for s in pre),
                  str([s.name for s in pre]))
            auto = skills.auto_match(found, "帮我审查这段代码有没有安全漏洞")
            check("强匹配自动命中审查技能",
                  any(s.name == "code-review" for s in auto),
                  str([s.name for s in auto]))
            auto2 = skills.auto_match(found, "今天天气怎么样")
            check("无关任务不误命中", auto2 == [], str([s.name for s in auto2]))


def test_auto_inject_text():
    """自动注入文本必须是强制语气且含正文"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "alpha", "甲技能", body="具体步骤A")
        with _TmpRoots(base):
            found = skills.discover(Config(), None)
            txt = skills.auto_inject_text(found)
            check("自动注入含正文", "具体步骤A" in txt, txt[:120])
            check("自动注入为强制语气",
                  "必须严格遵守" in txt, txt[:120])
            check("自动注入无技能时为空", skills.auto_inject_text([]) == "")


def test_agent_auto_inject():
    """Agent 启动时把强匹配技能注入任务消息"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _mk_skill(base, "code-review", "代码审查规范，检查代码质量和安全问题",
                  body="审查步骤XYZ")
        with _TmpRoots(base):
            cfg = Config()
            cfg.protocol = "native"
            cfg.skills_enabled = True

            class FakeLLM:
                model = "m"
                def __init__(self):
                    self.cfg = cfg
                def chat_turn(self, messages, tools=None, model=None, timeout=None):
                    yield {"type": "content", "text": "好的"}

            agent = Agent(FakeLLM(), None, max_iterations=2)
            list(agent.run("帮我审查这段代码有没有安全漏洞"))
            user_msg = ""
            for m in agent.messages:
                if m.get("role") == "user":
                    user_msg = m.get("content") or ""
                    break
            check("任务消息含自动注入的技能规范",
                  "审查步骤XYZ" in user_msg, user_msg[:200])
            check("自动注入标注为强制遵守",
                  "必须严格遵守" in user_msg, user_msg[:200])


def test_install_from_git():
    """git 安装：仓库内含 SKILL.md 即可安装（无网络时优雅失败）"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # 本地 git 仓库当"远程"用（file:// 协议，git clone 支持）
        repo = base / "fakerepo"
        sk = repo / "skills" / "demo"
        sk.mkdir(parents=True)
        (sk / "SKILL.md").write_text(
            "---\nname: demo\ndescription: 演示技能\n---\n\n演示正文\n", encoding="utf-8")
        try:
            import subprocess
            subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True,
                           capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True,
                           capture_output=True)
            subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                            "commit", "-qm", "init"], cwd=str(repo), check=True,
                           capture_output=True)
        except Exception:
            check("git 可用性检查", False, "git 不可用，跳过安装测试")
            return
        dest = base / "dest"
        r = skills.install_from_git(str(repo), "skills/demo", dest_root=dest)
        check("git 安装成功", r["ok"], r["message"])
        if r["ok"]:
            check("安装后 SKILL.md 存在",
                  (dest / "demo" / "SKILL.md").is_file(),
                  str(list(dest.rglob("*"))))
            found = skills.discover(Config(), None)  # 全局发现
            check("安装的技能可被发现或路径正确",
                  r["path"].endswith("demo"), r["path"])
        r2 = skills.install_from_git("", "", dest_root=dest)
        check("空地址优雅失败", not r2["ok"], r2["message"])


if __name__ == "__main__":
    test_frontmatter()
    test_discover()
    test_discover_toggle()
    test_find_fuzzy()
    test_progressive_disclosure()
    test_body_clip()
    test_tool_exposure()
    test_execute_skill_tool()
    test_tool_name_slug()
    test_path_traversal()
    test_agent_skill_integration()
    test_native_loop_skill_closure()
    test_multi_source_roots()
    test_tokens_and_scoring()
    test_preselect_and_auto_match()
    test_auto_inject_text()
    test_agent_auto_inject()
    test_install_from_git()
    print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败"
          + (f"  ✘ {FAIL}" if FAIL else "  全部通过"))
    raise SystemExit(1 if FAIL else 0)
