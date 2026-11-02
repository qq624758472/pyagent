"""V31 修复冒烟测试：search_files 内容搜索 + 重复调用熔断

背景（10-28 日志实证）：gpt-oss-20B 找代码符号 g_eop_info 时，拿着普通
标识符连调 search_files 5 次（该工具当时只搜文件名），烧光 8 轮迭代。
桩 LLM + 真实工具执行（临时目录，不碰工程配置），风格对齐 test_robustness。
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent
from core.config import Config
from core.tools import execute_tool
from core.workspace import Workspace

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (f"  | {detail}" if detail and not cond else ""))


class StubLLM:
    """文本协议桩：chat_stream 按脚本逐字符产出"""
    def __init__(self, script):
        self.script = script
        self.cfg = Config(protocol="text")
        self.last_reasoning = ""
        self.calls = 0

    def chat_stream(self, messages, model=None):
        self.calls += 1
        step = self.script[min(self.calls - 1, len(self.script) - 1)]
        yield from step


# 准备临时工作区：模拟 ACS 目录里的 C 源码
tmp = Path(tempfile.mkdtemp())
ws = Workspace(tmp)
(tmp / "acs_main.c").write_text(
    '#include "acs.h"\nstatic eop_info_t g_eop_info;\nint main(void) { return 0; }\n',
    encoding="utf-8")
(tmp / "notes.md").write_text("# 笔记\n这里提到 g_eop_info 的用途\n", encoding="utf-8")

print("== 1. search_files 基本能力 ==")
r = ws.search_files("*.c")
check("文件名 glob 匹配仍正常", "acs_main.c" in r, r)

r = ws.search_files("g_eop_info")   # 不带 content：无 glob 字符 → 自动转内容搜索
check("普通标识符自动转内容搜索并命中 .c",
      "acs_main.c" in r and "g_eop_info" in r, r[:120])
check("内容搜索带行号", ":2:" in r, r[:120])

r = ws.search_files("g_eop_info", content=True)
check("content=true 显式内容搜索（.md 也命中）", "notes.md" in r, r[:120])

r = ws.search_files("*.nomatch")
check("带 glob 字符未命中 → 提示文案含 content=true 指引",
      "content=true" in r and "文件名" in r, r[:160])

r = ws.search_files("zzz_absent_symbol")
check("内容也搜不到 → 明确说明已扫内容",
      "文件内容均未找到" in r, r[:160])

print("== 2. 重复调用熔断（文本协议路径） ==")
act = 'ACTION: {"name": "search_files", "arguments": {"pattern": "g_eop_info"}}'
agent = Agent(StubLLM([act, act, "ANSWER: done"]), workspace=ws, max_iterations=8)
events = list(agent.run("找出 g_eop_info 定义"))
obs = [e["content"] for e in events if e["type"] == "observation"]
check("两次相同调用都有观察结果", len(obs) == 2, str(len(obs)))
check("第 1 次调用无熔断警告", "重复调用警告" not in obs[0], obs[0][-120:])
check("第 2 次相同调用注入熔断警告", "重复调用警告" in obs[1], obs[1][-120:])
check("警告要求换策略或 ANSWER", "换用其他工具" in obs[1], obs[1][-120:])

print("== 3. 换参数不算重复 ==")
act2 = 'ACTION: {"name": "search_files", "arguments": {"pattern": "other_kw"}}'
agent2 = Agent(StubLLM([act, act2, "ANSWER: done"]), workspace=ws, max_iterations=8)
events2 = list(agent2.run("再搜一次"))
obs2 = [e["content"] for e in events2 if e["type"] == "observation"]
check("参数不同 → 不触发熔断", all("重复调用警告" not in o for o in obs2),
      obs2[-1][-120:] if obs2 else "")

print("== 4. 重复任务间状态重置 ==")
agent3 = Agent(StubLLM([act, act, "ANSWER: doneA"]), workspace=ws, max_iterations=8)
list(agent3.run("任务A"))
agent3.llm = StubLLM([act, act, "ANSWER: doneB"])   # 换新桩，复用同一 Agent
events4 = list(agent3.run("任务B（重置后再来一次重复）"))
obs4 = [e["content"] for e in events4 if e["type"] == "observation"]
check("新任务首次相同调用不再带旧任务计数",
      len(obs4) == 2 and "重复调用警告" not in obs4[0], obs4[0][-120:] if obs4 else "")

print()
print(f"结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
