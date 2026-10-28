"""V23 修复冒烟测试：桩 LLM + 真实工具执行（临时目录，不碰工程配置）"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent
from core.config import Config, _set_field
from core.llm_client import is_embedding_model
from core.tools import TOOLS, execute_tool, tool
from core.workspace import Workspace

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (f"  | {detail}" if detail and not cond else ""))


class StubLLM:
    """桩：chat_stream 按脚本产出；empty 脚本只吐 reasoning（模拟 gpt-oss 空回复）"""
    def __init__(self, script):
        self.script = script
        self.cfg = Config(protocol="text")   # 本套件测文本协议路径（V28 后默认 native）
        self.last_reasoning = ""
        self.calls = 0

    def chat_stream(self, messages, model=None):
        self.calls += 1
        step = self.script[min(self.calls - 1, len(self.script) - 1)]
        if step == "empty_reasoning":
            self.last_reasoning = "让我想想……" * 50      # 只有思考，没有正文
            return
        if step == "empty_plain":
            self.last_reasoning = ""
            return
        for ch in step:
            yield ch


print("== 1. 空回复：连续 3 次带原因终止 + 事件可见 ==")
agent = Agent(StubLLM(["empty_reasoning"]))
events = list(agent.run("你好"))
kinds = [e["type"] for e in events]
check("连续空回复产出 format_error 事件（喂看门狗）", kinds.count("format_error") == 2, str(kinds))
check("第 3 次空回复以 error 终止", "error" in kinds and kinds[-1] == "error")
err = [e for e in events if e["type"] == "error"][-1]["content"]
check("终止原因点破思考过程场景", "思考过程" in err, err)
check("达到 MAX_EMPTY_RETRIES=3 即终止", agent.llm.calls == 3, str(agent.llm.calls))

print("== 2. 空回复：纯空（无 reasoning）也能终止 ==")
agent2 = Agent(StubLLM(["empty_plain"]))
events2 = list(agent2.run("你好"))
check("纯空回复同样终止", [e["type"] for e in events2][-1] == "error")

print("== 3. 高危确认门 ==")
tmp = Path(tempfile.mkdtemp())
ws = Workspace(tmp)
(tmp / "sub").mkdir()
(tmp / "sub" / "a.txt").write_text("x", encoding="utf-8")
cfg = Config(allow_shell=True, confirm_dangerous=True)

r = execute_tool("run_command", ws, cfg, {"command": "echo hi"})
check("无 confirm 回调 → 拒绝", r.startswith("错误: 该操作需要用户确认"), r[:60])
r = execute_tool("run_command", ws, cfg, {"command": "echo hi"},
                 confirm=lambda n, why: False)
check("用户点否 → 拒绝", "用户已拒绝" in r, r[:60])
r = execute_tool("run_command", ws, cfg, {"command": "echo hi"},
                 confirm=lambda n, why: True)
check("用户点允许 → 执行", "exit_code=0" in r and "hi" in r, r[:60])

r = execute_tool("delete_file", ws, cfg, {"path": "sub", "recursive": "true"})
check("recursive='true'(字符串) 也触发确认", r.startswith("错误: 该操作需要用户确认"), r[:60])
r = execute_tool("delete_file", ws, cfg, {"path": "sub", "recursive": "false"})
check("recursive='false'(字符串) 归一为假 → 不确认直接删空目录失败提示非空",
      "非空" in r, r[:60])
r = execute_tool("delete_file", ws, cfg, {"path": "sub", "recursive": "true"},
                 confirm=lambda n, why: True)
check("确认后递归删除执行", r.startswith("已删除"), r[:60])
check("目录确实被删", not (tmp / "sub").exists())

cfg_nc = Config(allow_shell=True, confirm_dangerous=False)
r = execute_tool("run_command", ws, cfg_nc, {"command": "echo free"})
check("confirm_dangerous=false → 免确认", "exit_code=0" in r, r[:60])

print("== 3b. run_python 每任务只确认一次 ==")
(tmp / "ok.py").write_text("print('ran')", encoding="utf-8")
asked = []


def _yes(name, why):
    asked.append(name)
    return True


approved = {}                            # 模拟一次 Agent.run() 的任务作用域
r1 = execute_tool("run_python", ws, cfg, {"path": "ok.py"},
                  confirm=_yes, approved=approved)
check("首次执行需确认且成功", len(asked) == 1 and "ran" in r1, r1[:80])
r2 = execute_tool("run_python", ws, cfg, {"path": "ok.py"},
                  confirm=_yes, approved=approved)
check("同任务内第二次不再询问", len(asked) == 1, f"询问次数={len(asked)}")
check("同任务内第二次仍然执行", "ran" in r2, r2[:80])
r3 = execute_tool("run_python", ws, cfg, {"path": "ok.py"},
                  confirm=_yes, approved=approved)
check("同任务内第三次也不再询问", len(asked) == 1 and "ran" in r3)

approved2 = {}                          # 新任务 → 结论失效，必须重新询问
r4 = execute_tool("run_python", ws, cfg, {"path": "ok.py"},
                  confirm=_yes, approved=approved2)
check("新任务重新询问", len(asked) == 2 and "ran" in r4, f"询问次数={len(asked)}")

approved3 = {}                          # 用户拒绝 → 记 False，本任务内不再弹窗
rejected = []


def _no(name, why):
    rejected.append(name)
    return False


r5 = execute_tool("run_python", ws, cfg, {"path": "ok.py"},
                  confirm=_no, approved=approved3)
check("拒绝后不执行", "用户已拒绝" in r5, r5[:60])
check("拒绝记为 False", approved3.get("run_python") is False, str(approved3))
r6 = execute_tool("run_python", ws, cfg, {"path": "ok.py"},
                  confirm=_no, approved=approved3)
check("拒绝后再次调用不再弹窗（直接拒绝）", len(rejected) == 1,
      f"询问次数={len(rejected)}")
check("直接拒绝也有明确原因", "用户已拒绝" in r6 and "本任务内已拒绝过" in r6, r6[:70])

# shell 不在 ONCE_PER_TASK：同一任务内每条命令都要单独确认
shell_asked = []
approved4 = {}


def _yes2(name, why):
    shell_asked.append(name)
    return True


execute_tool("run_command", ws, cfg, {"command": "echo a"},
             confirm=_yes2, approved=approved4)
execute_tool("run_command", ws, cfg, {"command": "echo b"},
             confirm=_yes2, approved=approved4)
check("shell 命令每条都确认（不共享授权）", len(shell_asked) == 2,
      f"询问次数={len(shell_asked)}")

print("== 4. 参数校验：模型参数错误 vs handler 内部 bug ==")
r = execute_tool("list_files", ws, cfg, {"bad_param": 1})
check("未知参数 → 参数不匹配", r.startswith("错误: 参数不匹配"), r[:60])


@tool("boom_test", "测试用：必抛内部异常", {"x": "任意"})
def _boom(ws, cfg, x=""):
    raise ValueError("handler 真 bug")

r = execute_tool("boom_test", ws, cfg, {"x": "1"})
check("handler 内部 bug → 工具内部错误（不再误报参数不匹配）",
      r.startswith("错误: 工具内部错误(ValueError)"), r[:60])
TOOLS.pop("boom_test")

print("== 5. embedding 模型识别 ==")
check("bge-m3 识别为 embedding", is_embedding_model("bge-m3:latest"))
check("text-embedding-3 识别为 embedding", is_embedding_model("text-embedding-3-small"))
check("glm-4-flash 不是 embedding", not is_embedding_model("glm-4-flash"))
check("deepseek-r1:8b 不是 embedding", not is_embedding_model("deepseek-r1:8b"))

print("== 6. 配置：默认值 + 字符串类型归一 ==")
check("confirm_dangerous 默认 True", Config().confirm_dangerous is True)
c = Config()
_set_field(c, "confirm_dangerous", "false")
check("'false' 字符串归一为 False", c.confirm_dangerous is False)
_set_field(c, "ui_scale", "2.0")
check("'2.0' 字符串归一为 float 2.0", c.ui_scale == 2.0)
_set_field(c, "max_iterations", "12")
check("'12' 字符串归一为 int 12", c.max_iterations == 12)

print("== 7. _trim_history：首条用户消息（任务上下文）不被字符级裁掉 ==")
a3 = Agent(StubLLM(["ANSWER: ok"]))
a3.messages = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "第一个任务"},          # 索引 1，必须保留
    {"role": "user", "content": "x" * 9000},           # 索引 2，可被裁
    {"role": "user", "content": "y" * 9000},           # 索引 3，可被裁
]
a3._trim_history()
check("system + 首条用户消息保留",
      a3.messages[0]["role"] == "system" and a3.messages[1]["content"] == "第一个任务",
      str([m["content"][:12] for m in a3.messages]))
check("超长老消息被裁", sum(len(m["content"]) for m in a3.messages) <= Agent.KEEP_CHARS + 20)

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
