"""自主编码闭环回归：禁止无修改重跑、执行结果与完成判定严格一致。"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.agent import Agent
from core.config import Config
from core.llm_client import LLMError
from core.workspace import Workspace

PASS = []
FAIL = []

def check(label, ok):
    (PASS if ok else FAIL).append(label)
    print(("PASS " if ok else "FAIL ") + label)

class Model:
    def __init__(self, calls):
        self.calls = list(calls)
        self.cfg = Config(api_key="k", protocol="native", skills_enabled=False,
                          web_enabled=False, confirm_dangerous=True)
        self.model = "fake-gpt-oss"
        self.i = 0

    def chat_turn(self, messages, tools=None):
        spec = self.calls[min(self.i, len(self.calls) - 1)]
        self.i += 1
        if spec == "error500":
            raise LLMError("API 请求失败(HTTP 500)")
        if isinstance(spec, str):
            yield {"type": "content", "text": spec}
        else:
            name, arguments = spec
            yield {"type": "tool_call", "id": "call_" + str(self.i),
                   "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}

BAD = "import sys\nprint('broken')\nsys.exit(1)\n"
GOOD = ("import unittest\nclass Demo(unittest.TestCase):\n"
        "    def test_ok(self): self.assertEqual(2 + 2, 4)\n"
        "if __name__ == '__main__': unittest.main()\n")
with tempfile.TemporaryDirectory(prefix="pyagent_loop_test_") as tmp:
    model = Model([
        ("write_file", {"path": "test_app.py", "content": BAD}),
        ("run_python", {"path": "test_app.py"}),
        ("run_python", {"path": "test_app.py"}),
        ("write_file", {"path": "test_app.py", "content": GOOD}),
        ("run_python", {"path": "test_app.py"}),
        "已执行 unittest 测试且通过。"
    ])
    a = Agent(model, Workspace(tmp), max_iterations=10)
    events = list(a.run("创建 Python 测试脚本 test_app.py 并运行测试", 
                        confirm=lambda n, reason: n == "run_python"))
    obs = [e["content"] for e in events if e["type"] == "observation"]
    check("运行失败得到非零退出码", any("退出码 1" in x for x in obs))
    check("未修改时阻止重复执行", any("禁止无修改重复运行" in x for x in obs))
    check("代码修改后可重新执行测试", any("Ran 1 test" in x and "OK" in x for x in obs))
    check("真实测试通过后允许最终回答", events[-1]["type"] == "answer")
    check("代码确实修复落盘", Path(tmp, "test_app.py").read_text(encoding="utf8") == GOOD)

with tempfile.TemporaryDirectory(prefix="pyagent_unverified_") as tmp:
    model = Model([
        ("write_file", {"path": "test_app.py", "content": GOOD}),
        "任务完成，测试全部通过。",
        "任务完成，测试全部通过。",
        "任务完成，测试全部通过."
    ])
    a = Agent(model, Workspace(tmp), max_iterations=6)
    events = list(a.run("创建 Python 测试脚本 test_app.py 并执行测试"))
    check("阻止未经执行的假成功回答", not any(x["type"] == "answer" for x in events))
    check("验收失败明确产生错误或纠偏", any(x["type"] in ("error", "redirect") for x in events))

check("shell 退出码 1 视为失败", Agent._execution_failed("run_command", "exit_code=1\nerror"))
check("shell 退出码 0 视为成功", not Agent._execution_failed("run_command", "exit_code=0\nok"))
check("用户拒绝不会被报告成成功",
      Agent._execution_failed("run_command", "错误: 用户已拒绝执行高危操作"))

with tempfile.TemporaryDirectory(prefix="pyagent_500_resume_") as tmp:
    model = Model([
        ("write_file", {"path": "test_app.py", "content": GOOD}),
        "error500",
        ("run_python", {"path": "test_app.py"}),
        "已真实运行测试且通过"
    ])
    a = Agent(model, Workspace(tmp), max_iterations=8)
    events = list(a.run("创建 Python 测试脚本 test_app.py 并执行测试",
                        confirm=lambda n, why: n == "run_python"))
    check("HTTP500 后压缩上下文续跑", any(x["type"] == "redirect" and "模型服务" in x["content"] for x in events))
    check("HTTP500 恢复后继续工具执行", any(x["type"] == "action" and x["name"] == "run_python" for x in events))
    check("HTTP500 恢复没有虚报失败", events[-1]["type"] == "answer")
    check("恢复过程中保留工作区文件", Path(tmp, "test_app.py").read_text(encoding="utf8") == GOOD)

with tempfile.TemporaryDirectory(prefix="pyagent_reasoning_recovery_") as tmp:
    model = Model([
        ("write_file", {"path": "test_app.py", "content": GOOD}),
        "", "", "",  # native 只有 reasoning 没有正文或工具事件
        ("run_python", {"path": "test_app.py"}),
        "真实执行通过"
    ])
    agent = Agent(model, Workspace(tmp), max_iterations=8)
    events = list(agent.run("创建 Python 测试脚本 test_app.py 并执行测试",
                            confirm=lambda n, why: n == "run_python"))
    check("连续空回合后可缩减上下文并重试", any(
        e["type"] == "redirect" and "推理" in e["content"] for e in events))
    check("空回合恢复后可完成真实测试", events[-1]["type"] == "answer")

from core.llm_client import LLMClient
gpt_cfg = Config(api_key="k", model="fauxpaslife/gpt-oss-20B-Q5_K_M:latest")
gpt_payload = LLMClient(gpt_cfg)._payload([{"role": "user", "content": "hi"}])
check("gpt-oss 默认低推理预算", gpt_payload["reasoning_effort"] == "low")
check("gpt-oss 生成 token 有界", gpt_payload["max_tokens"] <= 4096)
generic_payload = LLMClient(Config(model="mock-model"))._payload([])
check("其他模型维持兼容参数", "reasoning_effort" not in generic_payload)

with tempfile.TemporaryDirectory(prefix="pyagent_docs_gate_") as tmp:
    readme = ("# CLI project\\nThis is a complete minimal task manager example.\\n"
              + "python app.py add task\\npython app.py list\\n"
              + "python app.py done 1\\n"
              + "Documentation explains the add/list/done sequence and storage path.\\n" * 3)
    model = Model([
        ("write_file", {"path": "app.py", "content": "print('demo')\\n"}),
        ("write_file", {"path": "test_app.py", "content": GOOD}),
        ("run_python", {"path": "test_app.py"}),
        "项目已经完成，单测通过",
        ("write_file", {"path": "README.md", "content": readme}),
        "全部完成"
    ])
    agent = Agent(model, Workspace(tmp), max_iterations=12)
    events = list(agent.run(
        "创建 Python 小项目 app.py、test_app.py、README.md，运行 unittest，"
        "README 必须说明 python app.py add/list/done 操作", 
        confirm=lambda n, why: n == "run_python"))
    check("README 缺失时不准提前宣称完成", any(
        e["type"] == "redirect" and "README" in e["content"] for e in events))
    check("补齐 README 后允许宣布完成", events[-1]["type"] == "answer")
    check("README 含完整命令", all(cmd in Path(tmp, "README.md").read_text(encoding="utf8")
                                for cmd in ("add", "list", "done")))

print("结果: %d 通过, %d 失败" % (len(PASS), len(FAIL)))
raise SystemExit(1 if FAIL else 0)
