"""小模型 Agent 回归测试：上下文完整性、协议回退与工具参数容错。"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent, _parse_tool_args
from core.config import Config
from core.llm_client import LLMError
from core import sessions


class StubLLM:
    def __init__(self):
        self.cfg = Config(protocol="auto")
        self.model = "test-small-model"

    def chat_turn(self, messages, tools=None):
        raise LLMError("HTTP 400: tools not supported")

    def chat_stream(self, messages):
        yield "ANSWER: fallback ok"


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


PASS = []
FAIL = []


def counted_check(name, condition):
    (PASS if condition else FAIL).append(name)
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


print("== 1. 原生协议回退保留历史 ==")
llm = StubLLM()
agent = Agent(llm, max_iterations=2)
agent.messages = [
    {"role": "system", "content": "old system"},
    {"role": "user", "content": "之前的问题"},
    {"role": "assistant", "content": "之前的回答"},
]
events = list(agent.run("当前任务"))
counted_check("回退后仍能回答", events[-1]["type"] == "answer")
counted_check("旧用户上下文保留", any(m.get("content") == "之前的问题" for m in agent.messages))
counted_check("旧助手上下文保留", any(m.get("content") == "之前的回答" for m in agent.messages))
counted_check("当前任务只追加一次", sum(
    m.get("role") == "user" and "当前任务" in str(m.get("content") or "")
    for m in agent.messages) == 1)

print("== 2. 历史裁剪不拆分工具调用组 ==")
agent.messages = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "首个任务"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-1", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path":"a"}'}}]},
    {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "x" * 80},
    {"role": "assistant", "content": "后续回答"},
    {"role": "user", "content": "新问题"},
]
agent.KEEP_CHARS = 35
agent._trim_history(keep=20)
roles = [m.get("role") for m in agent.messages]
counted_check("system 与首条任务保留", roles[:2] == ["system", "user"])
counted_check("没有孤立 tool 消息", all(
    m.get("role") != "tool" or any(
        c.get("id") == m.get("tool_call_id")
        for a in agent.messages if a.get("role") == "assistant"
        for c in a.get("tool_calls", [])) for m in agent.messages))

print("== 3. 非法工具参数不会以空对象执行 ==")
counted_check("非法 JSON 返回 None", _parse_tool_args("{broken") is None)
counted_check("非对象 JSON 返回 None", _parse_tool_args("[1, 2]") is None)
counted_check("合法对象正常解析", _parse_tool_args('{"path":"a"}') == {"path": "a"})

print("== 4. 会话保存保留原生工具调用字段 ==")
old_dir = sessions.SESSIONS_DIR
try:
    sessions.SESSIONS_DIR = Path(tempfile.mkdtemp())
    messages = [
        {"role": "system", "content": "hidden"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-2", "type": "function", "function": {
                "name": "list_files", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-2", "name": "list_files", "content": "ok"},
    ]
    sessions.save_session("quality-test", messages, "", "quality")
    data = sessions.load_session("quality-test")
    counted_check("会话 schema 版本为 2", data.get("schema_version") == 2)
    counted_check("assistant.tool_calls 保留", data["messages"][0].get("tool_calls") == messages[1]["tool_calls"])
    counted_check("tool_call_id 保留", data["messages"][1].get("tool_call_id") == "call-2")
finally:
    sessions.SESSIONS_DIR = old_dir

print(f"结果: {len(PASS)} 通过, {len(FAIL)} 失败")
