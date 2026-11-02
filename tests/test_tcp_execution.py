"""网络执行闭环回归：真实子进程、TCP 回环、运行证据门禁。"""
import json
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.agent import Agent
from core.config import Config
from core.tools import TOOLS, execute_tool, openai_tools
from core.workspace import Workspace

passed, failed = [], []


def check(name, condition):
    (passed if condition else failed).append(name)
    print(("PASS " if condition else "FAIL ") + name, flush=True)


class ScriptModel:
    def __init__(self, sequence):
        self.sequence = sequence
        self.n = 0
        self.model = "fake-gpt-oss"
        self.cfg = Config(api_key="k", protocol="native", skills_enabled=False,
                          web_enabled=False, confirm_dangerous=True)

    def chat_turn(self, messages, tools=None):
        action = self.sequence[min(self.n, len(self.sequence) - 1)]
        self.n += 1
        if isinstance(action, str):
            yield {"type": "content", "text": action}
        else:
            name, args = action
            yield {"type": "tool_call", "id": "call_" + str(self.n),
                   "name": name, "arguments": json.dumps(args)}


with tempfile.TemporaryDirectory(prefix="pyagent_tcp_") as temp:
    ws = Workspace(temp)
    cfg = Config(api_key="test", python_enabled=True, confirm_dangerous=True,
                 skills_enabled=False)
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    server = (
        "import socket\n"
        "s=socket.socket()\n"
        "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"
        f"s.bind(('127.0.0.1',{port}))\n"
        "s.listen(5)\n"
        "while True:\n"
        "    c,_=s.accept()\n"
        "    with c:\n"
        "        data=c.recv(4096)\n"
        "        if data: c.sendall(data)\n"
    )
    client = (
        "import socket\n"
        f"with socket.create_connection(('127.0.0.1',{port}),timeout=3) as c:\n"
        "    c.sendall(b'hello')\n"
        "    print(c.recv(4096).decode('utf-8'))\n"
    )
    ws.write_file("tcp_server.py", server)
    ws.write_file("tcp_client.py", client)
    for name in ("start_process", "process_status", "stop_process", "tcp_echo_test"):
        check("registered " + name, name in TOOLS
              and name in [a["function"]["name"] for a in openai_tools()])

    approved = {}
    start = execute_tool("start_process", ws, cfg, {"path": "tcp_server.py"},
                         confirm=lambda n, why: True, approved=approved)
    check("start_process launches resident server", "process_id=" in start
          and "错误:" not in start)
    import re
    match = re.search(r"process_id=([a-f0-9]+)", start)
    if match:
        process_id = match.group(1)
        status = execute_tool("process_status", ws, cfg, {"process_id": process_id})
        check("process_status sees running", "status=running" in status)
        with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
            connection.sendall(b"hello")
            value = connection.recv(1024)
        check("resident TCP socket exchange", value == b"hello")
        stopped = execute_tool("stop_process", ws, cfg, {"process_id": process_id})
        check("stop_process cleans up", "已停止" in stopped)
    else:
        check("process_status sees running", False)
        check("resident TCP socket exchange", False)
        check("stop_process cleans up", False)

    result = execute_tool("tcp_echo_test", ws, cfg,
        {"server_path": "tcp_server.py", "client_path": "tcp_client.py",
         "message": "hello", "port": port},
        confirm=lambda n, why: True)
    check("independent TCP integration passes", "TCP联调通过" in result)
    check("generated client really checked three times", "客户端连续 3 次" in result)

    ws.write_file("tcp_client_bad.py", "print('unrelated')\n")
    bad = execute_tool("tcp_echo_test", ws, cfg,
        {"server_path": "tcp_server.py", "client_path": "tcp_client_bad.py",
         "port": port}, confirm=lambda n, why: True)
    check("bad generated client is rejected", bad.startswith("错误:"))

    ws.write_file("tcp_client_false_pass.py", "print('hello')\nprint('接收错误: 10038')\n")
    false_pass = execute_tool("tcp_echo_test", ws, cfg,
        {"server_path": "tcp_server.py", "client_path": "tcp_client_false_pass.py",
         "port": port}, confirm=lambda n, why: True)
    check("client printing hello with socket error must fail", false_pass.startswith("错误:"))

    task = ("你自己打开TCP服务器接收，并用TCP客户端进行发送，然后进行调试。"
            "使用 tcp_echo_test 自动测试，必要时修改程序，汇报测试结果。")
    no_tool = Agent(ScriptModel(["我没有办法自己打开 socket，请在终端手工运行。"] * 5),
                    ws, max_iterations=6)
    events = list(no_tool.run(task))
    check("network task cannot falsely report completion",
          not any(e["type"] == "answer" for e in events))
    check("network task emits remediation", any(e["type"] == "redirect" for e in events))

    with_tool = Agent(ScriptModel([
        ("tcp_echo_test", {"server_path": "tcp_server.py",
                           "client_path": "tcp_client.py", "port": port,
                           "message": "hello"}),
        "已经完成服务端和客户端的 TCP 联调。"
    ]), ws, max_iterations=5)
    events2 = list(with_tool.run(task, confirm=lambda n, why: n == "tcp_echo_test"))
    check("successful network check permits answer",
          events2[-1]["type"] == "answer")
    check("test result propagated to agent",
          any("TCP联调通过" in e.get("content", "") for e in events2
              if e["type"] == "observation"))

print("结果: %d 通过, %d 失败" % (len(passed), len(failed)))
raise SystemExit(bool(failed))
