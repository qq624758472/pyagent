"""原生 function calling 协议单元测试（桩 SSE 服务器，不依赖真实模型）

场景：
  1 工具调用 → 执行 → role=tool 回传 → 模型给出正文 → answer 事件
  2 端点返回 HTTP 400（不认 tools）→ 自动回退文本协议
  3 工具参数 JSON 非法 → 不崩溃，按空参数执行
  4 纯文本回复（无工具）→ 直接 answer
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent                      # noqa: E402
from core.config import Config                    # noqa: E402
from core.llm_client import LLMClient             # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (f"  | {detail}" if detail and not cond else ""))


def sse(server, deltas, tool_calls=None):
    server.send_response(200)
    server.send_header("Content-Type", "text/event-stream")
    server.end_headers()
    lines = []
    for d in deltas:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": d}}]},
                                          ensure_ascii=False) + "\n\n")
    if tool_calls:
        lines.append("data: " + json.dumps(
            {"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False) + "\n\n")
    lines.append("data: [DONE]\n\n")
    for ln in lines:
        server.wfile.write(ln.encode("utf-8"))
        server.wfile.flush()


def make_server(handler):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(n)
            handler(self)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def agent_for(srv, protocol="native", **kw):
    cfg = Config(api_key="k", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                 protocol=protocol)
    return Agent(LLMClient(cfg), **kw)


print("== 1. 工具调用 → 执行 → 回传 → 正文 ==")
state = {"n": 0}


def h1(server):
    state["n"] += 1
    if state["n"] == 1:
        sse(server, [], tool_calls=[{"id": "c1", "index": 0, "type": "function",
                                     "function": {"name": "list_files",
                                                  "arguments": '{"path": "."}'}}])
    else:
        sse(server, ["工作区里有 demo.py 等文件。"])


srv1 = make_server(h1)
ag = agent_for(srv1)
ev = list(ag.run("工作区里有什么"))
kinds = [e["type"] for e in ev]
check("产生 action 事件", "action" in kinds, str(kinds))
check("产生 observation 事件", "observation" in kinds, str(kinds))
check("最终产出 answer", kinds[-1] == "answer", str(kinds))
check("答案内容正确", ev[-1]["content"].startswith("工作区里有"), ev[-1]["content"][:40])
check("工具结果已回传(role=tool)",
      any(m.get("role") == "tool" for m in ag.messages),
      str([m.get("role") for m in ag.messages]))
check("assistant 消息带 tool_calls",
      any(m.get("role") == "assistant" and m.get("tool_calls") for m in ag.messages))
srv1.shutdown()

print("== 2. 端点 HTTP 400 → 自动回退文本协议 ==")


def h400(server):
    body = b'{"error":"tools not supported"}'
    server.send_response(400)
    server.send_header("Content-Type", "application/json")
    server.send_header("Content-Length", str(len(body)))
    server.end_headers()
    server.wfile.write(body)


srv2 = make_server(h400)
ag2 = agent_for(srv2)
ev2 = list(ag2.run("你好"))
k2 = [e["type"] for e in ev2]
check("给出回退提示", "redirect" in k2, str(k2))
check("回退后标记 _force_text", getattr(ag2, "_force_text", False) is True)
check("回退后走文本协议", ag2._native_mode() is False)
srv2.shutdown()

print("== 3. 工具参数 JSON 非法 → 不执行并要求重试 ==")
state3 = {"n": 0}


def h3(server):
    state3["n"] += 1
    if state3["n"] == 1:
        sse(server, [], tool_calls=[{"id": "c1", "index": 0, "type": "function",
                                     "function": {"name": "list_files",
                                                  "arguments": "{不是合法 JSON"}}])
    else:
        sse(server, ["好的。"])


srv3 = make_server(h3)
ag3 = agent_for(srv3)
ev3 = list(ag3.run("列文件"))
k3 = [e["type"] for e in ev3]
check("非法参数未导致崩溃", "error" not in k3, str(k3))
check("非法参数不会执行工具", "action" not in k3, str(k3))
check("给出参数格式修正提示", "format_error" in k3, str(k3))
srv3.shutdown()

print("== 4. 纯文本回复 → 直接 answer ==")
srv4 = make_server(lambda s: sse(s, ["直接回答你的问题。"]))
ag4 = agent_for(srv4)
ev4 = list(ag4.run("你好"))
check("直接产出 answer", [e["type"] for e in ev4][-1] == "answer")
check("答案未带 ANSWER 前缀", not ev4[-1]["content"].startswith("ANSWER"))
srv4.shutdown()

print("== 5. 答案前缀清洗 ==")
for raw, expect in (("ANSWER: 内容", "内容"), ("**答：**\n内容", "内容"),
                     ("**ANSWER**\n内容", "内容"), ("正常内容", "正常内容")):
    got = Agent._clean_answer(raw)
    check(f"清洗 {raw!r}", got == expect, repr(got))

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
