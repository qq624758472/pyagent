"""V31.1 修复冒烟测试（对齐 gpt-oss-20B @ Ollama 实测特点）：

  1 同工具换参数连调 3 次 → 「方向警告」熔断（web_collect 刷屏对策）
  2 未知工具报错带近似工具提示（open_file → read_file）
  3 原生协议空回合重试不消耗迭代配额（MAX_EMPTY_RETRIES 仍封顶）

桩 SSE 服务器，不依赖真实模型；风格对齐 test_native_protocol.py。
"""
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent                      # noqa: E402
from core.config import Config                    # noqa: E402
from core.llm_client import LLMClient             # noqa: E402
from core.tools import execute_tool               # noqa: E402
from core.workspace import Workspace              # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (f"  | {detail}" if detail and not cond else ""))


def sse(server, deltas, tool_calls=None):
    server.send_response(200)
    server.send_header("Content-Type", "text/event-stream")
    server.end_headers()
    for d in deltas:
        server.wfile.write(("data: " + json.dumps(
            {"choices": [{"delta": {"content": d}}]}, ensure_ascii=False)
            + "\n\n").encode("utf-8"))
    if tool_calls:
        server.wfile.write(("data: " + json.dumps(
            {"choices": [{"delta": {"tool_calls": tool_calls}}]},
            ensure_ascii=False) + "\n\n").encode("utf-8"))
    server.wfile.write(b"data: [DONE]\n\n")
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


def agent_for(srv, max_iterations=8):
    cfg = Config(api_key="k", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                 protocol="native")
    return Agent(LLMClient(cfg), max_iterations=max_iterations)


def tc(name, args):
    return [{"id": "c", "index": 0, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]


print("== 1. 同工具换参数连调 3 次 → 方向警告 ==")
tmp = Path(tempfile.mkdtemp())
ws = Workspace(tmp)
# 替换 web_collect handler 为桩：测试只验证熔断逻辑，不真联网
from core import tools as _tools_mod
_orig_wc = _tools_mod.TOOLS["web_collect"].handler
_tools_mod.TOOLS["web_collect"].handler = (
    lambda ws, cfg, query="", criteria="", count=10,
           _progress=None, _should_stop=None: "未获取到有效数据。")
state = {"n": 0}


def h1(server):
    state["n"] += 1
    n = state["n"]
    if n <= 3:   # 三次 web_collect，参数各不相同（复刻 17:29 日志的刷屏模式）
        sse(server, [], tool_calls=tc("web_collect", {
            "query": f"kw{n}", "count": 5, "criteria": "source code"}))
    else:
        sse(server, ["已放弃联网，直接回答。"])


srv1 = make_server(h1)
ag = agent_for(srv1)
ag.workspace = ws
ev = list(ag.run("测试刷屏熔断"))
obs = [e["content"] for e in ev if e["type"] == "observation"]
check("三次调用都有观察", len(obs) == 3, str(len(obs)))
check("第 2 次不警告（streak=2 < 3）", "方向警告" not in obs[1], obs[1][-100:])
check("第 3 次触发方向警告", "方向警告" in obs[2], obs[2][-120:])
check("警告点名工具名", "web_collect" in obs[2], obs[2][-120:])
check("警告给出换路建议", "本地工具" in obs[2], obs[2][-120:])
_tools_mod.TOOLS["web_collect"].handler = _orig_wc   # 还原，不影响后续测试

print("== 2. 未知工具 → 近似工具提示 / 只读别名重定向 ==")
r = execute_tool("read_fil", ws, Config(), {"path": "x"})   # 拼写错误
check("拼写接近的真实工具名 → 给出近似建议",
      "read_file" in r and "未知工具" in r, r[:120])
r2 = execute_tool("完全不相干的名字", ws, Config(), {})
check("无近似匹配时只列可用工具", "可用工具" in r2, r2[:120])
r3 = execute_tool("open_file", ws, Config(), {"path": "x"})
check("open_file 走只读别名重定向（不再报未知工具）",
      "未知工具" not in r3, r3[:120])

print("== 3. 空回合重试不消耗迭代 ==")
state2 = {"n": 0}


def h2(server):
    state2["n"] += 1
    n = state2["n"]
    if n <= 2:                    # 前两轮全空（无正文无工具调用）
        sse(server, [])
    elif n == 3:                  # 第三轮才调工具
        sse(server, [], tool_calls=tc("list_files", {"path": "."}))
    else:
        sse(server, ["任务完成。"])


srv2 = make_server(h2)
ag2b = agent_for(srv2, max_iterations=2)
ag2b.workspace = ws
# 若空回合烧迭代：2 次空就耗尽 max_iterations=2 → 直接终止，永远到不了工具轮；
# 不烧迭代才能走完「2 空 + 1 工具 + 1 回答」共 4 次请求
ev2 = list(ag2b.run("测试空回合"))
kinds2 = [e["type"] for e in ev2]
check("空回合不烧迭代：max_iterations=2 仍能完成工具回合+回答",
      kinds2[-1] == "answer" and "action" in kinds2, str(kinds2))
fe = [e for e in ev2 if e["type"] == "format_error"]
check("空回合事件注明不消耗迭代", fe and "不消耗迭代" in fe[-1]["content"],
      str(fe)[:120])
check("Llm 请求数=4（2 空 + 1 工具 + 1 回答）", state2["n"] == 4, str(state2["n"]))

print("== 4. 连续 3 次空回合仍终止 ==")
state3 = {"n": 0}


def h3(server):
    state3["n"] += 1
    sse(server, [])               # 永远空


srv3 = make_server(h3)
ag3 = agent_for(srv3, max_iterations=12)
ev3 = list(ag3.run("测试连续空"))
kinds3 = [e["type"] for e in ev3]
check("第 3 次空后以 error 终止", kinds3[-1] == "error", str(kinds3))
check("请求数恰为 3（封顶生效，未死循环）", state3["n"] == 3, str(state3["n"]))

print()
print(f"结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
