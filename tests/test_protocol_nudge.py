"""V27 协议追问闭环测试：思考含工具意图 → 追问协议 → 真正执行工具

复现 07:57 实跑：gpt-oss 思考通道里说 "Use kb_search ..."，正文为空，
旧行为把思考原样当答案给用户；新行为应追问协议，让模型吐出可解析的 ACTION。
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent
from core.config import Config
from core.llm_client import LLMClient

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (f"  | {detail}" if detail and not cond else ""))


def make_server(script):
    """script: 每次请求返回一段 reasoning 文本（按顺序循环）"""
    state = {"i": 0}

    def handler(h):
        n = int(h.headers.get("Content-Length", 0) or 0)
        h.rfile.read(n)
        text = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        h.server.request_count = state["i"]
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        line = ("data: " + json.dumps(
            {"choices": [{"delta": {"reasoning_content": text}}]},
            ensure_ascii=False) + "\n\n")
        h.wfile.write(line.encode("utf-8"))
        h.wfile.flush()
        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            handler(self)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    srv.request_count = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def agent_for(srv, **kw):
    cfg = Config(api_key="k", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                 protocol="text")     # 本套件测文本协议路径（V28 后默认 native）
    return Agent(LLMClient(cfg), **kw)


print("== 1. 思考含工具意图 → 追问协议 → 执行工具 ==")
srv1 = make_server([
    # 第1轮：只有思考，且表达了要用 kb_search（无协议）
    "The user asks about X. I should use kb_search to look it up in the workspace.",
    # 第2轮：仍是思考通道，但这次写出了 ACTION 协议 → 应被解析并执行
    'Let me call the tool now. ACTION: {"name": "list_files", "arguments": {"path": "."}}',
    # 第3轮：拿到观察结果后给出 ANSWER
    "I have the file list now. ANSWER: 工作区内有 1 个文件。",
])
ag = agent_for(srv1)
import tempfile
from pathlib import Path
from core.workspace import Workspace
ws_dir = Path(tempfile.mkdtemp())
(ws_dir / "demo.py").write_text("print(1)", encoding="utf-8")
ag.set_workspace(Workspace(ws_dir))
events = list(ag.run("工作区里有什么文件"))
kinds = [e["type"] for e in events]
check("第1轮触发协议提醒（redirect）", "redirect" in kinds, str(kinds))
check("没有把思考直接当答案", kinds.index("redirect") < kinds.index("answer"), str(kinds))
check("工具真的被执行了", "action" in kinds, str(kinds))
obs = [e for e in events if e["type"] == "observation"]
check("拿到了工具观察结果（真实文件列表）", obs and "demo.py" in obs[0]["content"],
      obs[0]["content"][:80] if obs else "")
ans = [e for e in events if e["type"] == "answer"]
check("最终产出答案", len(ans) == 1 and "1 个文件" in ans[0]["content"])
srv1.shutdown()

print("== 2. 思考不含工具意图 → 直接作答，不浪费轮次 ==")
srv2 = make_server([
    "用户问了个常识问题，我直接回答就好，不需要工具。ANSWER: 僵尸毁灭工程是一款末日题材生存游戏。",
])
ag2 = agent_for(srv2)
ev2 = list(ag2.run("僵尸毁灭工程是什么"))
k2 = [e["type"] for e in ev2]
check("无 redirect（没触发追问）", "redirect" not in k2, str(k2))
check("直接产出答案", k2[-1] == "answer")
check("答案内容带上了回复", "生存游戏" in ev2[-1]["content"])
srv2.shutdown()

print("== 3. 追问次数有上限（模型一直不按协议 → 兜底直答，不死循环） ==")
srv3 = make_server([
    "我应该调用 list_files 看看文件。",
])
ag3 = agent_for(srv3)
ev3 = list(ag3.run("看看文件"))
k3 = [e["type"] for e in ev3]
check("redirect 次数不超过 2", k3.count("redirect") <= 2, str(k3))
check("最终以 answer/error 收尾", k3[-1] in ("answer", "error"), str(k3))
check("请求次数有界（<=4）", srv3.request_count <= 4, str(srv3.request_count))
srv3.shutdown()

print("== 4. 判定函数边界 ==")
check("提到工具名→需协议", Agent._reasoning_needs_protocol("use web_collect to search"))
check("提到 ACTION 痕迹→需协议", Agent._reasoning_needs_protocol("ANSWER: x"))
check("纯闲聊→不需协议", not Agent._reasoning_needs_protocol("今天天气不错"))
check("空文本→不需协议", not Agent._reasoning_needs_protocol(""))

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
