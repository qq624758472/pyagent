"""V26 reasoning_content 兜底：真实 HTTP SSE 服务器集成测试

复现用户日志中的故障：gpt-oss-20B 经网关把全部输出放进 reasoning_content、
content 为空 → Agent 连续空回复终止。修复：正文为空时把思考过程作为回复输出。
"""
import json
import sys
import threading
import time
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


def send_sse(h, deltas):
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.end_headers()
    for d in deltas:
        line = "data: " + json.dumps({"choices": [{"delta": d}]}, ensure_ascii=False) + "\n\n"
        h.wfile.write(line.encode("utf-8"))
        h.wfile.flush()
    h.wfile.write(b"data: [DONE]\n\n")
    h.wfile.flush()


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


def client_for(srv):
    cfg = Config(api_key="test-key", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
    return LLMClient(cfg)


print("== 1. 复现故障：只有 reasoning_content（用户日志同款） ==")
srv1 = make_server(lambda h: send_sse(h, [
    {"reasoning_content": "用户想让"},
    {"reasoning_content": "我罗列文件，我应该先 list_files。"},
    {"reasoning_content": "ACTION: {\"name\": \"list_files\", \"arguments\": {}}"},
]))
out1 = list(client_for(srv1).chat_stream([{"role": "user", "content": "罗列文件"}]))
joined = "".join(out1)
check("思考过程被兜底为回复输出", "list_files" in joined, repr(joined))
check("三段 reasoning 都输出", joined.startswith("用户想让")
      and joined.count("ACTION") == 1 and "罗列文件" in joined, repr(joined))
srv1.shutdown()

print("== 2. 正常网关：content + reasoning 分离（deepseek-r1 式） ==")
srv2 = make_server(lambda h: send_sse(h, [
    {"reasoning_content": "这是思考，不应出现在回复里"},
    {"content": "ANSWER: "},
    {"content": "你好，我是助手。"},
]))
llm2 = client_for(srv2)
out2 = "".join(llm2.chat_stream([{"role": "user", "content": "hi"}]))
check("只输出 content 不掺 reasoning", out2 == "ANSWER: 你好，我是助手。", repr(out2))
check("last_reasoning 仍记录思考", "思考" in llm2.last_reasoning)
srv2.shutdown()

print("== 3. 真空响应：既无 content 也无 reasoning ==")
srv3 = make_server(lambda h: send_sse(h, []))
llm3 = client_for(srv3)
out3 = list(llm3.chat_stream([{"role": "user", "content": "hi"}]))
check("无任何输出", out3 == [])
check("last_reasoning 为空", llm3.last_reasoning == "")
srv3.shutdown()

print("== 4. 连接被断 → 自动重试 → 成功 ==")
state = {"n": 0}


def drop_first(h):
    if state["n"] == 0:
        state["n"] += 1
        h.close_connection = True
        try:
            h.connection.close()
        except Exception:
            pass
        return
    send_sse(h, [{"content": "重试后成功"}])


srv4 = make_server(drop_first)
t0 = time.monotonic()
out4 = "".join(client_for(srv4).chat_stream([{"role": "user", "content": "hi"}]))
check("重试后拿到内容", out4 == "重试后成功", repr(out4))
check("重试有退避等待(>=1.4s)", time.monotonic() - t0 >= 1.4, f"{time.monotonic()-t0:.1f}s")
srv4.shutdown()

print("== 5. 端到端：Agent 经 reasoning-only 网关完成任务 ==")
srv5 = make_server(lambda h: send_sse(h, [
    {"reasoning_content": "思考中……用户只打了招呼，直接回答即可。"},
    {"reasoning_content": "ANSWER: 你好！有什么可以帮你的？"},
]))
ag = Agent(client_for(srv5))
events = list(ag.run("你好"))
kinds = [e["type"] for e in events]
answers = [e for e in events if e["type"] == "answer"]
check("产出 answer 事件（不再连续空回复终止）", len(answers) == 1, str(kinds))
check("answer 内容来自思考过程", "你好" in answers[0]["content"], answers[0]["content"] if answers else "")
check("无 error/cancelled 事件", "error" not in kinds and "cancelled" not in kinds)
check("对话历史已记录", any(m["role"] == "assistant" for m in ag.messages))
srv5.shutdown()

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
