"""V31.2 修复冒烟测试（18:17 GUI 会话失败复盘驱动）：

  A. search_files 参数容错：模型把关键词塞进 content / 漏掉 pattern 时
     也能搜到（此前 8 轮里有 4 轮耗在"missing a required argument: pattern"）
  B. 思考期心跳：推理模型只吐 reasoning 时持续产生 keepalive 事件，
     避免 UI 看门狗 45s 误报超时
  C. AutoHeightLabel：长文本块高度随内容自适应（此前固定高度算成 16px，
     展开执行过程卡看到的是空行）

用法：python tests/test_v312_fixes.py
"""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


# ---------------- A. search_files 参数容错 ----------------
tmp = Path(tempfile.mkdtemp())
ws = Workspace(tmp)
(tmp / "acs_main.c").write_text(
    "eop_info_t g_eop_info[MAX_RECORD_EOP] = {0};\nint main(void){return 0;}\n",
    encoding="utf-8")
(tmp / "sub").mkdir()
(tmp / "sub" / "other.c").write_text("int g_eop_info_local = 1;\n", encoding="utf-8")
cfg = Config()

print("== A. search_files 参数容错 ==")
r = execute_tool("search_files", ws, cfg, {"content": "g_eop_info"})
check("content 传字符串（无 pattern）→ 当关键字按内容搜",
      "acs_main.c" in r and "命中" in r, r[:120])

r = execute_tool("search_files", ws, cfg,
                 {"pattern": "g_eop_info", "content": True})
check("pattern + content=true → 内容搜索",
      "acs_main.c" in r and "命中" in r, r[:120])

r = execute_tool("search_files", ws, cfg, {"pattern": "*.c"})
check("pattern 为 glob → 文件名匹配", "acs_main.c" in r and "sub" in r, r[:120])

r = execute_tool("search_files", ws, cfg,
                 {"pattern": "g_eop_info_local", "path": "sub"})
check("path 限定子目录 → 只搜该目录",
      "sub" in r and "acs_main.c" not in r, r[:120])

r = execute_tool("search_files", ws, cfg,
                 {"content": "g_eop_info", "recursive": True, "query": "x"})
check("多余参数被忽略不报错", "命中" in r, r[:120])

r = execute_tool("search_files", ws, cfg, {})
check("完全没给关键字 → 给出可行动报错",
      "错误" in r and "pattern" in r, r[:120])

print("== A2. read_file 行区间 + 未知工具别名重定向 ==")
(tmp / "code.c").write_text("\n".join(f"line {i}" for i in range(1, 21)),
                             encoding="utf-8")
r = execute_tool("read_file", ws, cfg, {"path": "code.c"})
check("不带行号 → 全文", "line 1" in r and "line 20" in r, r[:80])

r = execute_tool("read_file", ws, cfg,
                 {"path": "code.c", "line_start": 5, "line_end": 7})
check("指定行区间 → 只返回该段", "line 5" in r and "line 7" in r
      and "line 4" not in r and "line 8" not in r, r[:120])
check("行区间带总行数提示", "共 20 行" in r, r[:80])

r = execute_tool("read_file", ws, cfg, {"path": "code.c", "line": 3})
check("IDE 风格 line=3 → 单行", "line 3" in r and "line 4" not in r, r[:80])

r = execute_tool("open_file", ws, cfg, {"path": "code.c"})
check("open_file 自动重定向为 read_file 并真的读到内容",
      "line 1" in r and "未知工具" not in r, r[:80])

r = execute_tool("grep", ws, cfg, {"pattern": "line 3"})
check("grep 自动重定向为 search_files", "line 3" in r and "未知工具" not in r,
      r[:80])

r = execute_tool("delete", ws, cfg, {"path": "code.c"})
check("写/删类别名不自动执行（仍拒绝并提示）", "未知工具" in r, r[:100])


# ---------------- B. 思考期心跳 ----------------
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


def sse_reasoning(server, n=30):
    server.send_response(200)
    server.send_header("Content-Type", "text/event-stream")
    server.end_headers()
    for _ in range(n):
        server.wfile.write(("data: " + json.dumps(
            {"choices": [{"delta": {"reasoning": "思考中..."}}]},
            ensure_ascii=False) + "\n\n").encode("utf-8"))
    server.wfile.write(b"data: [DONE]\n\n")
    server.wfile.flush()


print("== B. 思考期心跳（原生协议） ==")
srv = make_server(sse_reasoning)
c = Config(api_key="k", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
           protocol="native")
llm = LLMClient(c)
llm.REASONING_BEAT = 0.0          # 测试里让每个 reasoning 分片都出心跳
ag = Agent(llm, max_iterations=4)
ev = list(ag.run("思考型回复"))
kinds = [e["type"] for e in ev]
check("推理期间产生 keepalive 事件（喂看门狗）",
      kinds.count("keepalive") >= 5, str(kinds))
check("原生协议不把内部 reasoning 当作最终答案泄漏",
      "answer" not in kinds and "error" in kinds, str(kinds))


# ---------------- C. UI 文本块高度自适应 ----------------
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget   # noqa: E402

from ui.main_window import AutoHeightLabel, ToolCollapseCard      # noqa: E402

app = QApplication.instance() or QApplication([])

print("== C. AutoHeightLabel / 执行过程卡 ==")
host = QWidget()
lay = QVBoxLayout(host)
lab = AutoHeightLabel("这是一段很长的执行过程内容，" * 60, host, max_height=260)
lay.addWidget(lab)
host.resize(700, 500)
host.show()
app.processEvents()
check("长文本块高度 > 16px（不再被裁成一条线）", lab.height() > 16, str(lab.height()))
check("高度按封顶收敛（<= 270）", lab.height() <= 270, str(lab.height()))

card_host = QWidget()
cl = QVBoxLayout(card_host)
card = ToolCollapseCard({}, card_host)
cl.addWidget(card)
card_host.resize(700, 500)
card_host.show()
card.add_action("search_files", {"pattern": "g_eop_info", "content": True})
card.add_observation("命中 20 处：ACS\\acs_libsofa.c:6: eop_info_t g_eop_info;" * 6)
card.head.setChecked(True)
card._on_toggle(True)
app.processEvents()
body_labels = card.body.findChildren(AutoHeightLabel)
check("执行过程卡内的文本块已按内容撑开",
      body_labels and all(l.height() > 16 for l in body_labels),
      str([l.height() for l in body_labels]))
check("展开后卡片总高度大于折叠头高度（内容可见）",
      card.height() > card.head.height(), f"{card.height()} vs {card.head.height()}")

print("== D. 回答气泡高度自适应 ==")
host2 = QWidget()
lay2 = QVBoxLayout(host2)
stream = AutoHeightLabel("", host2, objname="streamText", max_height=2400)
lay2.addWidget(stream)
host2.resize(700, 500)
host2.show()
app.processEvents()
stream.setHtml("<p>定义在 <b>acs_libsofa.c</b> 第 6 行</p>" + "<p>细节行</p>" * 20)
app.processEvents()
check("回答气泡高度随内容增长（不再只有 24px 一行）",
      stream.height() > 60, str(stream.height()))

print()
print(f"结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
