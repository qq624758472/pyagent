"""真实本地慢 SSE：首字节等待、参数流心跳、取消、网络超时和界面看门狗。"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel
from core.agent import Agent
from core.config import Config
from core.llm_client import LLMClient, LLMError
import core.sessions as sessions
from ui.main_window import MainWindow, ToolCollapseCard
from ui.workers import AgentWorker

PASS, FAIL = [], []


def check(label, ok):
    (PASS if ok else FAIL).append(label)
    print(('PASS ' if ok else 'FAIL ') + label)


class Server(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def delta(self, value):
        self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': value}]}) + '\n\n').encode())
        self.wfile.flush()

    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.server.requests += 1
        self.server.started.set()
        try:
            if self.server.mode == 'blocked':
                self.server.release.wait(2)
            elif self.server.mode == 'slow':
                time.sleep(0.35)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            if self.server.mode == 'tools':
                self.delta({'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {
                    'name': 'write_file', 'arguments': '{"path":"note.txt","content":"'}}]})
                for _ in range(5):
                    time.sleep(0.07)
                    self.delta({'tool_calls': [{'index': 0, 'function': {'arguments': 'x'}}]})
                self.delta({'tool_calls': [{'index': 0, 'function': {'arguments': '"}'}}]})
            else:
                if self.server.mode == 'reasoning':
                    for _ in range(5):
                        time.sleep(0.07)
                        self.delta({'reasoning': 'thinking'})
                self.delta({'content': 'Completed'})
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def server(mode):
    srv = Server(('127.0.0.1', 0), Handler)
    srv.mode, srv.requests = mode, 0
    srv.started, srv.release = threading.Event(), threading.Event()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def client(srv, protocol='native'):
    llm = LLMClient(Config(api_key='test', base_url=f'http://127.0.0.1:{srv.server_port}/v1',
                           protocol=protocol, model='test', skills_enabled=False))
    llm.REASONING_BEAT = 0.05
    return llm


app = QApplication.instance() or QApplication([])
srv = server('slow')
llm = client(srv)
events = list(llm.chat_turn([{'role': 'user', 'content': 'hello'}]))
check('等待首字节期间提供等待状态', any(e['type'] == 'keepalive' and e.get('phase') == 'waiting' for e in events))
check('慢请求最终正常返回', ''.join(e.get('text', '') for e in events) == 'Completed')
srv.shutdown(); srv.server_close()

srv = server('tools')
events = list(client(srv).chat_turn([{'role': 'user', 'content': 'hello'}]))
calls = [e for e in events if e['type'] == 'tool_call']
check('长工具参数生成期间也有心跳', any(e['type'] == 'keepalive' for e in events[:-1]))
check('参数拼完只交付一次完整调用', len(calls) == 1 and json.loads(calls[0]['arguments'])['content'] == 'xxxxx')
srv.shutdown(); srv.server_close()

srv = server('reasoning')
parts = list(client(srv, 'text').chat_stream([{'role': 'user', 'content': 'hello'}]))
check('文本协议纯推理期间能产出心跳', '' in parts and ''.join(parts) == 'Completed')
srv.shutdown(); srv.server_close()

for protocol in ('native', 'text'):
    srv = server('blocked')
    llm = client(srv, protocol)
    agent = Agent(llm)
    collected = []
    worker = AgentWorker(agent, 'hello')
    worker.agent_event.connect(collected.append, Qt.DirectConnection)
    worker.start()
    check(f'{protocol}: 已进入实际阻塞请求', srv.started.wait(2))
    start = time.monotonic()
    worker.stop()
    check(f'{protocol}: 停止无需等网络读取超时', worker.wait(1500) and time.monotonic() - start < 1.5)
    check(f'{protocol}: 取消不会显示网络错误或假成功', any(e['type'] == 'cancelled' for e in collected)
          and not any(e['type'] in ('error', 'answer') for e in collected))
    srv.release.set()
    srv.mode = 'normal'
    next_events = list(agent.run('hello again'))
    check(f'{protocol}: 停止后可正常运行下一任务', next_events[-1]['type'] == 'answer')
    time.sleep(0.1)
    check(f'{protocol}: 被取消的旧请求不会继续重试', srv.requests == 2 and not llm._active_requests)
    srv.shutdown(); srv.server_close()

srv = server('blocked')
llm = client(srv)
llm.RETRY_ON_CONN = 0
llm.RETRY_BACKOFF = 0
try:
    list(llm.chat_turn([{'role': 'user', 'content': 'hello'}], timeout=(0.1, 0.1)))
    failed = False
except LLMError:
    failed = True
check('真实网络超时仍有界失败而非无限续命', failed and not llm._active_requests)
srv.release.set()
srv.shutdown(); srv.server_close()

with tempfile.TemporaryDirectory(prefix='pyagent_watchdog_') as tmp:
    sessions.SESSIONS_DIR = Path(tmp)
    win = MainWindow(Config())
    class Worker:
        stopped = False
        def isRunning(self): return True
        def stop(self): self.stopped = True
    win.worker, win._busy = Worker(), True
    win._on_watchdog()
    check('45秒无界面事件不取消仍运行的任务', not win.worker.stopped and win._busy and win.watchdog.isActive())
    check('等待状态不会伪报网络错误', not win.findChildren(QLabel, 'errLine'))
    win.on_event(dict(type='checkpoint', round=32, content='已处理32轮，继续执行', details='代码版本10 测试通过版本-1'))
    check('内部版本信息只出现在折叠执行卡', win._tool_card is not None
          and not any('代码版本' in label.text() for label in win.findChildren(QLabel, 'sysLine')))
    win._busy = False
    win.watchdog.stop()
    win._tool_card.finish()
    win.hide()

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
raise SystemExit(1 if FAIL else 0)
