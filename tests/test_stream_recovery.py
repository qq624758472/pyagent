"""真实 SSE 发出半截写文件调用后超时：丢弃片段，续做，禁止重复执行。"""
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
from core.agent import Agent
from core.config import Config
from core.llm_client import LLMClient, LLMError, LLMStreamInterrupted
from core.workspace import Workspace
from core.word_documents import inspect_docx
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
import core.sessions as sessions

PASS, FAIL = [], []
def check(label, ok):
    (PASS if ok else FAIL).append(label)
    print(('PASS ' if ok else 'FAIL ') + label)


class Server(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def emit(self, delta):
        self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': delta}]}) + '\n\n').encode())
        self.wfile.flush()
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.payloads.append(payload)
        n = len(self.server.payloads)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        try:
            if n == 1 or self.server.always_fail:
                if 'tools' in payload:
                    self.emit({'content': '现在直接生成文档。'})
                    self.emit({'tool_calls': [{'index': 0, 'id': 'partial', 'function': {
                        'name': 'write_file', 'arguments': '{"path":"DO_NOT_CREATE.docx","content":"partial'}}]})
                else:
                    self.emit({'content': 'ACTION: ' + json.dumps(dict(name='write_file', arguments=dict(path='DO_NOT_CREATE.docx', content='partial')))[:-3]})
                time.sleep(.35)
                return
            if n == 2:
                args = dict(path='output.docx', content='这是根据已收集资料生成的完整技术方案正文。' * 20)
                if 'tools' in payload:
                    self.emit({'tool_calls': [{'index': 0, 'id': 'good', 'function': {
                        'name': 'write_file', 'arguments': json.dumps(args)}}]})
                else:
                    self.emit({'content': 'ACTION: ' + json.dumps(dict(name='write_file', arguments=args))})
            else:
                self.emit({'content': '文档生成完成' if 'tools' in payload else 'ANSWER: 文档生成完成'})
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (ConnectionError, OSError): pass


class FastClient(LLMClient):
    RETRY_BACKOFF = 0
    def _timeouts(self, timeout=None): return (.2, .08)


server = Server(('127.0.0.1', 0), Handler)
server.payloads, server.always_fail = [], False
threading.Thread(target=server.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory(prefix='stream_recovery_') as tmp:
    root = Path(tmp)
    sessions.SESSIONS_DIR = root / 'sessions'
    for protocol in ('native', 'text'):
        ws = root / protocol
        ws.mkdir()
        server.payloads = []
        client = FastClient(Config(api_key='test', base_url=f'http://127.0.0.1:{server.server_port}/v1',
                                   protocol=protocol, llm_use_system_proxy=False, skills_enabled=False))
        agent = Agent(client, Workspace(ws))
        events = list(agent.run('请生成 Word 技术方案 output.docx'))
        check(protocol + ' 中途超时后自动生成有效文档', events[-1]['type'] == 'answer'
              and inspect_docx(ws / 'output.docx')['characters'] > 120)
        check(protocol + ' 半截工具调用没有执行', not (ws / 'DO_NOT_CREATE.docx').exists())
        check(protocol + ' 只执行一次完整写入', sum(e['type'] == 'action' for e in events) == 1)
        check(protocol + ' 新请求没有拼接损坏参数', len(server.payloads) == 3 and 'partial' not in json.dumps(server.payloads[1]['messages']))
        check(protocol + ' 自动恢复带任务和产出提醒', '已有足够资料' in json.dumps(server.payloads[1]['messages'], ensure_ascii=False))
        server.payloads, server.always_fail = [], True
        blocked = Agent(client, Workspace(ws))
        failed = list(blocked.run('请生成 Word 技术方案 output.docx'))
        check(protocol + ' 持续中断恢复最多两次后明确停止', failed[-1]['type'] == 'error' and len(server.payloads) == 3)
        server.always_fail = False

    app = QApplication([])
    win = MainWindow(Config())
    win.on_event(dict(type='delta', content='INCOMPLETE_PAYLOAD'))
    view = win._stream_view
    win.on_event(dict(type='redirect', discard_partial=True, content='自动续做'))
    check('界面清除失败正文避免混入下一轮', 'INCOMPLETE_PAYLOAD' not in view.source and win._stream_view is None)
    # 永久请求错误不能被当作网络中断自动续做。
    agent._stream_recoveries = 0
    check('普通 API 错误不触发中途恢复', not agent._recover_interrupted_stream(LLMError('HTTP 401')))
    win.hide()
server.shutdown()
server.server_close()
print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
sys.exit(bool(FAIL))
