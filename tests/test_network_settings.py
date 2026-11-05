"""网络设置、代理隔离、读取超时后的完整消息组压缩。"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PyQt5.QtWidgets import QApplication
from core.config import Config, _set_field
from core.llm_client import LLMClient, LLMError
from ui.main_window import SettingsDialog
from ui.workers import FetchModelsWorker

PASS, FAIL = [], []


def check(label, ok):
    (PASS if ok else FAIL).append(label)
    print(('PASS ' if ok else 'FAIL ') + label)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        body = b'{"data":[{"id":"mock-model"}]}'
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.payloads.append(payload)
        if len(payload['messages']) > 8:
            time.sleep(0.2)
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            body = {'choices': [{'delta': {'content': 'Recovered'}}]}
            self.wfile.write(('data: ' + json.dumps(body) + '\n\ndata: [DONE]\n\n').encode())
            self.wfile.flush()
        except OSError:
            pass


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
server.daemon_threads = True
server.payloads = []
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f'http://127.0.0.1:{server.server_port}/v1'
cfg = Config(api_key='test', base_url=url, llm_use_system_proxy=False)
client = LLMClient(cfg)
check('默认模型读取等待为300秒', client._timeouts() == (10, 300))
_set_field(cfg, 'llm_read_timeout', '600')
_set_field(cfg, 'llm_use_system_proxy', 'false')
check('网络配置字符串可归一化', client._timeouts() == (10, 600) and cfg.llm_use_system_proxy is False)
check('子任务明确的短超时不被覆盖', client._timeouts((0.2, 0.08)) == (0.2, 0.08))

proxy_env = {'HTTP_PROXY': 'http://127.0.0.1:1', 'HTTPS_PROXY': 'http://127.0.0.1:1',
             'ALL_PROXY': '', 'NO_PROXY': 'not-this-host.invalid', 'no_proxy': 'not-this-host.invalid'}
with patch.dict(os.environ, proxy_env):
    check('直连模式不受坏代理影响', client.list_models() == ['mock-model'])
    proxy_client = LLMClient(Config(api_key='test', base_url=url, llm_use_system_proxy=True))
    proxy_client.RETRY_ON_CONN = 0
    try:
        list(proxy_client.chat_turn([{'role': 'user', 'content': 'hi'}], timeout=(0.2, 0.2)))
        proxy_failed = False
    except LLMError:
        proxy_failed = True
    check('系统代理模式仍尊重用户代理选择', proxy_failed)
    check('代理诊断包含地址且不包含凭据', '127.0.0.1:1' in proxy_client._route_label(url))
    check('测试不会改写系统代理或环境选择', os.environ['HTTP_PROXY'] == proxy_env['HTTP_PROXY'])

messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'ORIGINAL_TASK'}]
for i in range(8):
    messages.extend([{'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': f'call_{i}', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': f'call_{i}', 'content': 'output' * 1000}])
original = json.dumps(messages)
client.RETRY_ON_CONN, client.RETRY_BACKOFF = 1, 0
# Exercise the transport retry with a window large enough for the first request.
cfg.llm_context_limit = 65536
events = list(client.chat_turn(messages, timeout=(0.2, 0.08)))
check('读取超时后缩短历史并成功重试', any(e.get('text') == 'Recovered' for e in events)
      and len(server.payloads) == 2 and len(server.payloads[1]['messages']) < len(server.payloads[0]['messages']))
check('重试时可看到具体恢复状态', any(e['type'] == 'progress' and '缩短历史' in e['content'] for e in events))
reduced = server.payloads[1]['messages']
calls = {c['id'] for m in reduced for c in m.get('tool_calls', [])}
results = {m['tool_call_id'] for m in reduced if m['role'] == 'tool'}
check('压缩保留完整工具调用及结果', calls == results and len(calls) == 2)
check('原任务和调用方历史没有损坏', reduced[1]['content'] == 'ORIGINAL_TASK' and json.dumps(messages) == original)

app = QApplication.instance() or QApplication([])
dialog = SettingsDialog(Config())
dialog.e_llm_timeout.setValue(600)
dialog.c_llm_proxy.setCurrentIndex(1)
values = dialog.values()
check('设置窗口支持等待时间与直连选择', values['llm_read_timeout'] == 600 and values['llm_use_system_proxy'] is False)
captured = []
class FakeClient:
    def __init__(self, config): captured.append(config.llm_use_system_proxy)
    def list_models(self): return ['mock-model']
with patch('ui.workers.LLMClient', FakeClient):
    FetchModelsWorker('test', url, use_system_proxy=False).run()
check('获取模型也使用所选连接方式', captured == [False])
dialog.hide()
server.shutdown()
server.server_close()

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
raise SystemExit(1 if FAIL else 0)
