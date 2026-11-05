"""Provider discovery, short-context fitting and safe recovery."""
import json
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent import Agent
from core.config import Config
from core.context_budget import ContextBudget, estimate_tokens, fit_context
from core.llm_client import LLMClient
from core.tools import openai_tools
from core.workspace import Workspace


class ContextTests(unittest.TestCase):
    def client(self, **kwargs):
        return LLMClient(Config(api_key='test', model='qwen', **kwargs))

    def test_discovery_and_manual_cap(self):
        client = self.client()
        client.base_url = 'http://localhost:11434/v1'
        with patch.object(client, '_model_metadata', return_value={'models': [
                {'name': 'different:latest', 'context_length': 32768},
                {'name': 'qwen:latest', 'context_length': 4096}]}) as probe:
            budget = client.context_budget()
            self.assertEqual((budget.window, budget.input_limit, budget.output), (4096, 2816, 1024))
            client.cfg.llm_context_limit = 2048
            self.assertEqual(client.context_budget().window, 2048)
            probe.assert_called_once_with('GET', 'http://localhost:11434/api/ps')
            self.assertNotEqual(client.context_budget('different').window, 4096)

        client = self.client()
        client.base_url = 'http://localhost:11434/v1'
        show = {'parameters': 'temperature 0.2\nnum_ctx 16384',
                'model_info': {'qwen.context_length': 131072}}
        with patch.object(client, '_model_metadata', side_effect=[{'models': []}, show]):
            self.assertEqual(client.context_budget().window, 16384)
        with patch.object(client, '_model_metadata', side_effect=[{'models': []}, dict(show, parameters='')]):
            budget = client.context_budget('another')
            self.assertEqual(budget.window, 4096)
            self.assertIn('保守', budget.source)
        client.base_url = 'http://localhost/v1/custom'
        with patch.object(client, '_model_metadata', side_effect=[{}, {'data': [
                {'id': 'other', 'max_model_len': 8192}, {'id': 'qwen', 'context_window': 2048}]}]):
            self.assertEqual(client.context_budget().window, 2048)

    def test_short_prompt_preserves_task_plan_and_native_evidence(self):
        budget = ContextBudget(2048, 'test', 512)
        messages = [
            {'role': 'system', 'content': '历史说明' * 1000, '_context_short': '工具执行助手'},
            {'role': 'user', 'content': '处理当前项目', '_context_required': True},
            {'role': 'assistant', 'content': '旧内容' * 2000},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'read_1', 'type': 'function', 'function': {
                    'name': 'read_file', 'arguments': '{"path":"main.c"}'}}]},
            {'role': 'tool', 'tool_call_id': 'read_1', 'content': 'int main(void) { return 0; }'},
            {'role': 'user', 'content': '当前子任务：验证 main.c', '_context_required': True},
        ]
        fitted, tools, _ = fit_context(messages, openai_tools(), budget)
        self.assertLessEqual(estimate_tokens({'messages': fitted, 'tools': tools}), budget.input_limit)
        self.assertIn('read_file', [tool['function']['name'] for tool in tools])
        self.assertEqual([m['tool_call_id'] for m in fitted if m['role'] == 'tool'], ['read_1'])
        self.assertEqual([c['id'] for m in fitted for c in m.get('tool_calls', [])], ['read_1'])
        self.assertIn('处理当前项目', [m['content'] for m in fitted])
        self.assertIn('当前子任务：验证 main.c', [m['content'] for m in fitted])
        self.assertNotIn('_context_', json.dumps(fitted))

    def test_unicode_pagination_continues_without_skipping(self):
        lines = ['第一行', '中😀' * 100, '末尾一行']
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(directory)
            ws.write_file('src.txt', '\n'.join(lines))
            line, column = 1, 1
            reconstructed = ''
            for _ in range(16):
                page = ws.read_file('src.txt', line_start=line, column_start=column, max_bytes=128)
                body = page.split('\n', 1)[1]
                cursor = re.search(r'\n下一页：.*line_start=(\d+)，column_start=(\d+)', body)
                if not cursor:
                    reconstructed += body
                    break
                body = body[:cursor.start()]
                next_line, column = map(int, cursor.groups())
                reconstructed += body + ('\n' if next_line > line + body.count('\n') else '')
                line = next_line
            self.assertEqual(reconstructed, '\n'.join(lines))

    def test_metadata_before_completion_and_retries_without_partial_write(self):
        calls = []
        requests = []

        class Server(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body, mime='application/json'):
                self.send_response(status)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                calls.append(self.path)
                self.reply(200, b'{"models":[{"name":"qwen:latest","context_length":8192}]}')

            def do_POST(self):
                calls.append(self.path)
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                if len(requests) == 1:
                    self.reply(400, b'{"error":"maximum context length is 4096 tokens"}')
                    return
                if len(requests) == 2:
                    choice = {'delta': {'tool_calls': [
                        {'index': 0, 'id': 'write_1', 'function': {
                            'name': 'write_file', 'arguments': '{"path":"partial.txt","content":"unsafe"}'}}]}}
                    choices = [choice, {'finish_reason': 'length'}]
                else:
                    choices = [{'delta': {'content': '可以继续。'}, 'finish_reason': 'stop'}]
                body = ''.join('data: ' + json.dumps({'choices': [choice]}) + '\n\n' for choice in choices)
                self.reply(200, (body + 'data: [DONE]\n\n').encode(), 'text/event-stream; charset=utf-8')

        with ThreadingHTTPServer(('127.0.0.1', 0), Server) as server, tempfile.TemporaryDirectory() as directory:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = self.client(base_url=f'http://127.0.0.1:{server.server_port}/v1',
                                     protocol='native', llm_use_system_proxy=False)
                agent = Agent(client)
                agent.workspace = Workspace(directory)
                events = list(agent.run('打个招呼'))
                self.assertEqual(calls, ['/api/ps'] + ['/v1/chat/completions'] * 3)
                self.assertEqual([r['max_tokens'] for r in requests], [2048, 1024, 1024])
                self.assertFalse(Path(directory, 'partial.txt').exists())
                self.assertFalse(any(event['type'] == 'action' for event in events))
                self.assertEqual(events[-1]['type'], 'answer')
                self.assertTrue(any('只生成一个短段落' in str(m.get('content'))
                                    for m in requests[-1]['messages']))
            finally:
                server.shutdown()


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContextTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = len(result.failures) + len(result.errors)
    print(f'结果: {result.testsRun - failed} 通过, {failed} 失败')
    raise SystemExit(not result.wasSuccessful())
