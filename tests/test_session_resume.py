"""真正落盘后新建 Agent，验证两个协议的续做和 GUI 保存/恢复路径。"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PyQt5.QtWidgets import QApplication
from core.agent import Agent
from core.config import Config
from core.workspace import Workspace
from core.session_context import is_continue, normalize_messages
import core.sessions as sessions
from ui.main_window import MainWindow

PASS, FAIL = [], []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(('PASS ' if ok else 'FAIL ') + name)


class Model:
    def __init__(self, calls, protocol):
        self.calls, self.i, self.snapshots = calls, 0, []
        self.cfg = Config(protocol=protocol, skills_enabled=False, web_enabled=False)

    def next(self, messages):
        self.snapshots.append(json.loads(json.dumps(messages)))
        call = self.calls[min(self.i, len(self.calls) - 1)]
        self.i += 1
        return call

    def chat_turn(self, messages, tools=None):
        call = self.next(messages)
        if isinstance(call, str):
            yield dict(type='content', text=call)
        else:
            yield dict(type='tool_call', id=f'c{self.i}', name=call[0], arguments=json.dumps(call[1]))

    def chat_stream(self, messages, **kwargs):
        call = self.next(messages)
        yield 'ANSWER: ' + call if isinstance(call, str) else 'ACTION: ' + json.dumps(dict(name=call[0], arguments=call[1]))


app = QApplication([])
GOOD = 'import unittest\nclass T(unittest.TestCase):\n def test_ok(self): self.assertEqual(2+2,4)\nunittest.main()\n'
BAD = "raise RuntimeError('RESUME_FAILURE')\n"
with tempfile.TemporaryDirectory(prefix='pyagent_resume_') as tmp:
    root = Path(tmp)
    sessions.SESSIONS_DIR = root / 'sessions'
    for protocol in ('native', 'text'):
        ws = root / protocol
        ws.mkdir()
        (ws / 'keep.txt').write_text('KEEP_EXISTING', encoding='utf-8')
        model = Model([('write_file', dict(path='test_app.py', content=BAD)),
                       ('run_python', dict(path='test_app.py'))], protocol)
        agent = Agent(model, Workspace(ws))
        list(agent.run('创建 Python 测试脚本 test_app.py 并运行测试；保留 keep.txt',
                       should_stop=lambda: bool(getattr(agent, '_last_failure', '')), confirm=lambda *_: True))
        sessions.save_session(protocol, agent.messages, str(ws), '续做', agent.session_memory())
        data = sessions.load_session(protocol)
        check(protocol + ' 保存失败与已写文件', 'RESUME_FAILURE' in data['continuity']['last_failure']
              and 'test_app.py' in data['continuity']['files'])
        model2 = Model([('read_file', dict(path='test_app.py')),
                        ('write_file', dict(path='test_app.py', content=GOOD)),
                        ('run_python', dict(path='test_app.py')), '修复并验证完成'], protocol)
        restored = Agent(model2, Workspace(ws))
        restored.restore_session(data)
        events = list(restored.run('继续', confirm=lambda *_: True))
        sent = json.dumps(model2.snapshots[0], ensure_ascii=False)
        check(protocol + ' 首次请求自动带目标约束错误和路径', all(s in sent for s in
              ('保留 keep.txt', 'RESUME_FAILURE', str(ws).replace('\\', '\\\\'), 'test_app.py', '不要从头重做')))
        check(protocol + ' 真实修复后重测完成', events[-1]['type'] == 'answer' and any(
            e['type'] == 'observation' and 'Ran 1 test' in e['content'] for e in events))
        check(protocol + ' 不重建已有其他文件', (ws / 'keep.txt').read_text() == 'KEEP_EXISTING')
        check(protocol + ' 不把交接提示累积成新目标', len(restored.session_memory()['requests']) == 1)
        restored.reset()
        check(protocol + ' 清空会话不会带入旧任务', not restored.session_memory().get('requests'))

    # 用 GUI 真正保存一次，覆盖原来 GUI 丢字段但底层保存函数正常的漏测。
    win = MainWindow(Config())
    win.agent.restore_session(sessions.load_session('native'))
    win._session_id = 'gui-roundtrip'
    win._save_current_session()
    gui_data = sessions.load_session('gui-roundtrip')
    check('GUI 保存空正文 assistant 工具调用', any(m.get('tool_calls') and not m.get('content') for m in gui_data['messages']))
    check('GUI 保存 tool_call_id', any(m.get('tool_call_id') for m in gui_data['messages']))
    win._open_session_by_id('text')
    check('手动打开保留 OBSERVATION', any(str(m.get('content')).startswith('OBSERVATION') for m in win.agent.messages))
    win.agent.reset()
    win._restore_session()
    check('自动恢复交接目标', bool(win.agent.session_memory()['requests']))
    win.hide()

    old = Agent(Model(['继续检查'], 'text'))
    old.restore_session(dict(messages=[['user', '分析 ACS 并写报告，保留原文件'],
                                       ['user', 'OBSERVATION: 已读取 ACS/a.c'],
                                       ['assistant', '还需补充报告']]))
    list(old.run('继续'))
    check('旧格式会话自动迁移并携带原要求', '保留原文件' in json.dumps(old.llm.snapshots[0], ensure_ascii=False))
    check('恢复残缺工具调用不向 API 发送非法组', not any(m.get('tool_calls') or m['role'] == 'tool' for m in normalize_messages([
        dict(role='assistant', content=None, tool_calls=[dict(id='lost', function=dict(name='write_file', arguments='{}'))])])) )
    check('旧版孤立工具结果保留为普通执行记录', normalize_messages([
        dict(role='tool', tool_call_id='lost', content='OLD_FAILURE')])[0]['content'].endswith('OLD_FAILURE'))
    check('普通新任务不误判为续做', not is_continue('继续开发一个新项目') and is_continue('请继续。'))
    # 现有文档通过检查就可继续交付，不能因本轮未重写而被迫从头生成。
    doc_ws = root / 'documents'
    doc_ws.mkdir()
    Workspace(doc_ws).write_file('report.docx', '完整的技术方案正文。' * 40)
    stamp = (doc_ws / 'report.docx').stat().st_mtime_ns
    for protocol in ('native', 'text'):
        doc_model = Model([('read_file', dict(path='report.docx')), '已检查现有文档'], protocol)
        doc_agent = Agent(doc_model, Workspace(doc_ws))
        doc_agent.restore_session(dict(messages=[dict(role='user', content='撰写 Word 技术方案 report.docx')],
                                      continuity=dict(requests=['撰写 Word 技术方案 report.docx'], results=[], files=['report.docx'])))
        doc_events = list(doc_agent.run('继续'))
        check(protocol + ' 已有合格文档不必重写', doc_events[-1]['type'] == 'answer'
              and (doc_ws / 'report.docx').stat().st_mtime_ns == stamp and doc_model.i == 2)

    huge = Agent(Model(['收到'], 'text'))
    huge.messages = [dict(role='system', content='s'), dict(role='user', content='原目标'),
                     dict(role='assistant', content='x'*100), dict(role='user', content='本次续做目标')]
    huge.KEEP_CHARS = 5
    huge._trim_history()
    check('预算不足仍保留本次续做目标', huge.messages[-1]['content'] == '本次续做目标')
    huge.messages += [dict(role='assistant', content='old output'*100) for _ in range(50)]
    huge._trim_history()
    check('保留任务时仍能裁剪长循环历史', len(huge.messages) <= 4 and huge.messages[2]['content'] == '本次续做目标')

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
sys.exit(bool(FAIL))
