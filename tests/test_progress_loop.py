"""以真实脚本验证长修复循环；模型桩只决定下一次工具调用。"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.agent import Agent
from core.config import Config
from core.workspace import Workspace

PASS, FAIL = [], []


def check(label, ok):
    (PASS if ok else FAIL).append(label)
    print(('PASS ' if ok else 'FAIL ') + label)


class Model:
    def __init__(self, calls, protocol):
        self.calls, self.i = calls, 0
        self.cfg = Config(protocol=protocol, skills_enabled=False, web_enabled=False)

    def next(self):
        call = self.calls[min(self.i, len(self.calls) - 1)]
        self.i += 1
        return call

    def chat_turn(self, messages, tools=None):
        call = self.next()
        if isinstance(call, str):
            yield {'type': 'content', 'text': call}
        else:
            name, args = call
            yield {'type': 'tool_call', 'id': f'call_{self.i}', 'name': name,
                   'arguments': json.dumps(args)}

    def chat_stream(self, messages, **kwargs):
        call = self.next()
        yield ('ANSWER: ' + call if isinstance(call, str) else
               'ACTION: ' + json.dumps(dict(name=call[0], arguments=call[1])))


GOOD = ('import unittest\nclass Demo(unittest.TestCase):\n'
        '    def test_ok(self): self.assertEqual(2 + 2, 4)\n'
        "if __name__ == '__main__': unittest.main()\n")
BAD = "raise RuntimeError('broken')\n"
TASK = '创建 Python 测试脚本 test_app.py 并运行测试'


def write(content, path='test_app.py'):
    return ('write_file', dict(path=path, content=content))


def run(path='test_app.py'):
    return ('run_python', dict(path=path))


def execute(calls, protocol, task=TASK, stop=None):
    with tempfile.TemporaryDirectory(prefix='pyagent_progress_') as tmp:
        model = Model(calls, protocol)
        agent = Agent(model, Workspace(tmp), max_iterations=8)
        events = list(agent.run(task, confirm=lambda n, _: n == 'run_python', should_stop=stop))
        return agent, model, events


for protocol in ('native', 'text'):
    calls = []
    for i in range(12):
        calls.extend([write(BAD + f'# attempt {i}\n'), run()])
    calls.extend([write(GOOD), run(), '完成'])
    agent, model, events = execute(calls, protocol)
    check(f'{protocol}: 超过24轮修复仍能成功', model.i == 27 and events[-1]['type'] == 'answer')
    check(f'{protocol}: 验证来自真实运行', any(
        e['type'] == 'observation' and 'Ran 1 test' in e['content'] and 'OK' in e['content'] for e in events))

    agent, model, events = execute([write(BAD), run(), write(BAD), run()], protocol)
    check(f'{protocol}: 原样覆盖不能绕过失败去重', any(
        e['type'] == 'observation' and '禁止无修改重复运行' in e['content'] for e in events))
    check(f'{protocol}: 无进展有明确原因且不会无限执行', events[-1]['type'] == 'error'
          and '没有新的有效结果' in events[-1]['content'] and model.i < 40)

    agent, model, events = execute(['已经全部测试通过'], protocol)
    check(f'{protocol}: 零工具调用不能虚报成功', events[-1]['type'] == 'error'
          and not any(e['type'] == 'answer' for e in events))

    agent, model, events = execute([write(GOOD), run(), write(BAD), '完成'], protocol)
    check(f'{protocol}: 修改后旧测试结果失效', events[-1]['type'] == 'error')

    # 同版本首次测试成功、后续调用失败，也必须作废旧成功证据。
    agent, model, events = execute([write(GOOD), run(),
        ('run_python', dict(path='test_app.py', args='--invalid')), '完成'], protocol)
    check(f'{protocol}: 后续执行失败不能沿用成功', events[-1]['type'] == 'error')

    readme = '# 使用说明\n' + 'python app.py add task\npython app.py list\npython app.py done 1\n' * 4
    agent, model, events = execute([
        write(readme, 'README.md'), write(GOOD), run(),
        write('最后一步也执行了', 'result.md'), '完成'], protocol)
    check(f'{protocol}: 测试通过后继续执行其余步骤', model.i == 5 and events[-1]['type'] == 'answer')

    # A/B 两种内容反复切换不算持续进展。
    agent, model, events = execute([write(BAD), write(BAD + '# B\n')] * 25, protocol)
    check(f'{protocol}: 来回改写会识别为空转', events[-1]['type'] == 'error' and model.i < 50)

    counter = {'n': 0}
    def stop():
        counter['n'] += 1
        return counter['n'] > 5
    agent, model, events = execute([write(BAD)], protocol, stop=stop)
    check(f'{protocol}: 长循环可随时停止', events[-1]['type'] == 'cancelled')

    agent, model, events = execute([write(GOOD), run(), '完成'], protocol,
                                 task=TASK + '，还要编写 README.md')
    check(f'{protocol}: 文档缺失不能提前交付', events[-1]['type'] == 'error'
          and any(e['type'] == 'redirect' and 'README.md' in e['content'] for e in events))

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
raise SystemExit(1 if FAIL else 0)
