"""复现 skill_report_writer 名称混淆，并验证技能去重、回传与无进展恢复。"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import skills
from core.agent import Agent
from core.config import Config
from core.tools import execute_tool, openai_tools
from core.workspace import Workspace

PASS, FAIL = [], []


def check(label, ok):
    (PASS if ok else FAIL).append(label)
    print(('PASS ' if ok else 'FAIL ') + label)


class Model:
    def __init__(self, calls, protocol):
        self.cfg = Config(protocol=protocol, skills_enabled=True, web_enabled=False)
        self.calls, self.index = calls, 0
        self.snapshots = []

    def next(self, messages):
        self.snapshots.append([dict(m) for m in messages])
        call = self.calls[min(self.index, len(self.calls)-1)]
        self.index += 1
        return call

    def chat_turn(self, messages, tools=None):
        call = self.next(messages)
        if isinstance(call, str):
            yield dict(type='content', text=call)
        else:
            yield dict(type='tool_call', id=f'call_{self.index}', name=call[0], arguments=json.dumps(call[1]))

    def chat_stream(self, messages, **kwargs):
        call = self.next(messages)
        yield 'ANSWER: ' + call if isinstance(call, str) else 'ACTION: ' + json.dumps(dict(name=call[0], arguments=call[1]))


with tempfile.TemporaryDirectory(prefix='pyagent_skill_recovery_') as tmp:
    base = Path(tmp)
    skill_root = base / 'skills'
    root = skill_root / 'report-writer'
    root.mkdir(parents=True)
    md = root / 'SKILL.md'
    md.write_text('---\nname: report-writer\ndescription: 报告方案规范\n---\n\nRULE_ONE 使用完整正文并核对。', encoding='utf-8')
    (root / 'reference.txt').write_text('REFERENCE', encoding='utf-8')
    workspace_root = base / 'workspace'
    workspace_root.mkdir()
    ws = Workspace(workspace_root)
    cfg = Config()
    with patch.object(skills, 'skill_roots', return_value=[(skill_root, 'project')]):
        for name in ('report-writer', 'report_writer', 'skill_report_writer', 'SKILL_REPORT_WRITER',
                     'project:report-writer', '`skill_report_writer`'):
            result = execute_tool('load_skill', ws, cfg, dict(name=name))
            check(f'归一名称 {name}', 'RULE_ONE' in result and not result.startswith('错误'))
        result = execute_tool('load_skill', ws, cfg, dict(name='skill_report_writer', path='reference.txt'))
        check('别名可读取技能附件', 'REFERENCE' in result)
        result = execute_tool('load_skill', ws, cfg, dict(name='skill_report_writer', path='../outside.txt'))
        check('别名不能绕过附件路径保护', result.startswith('错误'))
        result = execute_tool('load_skill', ws, cfg, dict(name='skill_missing'))
        check('未知名称不会猜测其他技能', result.startswith('错误') and 'RULE_ONE' not in result)
        result = execute_tool('load_skill', ws, Config(skills_enabled=False), dict(name='skill_report_writer'))
        check('关闭技能时不会通过别名绕过开关', result.startswith('错误'))
        entry = next(t for t in openai_tools(cfg=cfg, ws=ws) if t['function']['name'] == 'load_skill')
        check('通用加载器明确列出准确名称', 'report-writer' in entry['function']['description'])

        for protocol in ('native', 'text'):
            calls = [('load_skill', dict(name='skill_report_writer')),
                     ('load_skill', dict(name='report_writer')),
                     ('skill_report_writer', {}),
                     ('write_file', dict(path='result.md', content='Generated according to RULE_ONE')), '完成']
            model = Model(calls, protocol)
            agent = Agent(model, ws)
            # 避免任务预匹配掩盖显式加载的测试。
            events = list(agent.run('整理资料并保存 result.md'))
            observations = [e['content'] for e in events if e['type'] == 'observation']
            check(f'{protocol}: 用户给出的错误名称实际可用', 'RULE_ONE' in observations[0])
            check(f'{protocol}: 不同别名加载同一技能被去重', sum('无需重复加载' in text for text in observations) == 2)
            check(f'{protocol}: 加载后仍能继续实际写文件', events[-1]['type'] == 'answer'
                  and (workspace_root / 'result.md').read_text() == 'Generated according to RULE_ONE')
            check(f'{protocol}: 下一轮模型看得到真实工具结果', any(
                'RULE_ONE' in (m.get('content') or '') for m in model.snapshots[1]))

            model = Model([('load_skill', dict(name='skill_missing')), '无法加载该技能'], protocol)
            agent = Agent(model, ws)
            list(agent.run('询问一个不存在的技能'))
            check(f'{protocol}: 加载失败不会被声明已加载', not any(
                '已加载技能规范：load_skill' in (m.get('content') or '') for m in agent.messages))

            # 先反复加载，再只有恢复提示出现时模型才转向实际产出。
            class RecoveringModel(Model):
                wrote = False
                def next(self, messages):
                    self.snapshots.append([dict(m) for m in messages])
                    self.index += 1
                    if self.wrote:
                        return '完成'
                    if any('自动恢复' in (m.get('content') or '') for m in messages):
                        self.wrote = True
                        return 'write_file', dict(path='recovered.md', content='RECOVERED')
                    return 'load_skill', dict(name='skill_report_writer')
            model = RecoveringModel([], protocol)
            agent = Agent(model, ws)
            events = list(agent.run('整理资料并保存 recovered.md'))
            check(f'{protocol}: 首次连续16轮无进展先恢复', any(
                e['type'] == 'redirect' and '换策略' in e['content'] for e in events))
            check(f'{protocol}: 恢复后仍可完成实际产物', events[-1]['type'] == 'answer'
                  and (workspace_root / 'recovered.md').read_text() == 'RECOVERED')
            check(f'{protocol}: 恢复保留技能规则与原任务', any(
                '自动恢复' in (m.get('content') or '') and 'RULE_ONE' in m['content']
                and 'recovered.md' in m['content'] for snapshot in model.snapshots for m in snapshot))

            # 任务间不沿用「已加载」缓存。
            agent.llm = Model([('load_skill', dict(name='report-writer')), '完成'], protocol)
            events = list(agent.run('整理材料'))
            check(f'{protocol}: 新任务可重新加载技能', any(
                e['type'] == 'observation' and 'RULE_ONE' in e['content'] for e in events))

        ambiguous = [skills.Skill('a.b', '', md, root), skills.Skill('a b', '', md, root)]
        check('冲突别名不会任选一个技能', skills.find(ambiguous, 'skill_a_b') is None)

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
raise SystemExit(1 if FAIL else 0)
