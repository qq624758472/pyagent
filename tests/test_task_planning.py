"""实际文件验证串行小任务、断点恢复、变更失效、文档汇总与技能编写加载。"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from core.agent import Agent
from core.config import Config
from core.task_plan import TaskPlan
from core.workspace import Workspace
from core.word_documents import inspect_docx
from core.llm_client import LLMClient
from core.skills import discover
from unittest.mock import patch
import core.sessions as sessions
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.skills_dialog import SkillsDialog

PASS,FAIL=[],[]
def check(name,ok):
    (PASS if ok else FAIL).append(name)
    print(('PASS ' if ok else 'FAIL ')+name)


def note(step):
    source=step['source'] or 'module0.c'
    lines=['# '+step['title'], '源码依据：'+source+':1-3']
    for title in ['职责与功能','接口输入输出','数据与结构','执行流程与算法','异常和边界','验证依据与待核实项']:
        lines += ['## '+title]
        for i in range(4):
            lines.append(f'{title}条目{i+1}：本测试模块通过step接口对计数器进行递增，并在limit边界停止。证据位于{source}，该文件没有定义调度频率，因此不推测硬件执行周期。')
    return '\n'.join(lines)


class Model:
    def __init__(self,protocol):
        self.cfg=Config(protocol=protocol,skills_enabled=False,web_enabled=False)
        self.agent=None
        self.i=0
        self.snapshots=[]
    def next(self,messages):
        self.i+=1
        self.snapshots.append(json.loads(json.dumps(messages)))
        plan=self.agent._plan
        step=plan.current()
        if not step: return '已按计划完成全部模块分析、汇总与检查。'
        if step['kind']=='analyze':
            source=step['source'] or 'module0.c'
            if source not in step['reads']: return ('read_file',dict(path=source))
            if step['artifact'] not in step['writes']: return ('write_file',dict(path=step['artifact'],content=note(step)))
            return ('update_task',dict(id=step['id'],status='completed'))
        if step['kind']=='assemble': return ('assemble_document',{})
        if step['kind']=='verify':
            if plan.data['target'] not in step['reads']: return ('read_file',dict(path=plan.data['target']))
            return ('update_task',dict(id=step['id'],status='completed'))
        raise AssertionError(step)
    def chat_turn(self,messages,tools=None):
        call=self.next(messages)
        if isinstance(call,str): yield dict(type='content',text=call)
        else: yield dict(type='tool_call',id=f'c{self.i}',name=call[0],arguments=json.dumps(call[1]))
    def chat_stream(self,messages,**kwargs):
        call=self.next(messages)
        yield 'ANSWER: '+call if isinstance(call,str) else 'ACTION: '+json.dumps(dict(name=call[0],arguments=call[1]))


app=QApplication([])
with tempfile.TemporaryDirectory(prefix='pyagent_plan_') as tmp:
    root=Path(tmp)
    sessions.SESSIONS_DIR=root/'sessions'
    for protocol in ('native','text'):
        folder=root/protocol
        folder.mkdir()
        ws=Workspace(folder)
        for i in range(4): ws.write_file(f'module{i}.c',f'int step{i}(int value, int limit) {{ return value < limit ? value + 1 : value; }}\n')
        model=Model(protocol)
        agent=Agent(model,ws)
        model.agent=agent
        events=list(agent.run('读取工程源码并编写详细设计方案，叫report.docx'))
        plan=agent._plan
        info=inspect_docx(folder/'report.docx')
        check(protocol+' 自动拆分每个模块并串行完成',len(plan.data['steps'])==8 and plan.current() is None and events[-1]['type']=='answer')
        check(protocol+' 长于8轮的任务正常完成',model.i>20)
        check(protocol+' 文档汇总覆盖所有模块且不是短概述',info['characters']>=plan.minimum_document() and all(f'module{i}.c' in info['text'] for i in range(4)))
        check(protocol+' 先保存计划再执行模型工具',events[0]['type']=='plan' and (plan.directory/'plan.json').exists())
        check(protocol+' 临时笔记都在隐藏工作目录',all(s['artifact'].startswith('.agent/tasks/') for s in plan.data['steps'] if s['kind']=='analyze')
              and sorted(p.name for p in folder.iterdir() if p.is_file())==['module0.c','module1.c','module2.c','module3.c','report.docx'])
        check(protocol+' 默认列表与搜索不混入自身笔记','.agent' not in ws.list_files('.',True) and '.agent' not in ws.search_files('limit',content=True))
        check(protocol+' 可显式读取自身笔记','职责' in ws.read_file(plan.data['steps'][0]['artifact']))

        # 内容真正变化时，先前步骤及下游必须失效。
        ws.append_file('module1.c','\n// changed boundary\n')
        issue=plan.completion_issue()
        check(protocol+' 源码变化使受影响步骤及后续失效','验收未通过' in issue and plan.current()['source']=='module1.c')
        check(protocol+' 旧读取和旧笔记不能重复作为新证据',not plan.current()['reads'] and not plan.current()['writes'])

        # 新任务中断后恢复同一个 plan_id，而不是再次开始所有步骤。
        m2=Model(protocol); a2=Agent(m2,ws); m2.agent=a2
        list(a2.run('读取工程源码并编写详细方案 report.docx',should_stop=lambda: a2._plan is not None and sum(s['status']=='completed' for s in a2._plan.data['steps'])>=3))
        saved=dict(messages=a2.messages,continuity=a2.session_memory())
        old_id=a2._plan.data['id']
        m3=Model(protocol); a3=Agent(m3,ws); m3.agent=a3
        a3.restore_session(saved)
        resumed=list(a3.run('继续'))
        check(protocol+' 恢复后续做原计划而非重建',a3._plan.data['id']==old_id and resumed[-1]['type']=='answer')
        check(protocol+' 恢复首次请求已保留已完成进度','s004 [in_progress]' in json.dumps(m3.snapshots[0],ensure_ascii=False))

        pending=TaskPlan.start(ws,'读取源码并编写完整设计方案 report.docx',True)
        check(protocol+' 不能跳过步骤提前标记完成',pending.execute('update_task',dict(id='s002',status='completed')).startswith('错误'))
        check(protocol+' 未生成分析笔记不能宣称完成',pending.execute('update_task',dict(id='s001',status='completed')).startswith('错误'))
        check(protocol+' 禁止把中间草稿写到工作区根目录',bool(pending.guard('write_file',dict(path='draft.md'))))
        check(protocol+' 不能直接伪造计划完成状态',bool(pending.guard('write_file',dict(path=f'.agent/tasks/{pending.data["id"]}/plan.json'))))
        check(protocol+' 不能提前汇总短文交付',pending.execute('assemble_document',{}).startswith('错误'))
        check(protocol+' 指定文件名解析正确',pending.data['target']=='report.docx')
        brief=TaskPlan.start(ws,'生成简短的一页概览 report.docx',True)
        check(protocol+' 尊重明确要求的简短文档',not brief.data['strict'])
        if protocol=='native':
            valid_protocol=True
            for snapshot in model.snapshots:
                ids={call['id'] for message in snapshot for call in message.get('tool_calls',[])}
                valid_protocol &= all(message.get('tool_call_id') in ids for message in snapshot if message.get('role')=='tool')
            check('小任务上下文隔离保持工具协议完整',valid_protocol)

    generic=TaskPlan.start(Workspace(root/'native'),'开发复杂应用并实现数据导入和查询',False)
    check('一般复杂任务先细化而非直接开工',generic.data['needs_refinement'] and 'set_plan' in generic.guard('write_file',dict(path='app.py')))
    result=generic.execute('set_plan',dict(steps=[dict(title='明确数据格式和验收样例',kind='analyze',criteria='保存需求笔记'),
        dict(title='实现导入与查询接口',kind='implement',criteria='保存实现文件'),dict(title='运行样例和异常测试',kind='verify',criteria='真实执行成功')]))
    check('模型可通过计划工具细化一般任务',not result.startswith('错误') and not generic.data['needs_refinement'])
    large=root/'large'; large.mkdir()
    for i in range(4): (large/f'big{i}.c').write_text(('/* source reference */\n'*4000),encoding='utf-8')
    large_plan=TaskPlan.start(Workspace(large),'编写详细设计方案 report.docx',True)
    check('大型模块继续拆分接口流程异常三个小任务',len([s for s in large_plan.data['steps'] if s['source']=='big0.c'])==3)
    id_plan=TaskPlan.start(Workspace(root/'native'),'编写详细设计方案 report.docx',True)
    custom=[dict(id=f'task-{i+1}',title=s['title'],kind=s['kind'],source=s['source'],criteria=s['criteria']) for i,s in enumerate(id_plan.data['steps'])]
    result=id_plan.execute('set_plan',dict(steps=custom))
    check('模型自定义的小任务编号会被真实保留',not result.startswith('错误') and id_plan.current()['id']=='task-1' and id_plan.current()['artifact'].endswith('/task-1.md'))
    duplicate=TaskPlan.start(Workspace(root/'native'),'编写详细设计方案 report.docx',True)
    custom[1]['id']=custom[0]['id']
    check('拒绝重复编号避免覆盖其他小任务笔记','编号必须唯一' in duplicate.execute('set_plan',dict(steps=custom)))

    ui_ws=Workspace(root/'native')
    win=MainWindow(Config())
    win.agent.set_workspace(ui_ws)
    check('技能和任务计划按钮在主界面可见',win.b_skills_load.text()=='加载技能' and win.b_skills_write.text()=='编写技能' and win.b_task_plan.text()=='任务计划与进度')
    editor=SkillsDialog(win,edit=True)
    editor.name.setText('design-notes')
    editor.description.setText('编写详细设计文档时使用')
    editor.body.setPlainText('## 执行步骤\n\nMANUAL_RULE 先拆分小任务，每个模块说明接口输入输出与异常处理，逐项保存证据并验证。\n')
    name,path=editor.write_skill()
    check('技能保存到当前工作区.agent',path==ui_ws.root/'.agent/skills/design-notes/SKILL.md' and path.exists())
    check('工作区技能可以被发现',any(s.name==name and s.source=='workspace' for s in discover(win.config,ui_ws)))
    global_home=root/'global-home'
    global_skill=global_home/'.agents/skills/design-notes/SKILL.md'
    global_skill.parent.mkdir(parents=True)
    global_skill.write_text('---\nname: design-notes\ndescription: global copy\n---\nGLOBAL_RULE\n',encoding='utf-8')
    with patch('core.skills.os.path.expanduser',return_value=str(global_home)):
        selected=next(s for s in discover(win.config,ui_ws) if s.name==name)
        check('工作区编写的副本优先于全局同名技能','MANUAL_RULE' in selected.body and selected.source=='workspace')
    editor.refresh()
    row=next(i for i,s in enumerate(editor.skills) if s.name==name)
    editor.list.setCurrentRow(row)
    editor.load_skill()
    check('加载按钮选择技能进入当前会话',name in win.agent.session_memory()['selected_skills'])
    class ManualModel:
        cfg=win.config
        snapshots=[]
        def chat_turn(self,messages,tools=None):
            self.snapshots.append(messages)
            yield dict(type='content',text='规范已加载')
    manual=ManualModel()
    win.agent.set_llm(manual)
    manual_events=list(win.agent.run('解释技能规范'))
    check('手动技能会实际提供给模型而不只是界面提示','MANUAL_RULE' in json.dumps(manual.snapshots[0],ensure_ascii=False)
          and any(e['type']=='skill_loaded' for e in manual_events))
    editor.hide(); win.hide()

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
sys.exit(bool(FAIL))
