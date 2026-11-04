"""复现不断换行号读取函数清单的空转，验证实际证据、阶段推进和搜索范围。"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.agent import Agent
from core.config import Config
from core.workspace import Workspace
from core.word_documents import inspect_docx
from core.research_state import ResearchState
from core.tools import execute_tool, openai_tools

PASS, FAIL = [], []
def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(('PASS ' if ok else 'FAIL ') + name)


class ReadingModel:
    def __init__(self, protocol, stubborn=False):
        self.cfg = Config(protocol=protocol, skills_enabled=False, web_enabled=False)
        self.i, self.stubborn = 0, stubborn
        self.snapshots, self.tools = [], []
        self.written = False

    def next(self, messages, tools=None):
        self.snapshots.append(json.loads(json.dumps(messages)))
        self.tools.append(tools)
        self.i += 1
        if self.written:
            return '已生成并核对'
        if not self.stubborn and any('本阶段暂不继续读取' in str(m.get('content')) for m in messages):
            self.written = True
            return ('write_file', dict(path='方案.docx', content='根据源码可确认本模块包含数据结构和主入口。' * 20))
        return ('read_file', dict(path='functions.txt', line_start=1+(self.i%5), line_end=80+(self.i%9)))

    def chat_turn(self, messages, tools=None):
        call = self.next(messages, tools)
        if isinstance(call, str): yield dict(type='content', text=call)
        else: yield dict(type='tool_call', id=f'c{self.i}', name=call[0], arguments=json.dumps(call[1]))

    def chat_stream(self, messages, **kwargs):
        call = self.next(messages)
        yield 'ANSWER: '+call if isinstance(call,str) else 'ACTION: '+json.dumps(dict(name=call[0],arguments=call[1]))


with tempfile.TemporaryDirectory(prefix='pyagent_research_') as tmp:
    ws = Workspace(tmp)
    p = Path(tmp)
    (p/'functions.txt').write_text('\n'.join(f'void ModuleMain_{i}(struct Data *data);' for i in range(300)), encoding='utf-8')
    (p/'one.h').write_text('struct adcs_control_actions {\n int unique_one;\n};\n',encoding='utf-8')
    (p/'two.h').write_text('struct adcs_control_actions {\n int unique_two;\n};\n',encoding='utf-8')
    single = ws.search_files('adcs_control_actions', content=True, path='one.h')
    check('指定文件搜索不回退全工作区', 'one.h:1:' in single and 'two.h' not in single and '扫描 1 个' in single)
    result = execute_tool('search_files',ws,Config(),dict(pattern=r'struct\s+adcs_control_actions\s*\{',content=True,path='one.h',regex=True))
    check('正则内容搜索支持结构体定位', 'one.h:1:' in result and 'two.h' not in result)
    check('兼容模型省略 regex 的明确正则转义', 'one.h:1:' in ws.search_files(r'struct\s+adcs_control_actions\s*\{',content=True,path='one.h'))
    check('无效正则明确报错', ws.search_files('[',content=True,regex=True).startswith('错误'))
    check('普通括号按字面匹配', '未找到' in ws.search_files('missing()',content=True,path='one.h'))
    check('搜索工具向模型说明 regex 参数', 'regex' in next(t for t in openai_tools() if t['function']['name']=='search_files')['function']['parameters']['properties'])

    state = ResearchState('读取代码并编写Word方案.docx')
    body = ws.read_file('functions.txt',1,80)
    check('首次看到内容算进展', state.observe('read_file',dict(path='functions.txt'),body,ws))
    overlap = ws.read_file('functions.txt',20,60)
    check('子区间重复读取不算新进展', state.observe('read_file',dict(path='functions.txt',line_start=20),overlap,ws) is False)
    check('真正新增行仍可算进展', state.observe('read_file',dict(path='functions.txt'),ws.read_file('functions.txt',75,90),ws))
    check('无需保留旧工具大输出也能携带资料摘录', 'ModuleMain_' in state.context() and len(state.context()) < 5000)
    saved = state.snapshot()
    check('恢复会话仍有资料摘录', 'ModuleMain_' in ResearchState('写Word方案',saved).context())
    restored = ResearchState('写Word方案',saved)
    check('恢复后仍识别已覆盖内容', restored.observe('read_file',dict(path='functions.txt'),overlap,ws) is False)
    for _ in range(3):
        restored.observe('read_file',dict(path='functions.txt'),overlap,ws)
    resumed = ResearchState('写Word方案',restored.snapshot())
    check('恢复后保留待产出阶段而非重新收集', resumed.drafting)
    resumed.authored()
    check('形成实际草稿后可以继续查缺补漏', not resumed.drafting and resumed.calls==0)

    for protocol in ('native','text'):
        model = ReadingModel(protocol)
        agent = Agent(model,ws,max_iterations=8)
        events = list(agent.run('当前工作区：agent测试目录。用户任务：读取代码并编写设计方案.docx'))
        check(protocol+' 重复读转为实际生成并通过验证', events[-1]['type']=='answer' and (p/'方案.docx').exists()
              and inspect_docx(p/'方案.docx')['characters']>120 and model.i<20)
        check(protocol+' 文档任务检查点不展示代码验证失败', all('已验证版本' not in e.get('details','') for e in events if e['type']=='checkpoint'))
        check(protocol+' 任务交接持久保存资料记录', bool(agent.session_memory().get('research',{}).get('notes')))
        agent._task = '当前工作区：agent测试目录。用户任务：分析源码并撰写Word技术方案，不要修改代码或运行测试。'
        check(protocol+' 否定测试要求不触发编程验收', not agent._completion_issue(agent._task))
        if protocol=='native':
            phase_tools = model.tools[-2]
            check('生成阶段工具列表推动模型产出', not any(t['function']['name']=='read_file' for t in phase_tools))
            check('纯文档任务不提供 shell 或脚本执行工具', not any(t['function']['name'] in ('run_command','run_python') for t in model.tools[0]))
        check(protocol+' 首次请求说明无需依赖的原生Word生成方式', '无需 shell' in json.dumps(model.snapshots[0],ensure_ascii=False))
        stubborn = ReadingModel(protocol,stubborn=True)
        blocked = Agent(stubborn,ws)
        failed = list(blocked.run('读取代码并撰写Word设计方案.docx'))
        check(protocol+' 无视产出提示也不能空转108轮', failed[-1]['type']=='error' and stubborn.i<60)
        (p/'方案.docx').unlink(missing_ok=True)

    # 长而真实的资料收集可以跨多轮推进：每段形成产物后继续收集，不限制总轮数。
    stage = ResearchState('编写 Word 报告')
    for i in range(40):
        stage.observe('read_file',dict(path='functions.txt'),f'NEW_FACT_{i}',ws)
        if stage.drafting:
            stage.authored()
    check('超过总轮数仍允许有效资料收集与阶段产出', stage.calls<12 and len(stage.seen)==40)

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
sys.exit(bool(FAIL))
