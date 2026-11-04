"""DOCX 真实格式、损坏文件复现、无损失败和双协议交付门禁。"""
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.agent import Agent
from core.config import Config
from core.tools import execute_tool
from core.workspace import Workspace
from core.word_documents import DocumentError, build_docx, inspect_docx

PASS, FAIL = [], []


def check(label, ok):
    (PASS if ok else FAIL).append(label)
    print(('PASS ' if ok else 'FAIL ') + label)


def invalid(data):
    try:
        inspect_docx(data)
    except DocumentError:
        return True
    return False


CONTENT = '# ACS 设计方案\n\n## 范围\n中文 & <标签> **重点** `code`\n- 条目一\n- 条目二\n\n| 模块 | 职责 |\n| --- | --- |\n| 控制 | 控制流程 |\n'
CONTENT += '\n## 设计说明\n' + '本节描述已知模块的功能边界、输入输出与验证方式，不推断未提供的性能数据。' * 5

with tempfile.TemporaryDirectory(prefix='pyagent_word_') as tmp:
    ws = Workspace(tmp)
    result = execute_tool('write_file', ws, Config(), dict(path='方案.docx', content=CONTENT))
    path = Path(tmp, '方案.docx')
    data = path.read_bytes()
    check('write_file 创建真正的 ZIP/OOXML', data.startswith(b'PK') and '结构校验通过' in result)
    info = inspect_docx(path)
    check('中文与 XML 特殊符号无损保留', '中文 & <标签> 重点 code' in info['text'])
    check('表格内容可读', '控制流程' in info['text'])
    check('read_file 支持 Word 正文提取', 'DOCX 结构校验通过' in ws.read_file('方案.docx'))
    with zipfile.ZipFile(path) as archive:
        xml = archive.read('word/document.xml')
        check('标题和表格具有原生 Word 结构', b'<w:tbl>' in xml and b'w:val="Title"' in xml)
        check('列表具有原生编号部件', 'word/numbering.xml' in archive.namelist())
    result = execute_tool('append_file', ws, Config(), dict(path='方案.docx', content='破坏文本'))
    check('拒绝直接追加 DOCX 且保留原文件', result.startswith('错误') and path.read_bytes() == data)
    for content in ('', '\x00bad'):
        result = execute_tool('write_file', ws, Config(), dict(path='方案.docx', content=content))
        check(f'非法正文不覆盖旧文件 {content!r}', result.startswith('错误') and path.read_bytes() == data)
    check('复现截图中的假 DOCX 并检测失败', invalid('# 设计方案\n此文档旨在对'.encode()))
    check('截断的 ZIP 被拒绝', invalid(data[:80]))
    broken = io.BytesIO()
    with zipfile.ZipFile(broken, 'w') as archive:
        archive.writestr('word/document.xml', '<broken>')
    check('损坏 XML/缺少包结构被拒绝', invalid(broken.getvalue()))
    wrong_type = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(wrong_type, 'w') as out:
        for name in source.namelist():
            part = source.read(name)
            if name == '[Content_Types].xml':
                part = part.replace(b'wordprocessingml.document.main+xml', b'plain-text')
            out.writestr(name, part)
    check('错误的 Word 内容类型被拒绝', invalid(wrong_type.getvalue()))
    result = execute_tool('write_file', ws, Config(), dict(path='方案.doc', content='text'))
    check('旧 DOC 后缀不能伪造纯文本', result.startswith('错误') and not Path(tmp, '方案.doc').exists())
    result = execute_tool('write_file', ws, Config(), dict(path='../outside.docx', content=CONTENT))
    check('Word 路径越界保护仍生效', result.startswith('错误') and '路径越界' in result)
    ws.write_file('notes.md', '# 普通文本')
    check('普通 Markdown 写入不受影响', ws.read_file('notes.md') == '# 普通文本')


class Model:
    def __init__(self, calls, protocol):
        self.calls, self.i = calls, 0
        self.cfg = Config(protocol=protocol, skills_enabled=False, web_enabled=False)

    def next(self):
        call = self.calls[min(self.i, len(self.calls)-1)]
        self.i += 1
        return call

    def chat_turn(self, messages, tools=None):
        call = self.next()
        if isinstance(call, str):
            yield dict(type='content', text=call)
        else:
            yield dict(type='tool_call', id=f'call_{self.i}', name=call[0], arguments=json.dumps(call[1]))

    def chat_stream(self, messages, **kwargs):
        call = self.next()
        yield 'ANSWER: ' + call if isinstance(call, str) else 'ACTION: ' + json.dumps(dict(name=call[0], arguments=call[1]))


for protocol in ('native', 'text'):
    with tempfile.TemporaryDirectory(prefix='pyagent_word_gate_') as tmp:
        ws = Workspace(tmp)
        def execute(calls):
            return list(Agent(Model(calls, protocol), ws).run('生成 Word 设计方案', confirm=lambda n, _: True))
        events = execute(['文档已经完成'])
        check(f'{protocol}: 没有文档不能宣称完成', events[-1]['type'] == 'error')
        events = execute([('write_file', dict(path='方案.docx', content='# 设计方案\n此文档旨在对')), '已完成'])
        check(f'{protocol}: 只有标题和断句不能交付方案', events[-1]['type'] == 'error'
              and any('疑似只有标题或未写完' in e.get('content', '') for e in events))
        events = execute([('write_file', dict(path='方案.docx', content=CONTENT)),
                          ('read_file', dict(path='方案.docx')), '已完成'])
        check(f'{protocol}: 真正的 Word 正文允许交付', events[-1]['type'] == 'answer')
        ws.write_file('break_doc.py', "from pathlib import Path\nPath('bad.docx').write_text('fake docx')\n")
        events = execute([('run_python', dict(path='break_doc.py')), '已完成'])
        check(f'{protocol}: 脚本生成的伪文档不能交付', events[-1]['type'] == 'error'
              and any('无法打开' in e.get('content', '') for e in events))
        ws.write_file('create_doc.py', f"from pathlib import Path\nPath('generated.docx').write_bytes({build_docx(CONTENT)!r})\n")
        events = execute([('run_python', dict(path='create_doc.py')), '已完成'])
        check(f'{protocol}: 支持脚本生成的合法 Word', events[-1]['type'] == 'answer')

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
raise SystemExit(1 if FAIL else 0)
