"""离屏验证 Markdown 真实文档结构、流式回答、阅读滚动与状态显示。"""
import os
import sys
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QTextTable, QTextFormat
from core.config import Config
import core.sessions as sessions
from ui.main_window import MainWindow, MarkdownView, ToolCollapseCard
from ui.theme import THEMES

PASS, FAIL = [], []


def check(label, ok):
    (PASS if ok else FAIL).append(label)
    print(('PASS ' if ok else 'FAIL ') + label)


app = QApplication.instance() or QApplication([])


def settle():
    for _ in range(8):
        app.processEvents()


SOURCE = ('## 测试结果\n\n| 项目 | 状态 |\n| --- | --- |\n| 单测 | 通过 |\n\n'
          '1. 检查代码\n   - 检查边界\n   - 检查异常\n\n- [x] 已验证\n\n'
          '```python\nprint("**literal** & <safe>")\n```\n\n'
          '> 保留验证证据\n\n[文档](https://example.com)\n\n'
          '<script>alert(1)</script>\n')
with tempfile.TemporaryDirectory(prefix='pyagent_md_') as tmp:
    sessions.SESSIONS_DIR = Path(tmp)
    win = MainWindow(Config())
    win.resize(1200, 820)
    win.show()
    settle()
    win._start_assistant()
    view = win._stream_view
    for chunk in (SOURCE[:18], SOURCE[18:95], SOURCE[95:]):
        win.on_event(dict(type='delta', content=chunk))
    view.flush()
    settle()
    check('任务完成前已渲染 Markdown', '## 测试结果' not in view.toPlainText() and '测试结果' in view.toPlainText())
    tables = [f for f in view.document().rootFrame().childFrames() if isinstance(f, QTextTable)]
    check('原生表格包含正确行列', len(tables) == 1 and tables[0].rows() == 2 and tables[0].columns() == 2)
    blocks = []
    block = view.document().begin()
    while block.isValid():
        blocks.append(block)
        block = block.next()
    check('嵌套列表保留层级', any(b.textList() and b.textList().format().indent() > 1 for b in blocks))
    code = [b for b in blocks if b.blockFormat().hasProperty(QTextFormat.BlockCodeFence) and 'print' in b.text()]
    check('代码块保留原文与等宽字体', code and '**literal** & <safe>' in code[0].text()
          and code[0].begin().fragment().charFormat().fontFamily() == 'Consolas')
    check('HTML 不作为脚本/标签执行', '<script>' not in view.document().toHtml())
    win.on_event(dict(type='answer', content=SOURCE))
    settle()
    check('最终完整答案不重复追加', view.source == SOURCE and view.toPlainText().count('测试结果') == 1)
    check('完成时清除延迟渲染任务', not view.timer.isActive())
    view.show_source(True)
    check('原文视图完整保留 Markdown', view.toPlainText() == SOURCE)
    view.show_source(False)
    view.copy_text()
    check('复制全文为可读正文', QApplication.clipboard().text() == view.toPlainText())
    check('切换视图不损坏源文', view.source == SOURCE)
    win.config.theme = 'light'
    win._apply_theme()
    check('历史消息切换亮色主题', view.colors == THEMES['light'] and view.source == SOURCE)
    win.ui_scale = 1.3
    win._apply_theme()
    check('历史 Markdown 随界面缩放', view.document().defaultFont().pixelSize() == round(14 * 1.3))
    win._clear_chat()
    win._start_assistant()
    long_view = win._stream_view
    long_view.set_source('\n\n'.join(f'Paragraph {i}' for i in range(160)))
    settle()
    bar = win.scroll.verticalScrollBar()
    bar.setValue(0)
    long_view.append_source('\n\nNew paragraph')
    long_view.flush()
    settle()
    check('上翻阅读时不强制滚到底部', bar.maximum() > 1000 and bar.value() == 0)
    check('长回答使用外层滚动而不截断', long_view.height() > 2400)
    win._clear_chat()
    win._replay([dict(role='assistant', content=SOURCE)])
    settle()
    replay = win.findChildren(MarkdownView)[-1]
    check('历史回答也使用 Markdown 排版', replay.source == SOURCE and '## 测试结果' not in replay.toPlainText())
    win._append_user('请检查代码并运行测试，然后汇报执行结果。')
    settle()
    check('用户消息避免挤成窄列', win.findChildren(QLabel, 'userBubble')[-1].width() > 200)
    card = ToolCollapseCard(THEMES['light'])
    card.add_action('write_file', dict(path='app.py', content='x' * 3000))
    card.add_observation('first\n' + 'x' * 7000 + '\nLAST_ERROR')
    from ui.main_window import AutoHeightLabel
    details = card.findChildren(AutoHeightLabel)
    check('写入摘要不显示整段正文 JSON', '3000 字符' in details[0].toPlainText())
    check('长执行结果保留末尾错误', 'LAST_ERROR' in details[-1].toPlainText())
    card.finish('error')
    check('执行失败不显示完成勾号', '执行失败' in card.head.text() and '✔' not in card.head.text())
    card.finish('cancelled')
    check('用户停止显示停止状态', '已停止' in card.head.text())
    card.hide()
    win.hide()

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
raise SystemExit(1 if FAIL else 0)
