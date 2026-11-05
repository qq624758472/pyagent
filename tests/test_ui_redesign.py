"""统一富文本显示、可视化技能编辑、历史/文件导航以及单一对话滚动。"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from PyQt5.QtWidgets import QApplication,QLabel,QToolButton
from PyQt5.QtCore import Qt,QPoint,QPointF
from PyQt5.QtGui import QTextTable,QWheelEvent
from core.config import Config
from core.workspace import Workspace
from core.task_plan import TaskPlan
import core.sessions as sessions
from ui.main_window import MainWindow,MarkdownView,ToolCollapseCard
from ui.markdown_document import MarkdownPanel,MarkdownEditor
from ui.skills_dialog import SkillsDialog

app=QApplication([])
PASS,FAIL=[],[]
def check(name,ok):
    (PASS if ok else FAIL).append(name)
    print(('PASS ' if ok else 'FAIL ')+name)
def settle():
    for _ in range(15): app.processEvents()

SOURCE='## 验证结果\n\n**完成**\n\n| 项目 | 状态 |\n| --- | --- |\n| 文档 | 通过 |\n\n- 第一项\n- 第二项'
with tempfile.TemporaryDirectory(prefix='zhixing_ui_') as tmp:
    root=Path(tmp);sessions.SESSIONS_DIR=root/'sessions'
    ws=Workspace(root)
    (root/'example.md').write_text(SOURCE,encoding='utf-8')
    (root/'nested').mkdir();(root/'nested'/'readme.md').write_text(SOURCE,encoding='utf-8')
    sessions.save_session('history-1',[dict(role='user',content='历史标题')],str(root),'历史标题')
    win=MainWindow(Config());win.show();settle()
    check('品牌及窗口标题已经更名','知行' in win.windowTitle())
    check('历史导航与工作区分开显示',win.tree.topLevelItemCount()==1 and win.tab_ws.text()=='工作区')
    node=win.tree.topLevelItem(0).child(0)
    check('会话行包含更新时间和工作区元信息',bool(node.data(0,Qt.UserRole+1)))
    check('会话行高度足够显示标题和元信息',win.tree.visualItemRect(node).height()>=55 and win.tree.columnWidth(1)<=40)
    win._switch_side_tab(1)
    directory=next(win.ws_tree.topLevelItem(0).child(i) for i in range(win.ws_tree.topLevelItem(0).childCount()) if win.ws_tree.topLevelItem(0).child(i).text(0)=='nested')
    directory.setExpanded(True);settle()
    check('工作区支持展开子目录',directory.childCount()==1 and directory.child(0).text(0)=='readme.md')
    with patch('ui.main_window.QDialog.exec_',return_value=0): win._open_path(root/'example.md')
    check('工作区Markdown文件使用内置排版阅读器',any(p.source==SOURCE for p in win.findChildren(MarkdownPanel)))
    win._switch_side_tab(0)
    win.agent.messages.append(dict(role='assistant',content='尚未归档的内容'))
    win._open_session_by_id('history-1')
    check('点击当前会话不会丢失未归档内容',win.agent.messages[-1]['content']=='尚未归档的内容')
    win._append_user(SOURCE);settle()
    bubble=win.findChildren(QLabel,'userBubble')[-1]
    check('用户消息中的Markdown也按富文本呈现','<table' in bubble.text() and '## 验证结果' not in bubble.text())
    win._start_assistant();win._stream(SOURCE);win._finish_assistant();settle()
    check('界面不再提供默认源码展示入口',not any(b.text()=='原文' for b in win.findChildren(QToolButton)))
    win._append_error('**验证失败**：请检查参数')
    check('富文本错误信息保持错误颜色',win._c['error'].lower() in win.findChildren(QLabel,'errLine')[-1].text().lower())
    card=ToolCollapseCard(win._c,win)
    win._add_widget(card)
    card.add_action('read_file',dict(path='example.md'))
    card.add_observation(SOURCE+'\n\n'+'\n\n'.join(f'段落 {i}：检查输出内容。' for i in range(80)))
    card.head.setChecked(True);card._on_toggle(True)
    toggle=card.findChild(QToolButton,'resultToggle');toggle.setChecked(True);settle()
    output=card.findChild(MarkdownView,'toolObs')
    check('执行结果中的标题表格正常渲染','## 验证结果' not in output.toPlainText() and any(isinstance(f,QTextTable) for f in output.document().rootFrame().childFrames()))
    check('长执行结果不截断成260像素内部窗口',output.height()>260 and output.verticalScrollBarPolicy()==Qt.ScrollBarAlwaysOff)
    bar=win.scroll.verticalScrollBar();bar.setValue(bar.maximum());before=bar.value()
    event=QWheelEvent(QPointF(20,20),QPointF(output.mapToGlobal(QPoint(20,20))),QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
    QApplication.sendEvent(output.viewport(),event);settle()
    check('结果区域滚轮操作滚动外层对话',before>0 and bar.value()<before)
    bar.setValue(0);win._relayout_streams();settle()
    check('重新布局不会把正在阅读的历史强制拉到底部',bar.value()==0)
    plan=TaskPlan.start(ws,'生成方案 report.docx',True)
    panel=MarkdownPanel(win._c);panel.set_markdown(plan.markdown())
    check('任务计划按Markdown表格呈现',any(isinstance(f,QTextTable) for f in panel.document().rootFrame().childFrames()) and '# 任务计划' not in panel.toPlainText())
    editor=SkillsDialog(win,edit=True)
    editor.body.setPlainText('## 技能步骤\n\n**检查来源**\n\n- 阅读相关代码和用户要求，保存有依据的分析笔记。\n- 验证实际结果，发现问题后修复并再次检查。\n')
    check('技能编写器是可视化富文本而非Markdown源码','##' not in editor.body.toPlainText() and editor.body.document().begin().blockFormat().headingLevel()==2)
    editor.name.setText('visual-skill');editor.description.setText('可视化编辑验证')
    _,path=editor.write_skill()
    saved=path.read_text(encoding='utf-8')
    check('可视化编辑保存后仍保留Markdown结构','## 技能步骤' in saved and '**检查来源**' in saved)
    if '## 技能步骤' not in saved or '**检查来源**' not in saved: print(repr(saved))
    editor.body.setPlainText('请核实重点内容并保存实际验证结果。')
    cursor=editor.body.document().find('重点')
    editor.body.setTextCursor(cursor);editor.body.format_bold()
    check('可视化加粗操作可以正确保存','**重点**' in editor.body.markdown())
    editor.hide();panel.hide();win.hide()

print(f'结果: {len(PASS)} 通过, {len(FAIL)} 失败')
sys.exit(bool(FAIL))
