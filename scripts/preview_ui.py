"""Render Studio previews without opening a window or changing user sessions.

    python scripts/preview_ui.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import QApplication

from core.config import Config
import core.sessions as sessions
from ui.main_window import MainWindow


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    # Windows' offscreen plugin does not enumerate installed fonts.
    if sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("segoeui.ttf", "seguisb.ttf", "msyh.ttc", "msyhbd.ttc"):
            QFontDatabase.addApplicationFont(str(fonts / name))
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as directory:
        sessions.SESSIONS_DIR = Path(directory)
        config = Config()
        config.api_key = "preview-only"
        win = MainWindow(config)
        win.show()

        def capture(name):
            for _ in range(8):
                app.processEvents()
            win.grab().save(str(output / name))

        capture("studio-dark.png")
        config.theme = "light"
        win._apply_theme()
        capture("studio-light.png")
        win._append_user("请帮我梳理项目结构，并给出优化建议。")
        win._start_assistant("项目分析")
        win._stream("## 从项目结构开始\n\n我可以先查看工作区的目录与入口文件，再整理核心模块之间的关系。\n\n"
                    "- **结构概览**：梳理目录、依赖与运行方式\n"
                    "- **改进建议**：关注可维护性与使用体验\n\n"
                    "请通过左下角的「连接工作区」选择项目文件夹。")
        win._finish_assistant()
        capture("studio-conversation.png")
        win._clear_chat()
        win._append_user("检查项目，执行测试并给出 Markdown 结果。")
        win._ensure_tool_card().add_action("run_python", {"path": "test_app.py"})
        win._tool_card.add_observation("Ran 4 tests in 0.12s\nOK")
        win._tool_card.finish()
        win._start_assistant("检查结果")
        win._stream("## 项目检查结果\n\n已完成代码检查与测试，当前结果如下。\n\n"
                    "| 检查项 | 状态 | 说明 |\n| --- | --- | --- |\n"
                    "| 程序运行 | 通过 | 返回码为 0 |\n| 自动测试 | 通过 | 4 项测试通过 |\n\n"
                    "### 使用方法\n\n```python\npython app.py add \"整理项目\"\npython app.py list\n```\n\n"
                    "- **验证结果**：最后一次修改后已重新测试\n"
                    "- **下一步**：可继续补充更多异常输入的测试\n")
        win._finish_assistant()
        capture("studio-markdown-light.png")
        config.theme = "dark"
        win._apply_theme()
        capture("studio-markdown-dark.png")
        win._clear_chat()
        win.resize(920, 680)
        capture("studio-compact.png")
        from ui.skills_dialog import SkillsDialog
        editor=SkillsDialog(win,edit=True)
        editor.show()
        for _ in range(8): app.processEvents()
        editor.grab().save(str(output/'studio-skills.png'))
        editor.hide()
        from core.workspace import Workspace
        from PyQt5.QtWidgets import QToolButton
        project=Path(directory)/'示例项目'
        project.mkdir()
        (project/'README.md').write_text('# 项目说明\n\n用于界面预览的临时项目。',encoding='utf-8')
        (project/'src').mkdir()
        sessions.save_session('nav-demo-1',[{'role':'user','content':'梳理模块并补充设计文档'}],str(project),'梳理模块并补充设计文档')
        sessions.save_session('nav-demo-2',[{'role':'user','content':'检查接口和测试结果'}],str(project),'检查接口和测试结果')
        win.agent.set_workspace(Workspace(project));win._session_id='nav-demo-1'
        win.config.theme='light';win.resize(1200,820);win._apply_theme()
        win._refresh_tree();win._refresh_workbench();win._refresh_status();win._clear_chat()
        win._append_user('## 检查项目\n\n请先阅读说明，再检查接口和测试结果。')
        card=win._ensure_tool_card();card.add_action('read_file',{'path':'README.md'})
        card.add_observation('## 项目检查\n\n| 检查项 | 状态 |\n| --- | --- |\n| 项目说明 | 已读取 |\n| 接口测试 | 进行中 |\n\n已收集模块说明，正在核对接口。')
        card.head.setChecked(True);card._on_toggle(True)
        card.findChild(QToolButton,'resultToggle').setChecked(True)
        win.plan_progress.setText('任务进度 2/6 · 检查接口与测试结果')
        capture('studio-execution.png')
        # Do not close through MainWindow.closeEvent (which persists sessions).
        win.hide()
    print(f"Previews: {output}")


if __name__ == "__main__":
    main()
