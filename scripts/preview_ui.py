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
        win.resize(920, 680)
        capture("studio-compact.png")
        # Do not close through MainWindow.closeEvent (which persists sessions).
        win.hide()
    print(f"Previews: {output}")


if __name__ == "__main__":
    main()
