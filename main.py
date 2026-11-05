#!/usr/bin/env python3
"""PyAgent 入口：默认 GUI，-c 进入控制台；核心逻辑与 UI 完全解耦"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    from core.runtime import initialize_stdio
    initialize_stdio()
    from multiprocessing import freeze_support
    freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--run-python":
        from core.runtime import run_python_worker
        try:
            run_python_worker()
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(1)
        raise SystemExit(0)

from core.config import Config
from core.log import setup_logging, LOG_FILE, config_snapshot
from core.workspace import Workspace, WorkspaceError


def gui_available() -> bool:
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import PyQt5.QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="知行 - 可挂载工作区的智能助手（GUI/控制台双模式）")
    parser.add_argument("-c", "--console", action="store_true", help="使用控制台模式")
    parser.add_argument("--gui", action="store_true", help="强制使用图形界面")
    parser.add_argument("-w", "--workspace", metavar="PATH", help="启动时挂载的工作区目录")
    parser.add_argument("--task", metavar="TEXT", help="单次任务（执行完退出，隐含控制台模式）")
    args = parser.parse_args()

    config = Config.load()
    setup_logging(config.log_level)
    import logging
    use_gui = (args.gui or not args.console) and not args.task and gui_available()
    logging.getLogger("main").info(
        "PyAgent 启动 mode=%s argv=%s | %s | 日志文件: %s",
        "GUI" if use_gui else "console",
        sys.argv[1:], config_snapshot(config), LOG_FILE)

    workspace = None
    if args.workspace:
        try:
            workspace = Workspace(args.workspace)
        except WorkspaceError as e:
            print(f"[错误] {e}", file=sys.stderr)
            sys.exit(1)

    if args.gui and not gui_available():
        print("[提示] 当前环境无法使用 GUI（缺少 PyQt5 或显示服务），回退控制台模式")

    if use_gui:
        from ui.main_window import run_gui
        run_gui(config, workspace)
    else:
        import cli
        cli.run(config, workspace, task=args.task)


if __name__ == "__main__":
    main()
