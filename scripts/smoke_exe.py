"""Exercise the built EXE from an isolated directory (including Python tasks).

    python scripts/smoke_exe.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def inside():
    import ctypes
    assert ctypes.windll.kernel32.GetConsoleWindow() == 0, "worker allocated a console"
    from PyQt5.QtWidgets import QApplication
    from core.config import Config, CONFIG_FILE
    from core.log import LOG_FILE, setup_logging
    from core.runtime import DATA_ROOT, RESOURCE_ROOT
    from core.sessions import save_session, load_session
    from core.skills import discover
    from core.tools import _run_python
    from core.workspace import Workspace
    from ui.main_window import MainWindow

    assert DATA_ROOT == Path(sys.executable).parent
    assert RESOURCE_ROOT != DATA_ROOT
    assert (RESOURCE_ROOT / "icons" / "app.png").is_file()
    cfg = Config()
    cfg.save()
    assert CONFIG_FILE.parent == DATA_ROOT and CONFIG_FILE.is_file()
    setup_logging()
    assert LOG_FILE.is_file() and LOG_FILE.parent.parent == DATA_ROOT
    skills = {s.name for s in discover(cfg)}
    assert {"code-review", "report-writer"} <= skills, skills

    workspace = DATA_ROOT / "workspace"
    workspace.mkdir(exist_ok=True)
    script = workspace / "hello.py"
    script.write_text("import csv, sqlite3, statistics, sys, ctypes\n"
                      "assert ctypes.windll.kernel32.GetConsoleWindow() == 0\n"
                      "print('BUNDLED_PYTHON_OK', statistics.mean([2,4]), sys.argv[1])\n"
                      "print('STDERR_OK', file=sys.stderr)\n",
                      encoding="utf-8")
    result = _run_python(Workspace(workspace), cfg, "hello.py", args="test-arg",
                         _confirmed=True)
    assert "BUNDLED_PYTHON_OK 3 test-arg" in result, result
    assert "STDERR_OK" in result, result
    from core.process_tools import _spawn, _tail
    proc, output = _spawn(Workspace(workspace), script, "test-arg")
    try:
        assert proc.wait(timeout=30) == 0
        assert "STDERR_OK" in _tail(output)
    finally:
        output.close()
    bad_script = workspace / "bad.py"
    bad_script.write_text("raise ValueError('EXPECTED_SCRIPT_ERROR')\n", encoding="utf-8")
    failure = _run_python(Workspace(workspace), cfg, "bad.py", _confirmed=True)
    assert "EXPECTED_SCRIPT_ERROR" in failure and "退出码 1" in failure, failure
    save_session("smoke", [{"role": "user", "content": "exe smoke test"}], "", "Smoke")
    assert load_session("smoke")["title"] == "Smoke"

    app = QApplication([])
    app.setStyle("Fusion")
    win = MainWindow(cfg)
    win.show()
    app.processEvents()
    assert not win.windowIcon().isNull()
    assert win.content_stack.currentIndex() == 1  # restores the isolated session
    win.new_chat()
    assert win.content_stack.currentIndex() == 0
    win.toggle_theme()
    assert cfg.theme == "light"
    win.hide()
    print(json.dumps({"result": "PASS", "skills": sorted(skills),
                      "checks": ["portable data", "resources", "logging", "skills",
                                 "no console", "bundled Python stdout/stderr",
                                 "background process output", "script errors",
                                 "sessions", "Qt GUI", "theme"]}))


def main():
    root = Path(__file__).resolve().parent.parent
    target = root / "build" / "exe smoke"
    target.mkdir(parents=True, exist_ok=True)
    exe = target / "PyAgent.exe"
    shutil.copy2(root / "dist" / "PyAgent.exe", exe)
    import pefile
    with pefile.PE(str(exe), fast_load=True) as pe:
        assert pe.OPTIONAL_HEADER.Subsystem == 2, "EXE must use Windows GUI subsystem"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    for key in list(env):
        if key.startswith("AGENT_"):
            del env[key]
    result = subprocess.run([str(exe), "--run-python", str(Path(__file__).resolve()), "--inside"],
                            cwd=str(target), env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=120,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)
    (root / "build" / "exe-smoke.json").write_text(result.stdout, encoding="utf-8")


if __name__ == "__main__":
    if "--inside" in sys.argv:
        inside()
    else:
        main()
