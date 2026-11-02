"""GUI 危险操作确认状态机测试：用模拟弹窗避免无显示器 Qt 模态框崩溃。"""
import os
import sys
import threading
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtWidgets import QApplication, QWidget
import ui.main_window as gui

app = QApplication.instance() or QApplication([])
widget = QWidget()
passed, failed = [], []


def check(name, ok):
    (passed if ok else failed).append(name)
    print(("PASS " if ok else "FAIL ") + name)


class Timer:
    def __init__(self, parent):
        self.timeout = self
    def setSingleShot(self, value): pass
    def connect(self, callback): pass
    def start(self, millis): pass
    def stop(self): pass


class Dialog:
    Question, Yes, No = 1, 2, 4
    on_exec = None
    def __init__(self, *args):
        self.answer = self.Yes
    def setDefaultButton(self, button): pass
    def exec_(self):
        if self.on_exec:
            self.on_exec()
        return self.answer
    def reject(self): pass


gui.QMessageBox = Dialog
gui.QTimer = Timer


def box(expired=False):
    return {"ok": False, "ev": threading.Event(),
            "lock": threading.Lock(), "expired": expired}


expired = box(expired=True)
gui.MainWindow._on_confirm_dialog(widget, expired, "run_python", "test")
check("pre-expired confirmation ignored", not expired["ok"] and expired["ev"].is_set())

fresh = box()
Dialog.on_exec = None
gui.MainWindow._on_confirm_dialog(widget, fresh, "run_python", "test")
check("fresh affirmative confirmation accepted", fresh["ok"] and fresh["ev"].is_set())

late = box()
Dialog.on_exec = staticmethod(lambda: late.update(expired=True))
gui.MainWindow._on_confirm_dialog(widget, late, "run_python", "test")
check("late affirmative confirmation ignored", not late["ok"] and late["ev"].is_set())

print("结果: %d 通过, %d 失败" % (len(passed), len(failed)))
raise SystemExit(bool(failed))
