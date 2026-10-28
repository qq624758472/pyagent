"""V24 会话删除测试：delete_session + GUI 删除按钮全流程（offscreen + 桩对话框）"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, r"E:/ai_work/pyagent")

from PyQt5.QtWidgets import QApplication, QMessageBox
app = QApplication([])

# 重定向 sessions 目录到临时目录，避免动真实工程数据
import core.sessions as S
TMP = Path(tempfile.mkdtemp())
S.SESSIONS_DIR = TMP
from core.sessions import save_session, delete_session, list_sessions, session_path

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (f"  | {detail}" if detail and not cond else ""))


print("== 1. delete_session 基础 ==")
save_session("s-del-1", [{"role": "user", "content": "甲"},
                         {"role": "assistant", "content": "甲你好"}], "", "会话甲")
save_session("s-keep-2", [{"role": "user", "content": "乙"}], "", "会话乙")
check("删除存在的会话返回 True", delete_session("s-del-1") is True)
check("文件确实被删", not session_path("s-del-1").exists())
check("删除不存在的会话也返回 True（幂等）", delete_session("s-del-1") is True)
check("非法 id（路径逃逸）被拒绝", delete_session("../evil") is False)
check("空 id 被拒绝", delete_session("") is False)
check("其余会话不受影响", session_path("s-keep-2").exists())
check("list_sessions 只剩 1 个", len(list_sessions()) == 1)
delete_session("s-keep-2")               # 清场，后续 GUI 场景只放两个会话
check("清场后无会话", len(list_sessions()) == 0)

print("== 2. GUI：启动恢复 + 删除按钮渲染 ==")
import ui.main_window as mwmod


class FakeMB:
    Yes = QMessageBox.Yes
    No = QMessageBox.No
    answer = None
    calls = []

    @staticmethod
    def question(*a, **k):
        FakeMB.calls.append(a[1] if len(a) > 1 else "")
        return FakeMB.answer

    @staticmethod
    def information(*a, **k):
        return None

    @staticmethod
    def warning(*a, **k):
        return None


mwmod.QMessageBox = FakeMB

# 先存旧会话，再存当前会话 → 最近的是当前会话
save_session("s-other", [{"role": "user", "content": "另一个会话的问题"}], "", "别的会话")
save_session("s-current", [{"role": "user", "content": "当前会话的问题"},
                           {"role": "assistant", "content": "当前会话的回答"}], "", "当前会话")

from core.config import Config
mw = mwmod.MainWindow(Config())

check("启动恢复了最近会话", mw._session_id == "s-current", mw._session_id)
check("恢复后内存有对话", len(mw.agent.messages) > 0)


def find_btn(sid):
    """在树里找某会话的删除按钮"""
    top = mw.tree.topLevelItem(0)
    for i in range(top.childCount()):
        node = top.child(i)
        if node.data(0, 0x0100) == ("session", sid):      # Qt.UserRole
            return mw.tree.itemWidget(node, 1)
    return None


btn_cur, btn_other = find_btn("s-current"), find_btn("s-other")
check("当前会话有删除按钮", btn_cur is not None)
check("其他会话有删除按钮", btn_other is not None)

print("== 3. 删除其他会话：当前对话不受影响 ==")
FakeMB.answer = QMessageBox.Yes
btn_other.click()
check("确认框已弹出", len(FakeMB.calls) == 1)
check("其他会话文件已删", not session_path("s-other").exists())
check("当前会话 id 不变", mw._session_id == "s-current")
check("当前对话内存保留", len(mw.agent.messages) > 0)
check("树刷新后只剩 1 个会话按钮", find_btn("s-current") is not None and find_btn("s-other") is None)

print("== 4. 删除当前会话：对话 + 历史记录一并清除 ==")
FakeMB.answer = QMessageBox.Yes
find_btn("s-current").click()
check("当前会话文件已删", not session_path("s-current").exists())
check("_session_id 已置空（不会把内容又写回磁盘）", mw._session_id is None)
check("内存对话已清空", len(mw.agent.messages) == 0, str(len(mw.agent.messages)))
check("sessions 目录已空", len(list_sessions()) == 0)
check("树里无会话按钮", find_btn("s-current") is None)

print("== 5. 用户点「否」= 不删除 ==")
save_session("s-cancel", [{"role": "user", "content": "别删我"}], "", "取消测试")
mw._refresh_tree()
btn = find_btn("s-cancel")
FakeMB.answer = QMessageBox.No
btn.click()
check("点了否，文件还在", session_path("s-cancel").exists())
check("点了否，会话数不变", len(list_sessions()) == 1)

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
