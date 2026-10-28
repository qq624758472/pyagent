"""V25 界面功能补全测试：置顶/筛选、仅提问模式、搜索、Tabs、工作台、菜单、导出"""
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
app = QApplication([])

# sessions 目录重定向到临时目录
import core.sessions as S
TMP = Path(tempfile.mkdtemp())
S.SESSIONS_DIR = TMP
from core.sessions import (save_session, delete_session, list_sessions,
                           set_session_pinned, filter_sessions, session_path)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (f"  | {detail}" if detail and not cond else ""))


def set_mtime(sid, seconds_ago):
    t = time.time() - seconds_ago
    os.utime(session_path(sid), (t, t))


print("== 1. 置顶 + 筛选数据层 ==")
save_session("s-new", [{"role": "user", "content": "新会话"}], "", "新会话")
save_session("s-old", [{"role": "user", "content": "老会话"}], "", "老会话")
save_session("s-pin", [{"role": "user", "content": "置顶会话"}], "", "置顶会话")
set_mtime("s-old", 8 * 86400)                 # 8 天前 → 归档
set_mtime("s-pin", 8 * 86400)                 # 也旧，但将被置顶
check("置顶写入", set_session_pinned("s-pin", True) is True)
check("置顶点查询", (load := (S.load_session("s-pin") or {})).get("pinned") is True)
check("取消置顶", set_session_pinned("s-pin", False) is True
      and "pinned" not in (S.load_session("s-pin") or {}))
set_session_pinned("s-pin", True)
ss = list_sessions()
check("置顶排最前", ss[0]["id"] == "s-pin", str([s["id"] for s in ss]))
check("筛选-全部=3", len(filter_sessions(ss, "all")) == 3)
check("筛选-置顶=1", len(filter_sessions(ss, "pinned")) == 1)
check("筛选-进行中(24h)=2", len(filter_sessions(ss, "active")) == 2)
check("筛选-归档(>7d)=1（置顶的也旧）", len(filter_sessions(ss, "archived")) == 1)
# 自动归档后置顶不丢
save_session("s-pin", [{"role": "user", "content": "置顶会话"}], "", "置顶会话")
check("save 后置顶保留", (S.load_session("s-pin") or {}).get("pinned") is True)

print("== 2. 仅提问模式（agent 层） ==")
from core.agent import Agent
from core.config import Config


class ScriptLLM:
    def __init__(self, script):
        self.script = script
        self.cfg = Config()
        self.last_reasoning = ""
        self.calls = 0

    def chat_stream(self, messages, model=None):
        self.calls += 1
        for ch in self.script[min(self.calls - 1, len(self.script) - 1)]:
            yield ch


qa_agent = Agent(ScriptLLM(['ACTION: {"name": "list_files", "arguments": {}}']),
                 qa_mode=True)
qa_agent._refresh_system_prompt()
sys_prompt = qa_agent.messages[0]["content"]
check("qa 系统提示无工具清单", "list_files" not in sys_prompt)
check("qa 系统提示声明仅提问", "仅提问" in sys_prompt)
events = list(qa_agent.run("罗列文件"))
kinds = [e["type"] for e in events]
check("qa 模式 ACTION 转直答（不执行工具）", kinds == ["delta", "delta", "answer"]
      or (kinds[-1] == "answer" and "action" not in kinds), str(kinds))
check("qa 模式没有 action 事件", "action" not in kinds)

ag_agent = Agent(ScriptLLM(['ANSWER: 好的']))
ag_agent._refresh_system_prompt()
check("Agent 模式系统提示含工具清单", "list_files" in ag_agent.messages[0]["content"])
check("set_qa_mode 切换", ag_agent.set_qa_mode(True) is None
      and ag_agent.qa_mode is True and "list_files" not in ag_agent.messages[0]["content"])

print("== 3. GUI：搜索 / 筛选 / Tabs / 工作台 ==")
import ui.main_window as mwmod


class FakeMB:
    Yes = QMessageBox.Yes
    No = QMessageBox.No
    answer = None
    info, warn = [], []

    @staticmethod
    def question(*a, **k):
        return FakeMB.answer

    @staticmethod
    def information(*a, **k):
        FakeMB.info.append(a[1] if len(a) > 1 else "")

    @staticmethod
    def warning(*a, **k):
        FakeMB.warn.append(a[1] if len(a) > 1 else "")


mwmod.QMessageBox = FakeMB


class FakeFD:
    @staticmethod
    def getSaveFileName(*a, **k):
        return (str(TMP / "export.md"), "")


mwmod.QFileDialog = FakeFD


class FakeMenu:
    made = []

    def __init__(self, parent=None):
        self.actions = []
        FakeMenu.made.append(self)

    def addAction(self, *a):
        self.actions.append(a)
        return a[0]

    def addSeparator(self):
        pass

    def exec_(self, *a):
        return None


mwmod.QMenu = FakeMenu

# 建工作区与若干会话
WS = Path(tempfile.mkdtemp())
(WS / "a.py").write_text("print(1)", encoding="utf-8")
(WS / "sub").mkdir()
save_session("s-alpha", [{"role": "user", "content": "关于部署的问题"}], "", "Alpha 部署")
save_session("s-beta", [{"role": "user", "content": "午饭吃什么"}], "", "Beta 闲聊")

from core.config import Config
from core.workspace import Workspace
mw = mwmod.MainWindow(Config())
mw.agent.set_workspace(Workspace(WS))
mw._refresh_tree()

chip_texts = {m: b.text() for m, b in mw._chips.items()}
check("chip 动态计数（全部 5）", chip_texts["all"] == "全部 5", str(chip_texts))
check("chip 置顶计数=1", chip_texts["pinned"] == "置顶 1")


def tree_session_ids():
    top = mw.tree.topLevelItem(0)
    out = []
    for i in range(top.childCount()):
        d = top.child(i).data(0, Qt.UserRole) if False else None
    for i in range(top.childCount()):
        n = top.child(i)
        d = n.data(0, 0x0100)
        if d and d[0] == "session":
            out.append(d[1])
    return out


check("默认显示全部会话", len(tree_session_ids()) == 5, str(tree_session_ids()))

# 筛选 chip
mw._set_session_filter("pinned")
check("置顶筛选只剩 1 个", tree_session_ids() == ["s-pin"], str(tree_session_ids()))
check("chip 互斥选中", mw._chips["pinned"].isChecked()
      and not mw._chips["all"].isChecked())
mw._set_session_filter("archived")
check("归档筛选命中 s-old", tree_session_ids() == ["s-old"], str(tree_session_ids()))
mw._set_session_filter("all")

# 搜索：标题匹配
mw.search.setText("alpha")
check("标题搜索过滤", tree_session_ids() == ["s-alpha"], str(tree_session_ids()))
mw.search.setText("不存在的词xyz")
check("无匹配显示空态", tree_session_ids() == [])
mw.search.setText("")
check("清空恢复全部", len(tree_session_ids()) == 5)

# 搜索：内容 fallback（标题不含"部署"，消息含）
mw.search.setText("部署")
ids = tree_session_ids()
check("内容 fallback 命中", "s-alpha" in ids, str(ids))
mw.search.setText("")

# 回车打开第一个匹配
mw.search.setText("beta")
mw._on_search_enter()
check("回车打开匹配会话", mw._session_id == "s-beta", mw._session_id)
mw.search.setText("")

# Tabs + 工作台（注意：上面切换会话会把工作区换成该会话记录的路径，
# s-beta 无工作区记录 → 已卸载；这里按真实使用场景重新挂载再看工作台）
mw.agent.set_workspace(Workspace(WS))
mw._refresh_workbench()
mw._switch_side_tab(1)
check("工作台 Tab 切换", mw.side_stack.currentIndex() == 1
      and mw.tab_ws.isChecked() and not mw.tab_chat.isChecked())
check("工作台树有工作区节点", mw.ws_tree.topLevelItemCount() == 1)
root = mw.ws_tree.topLevelItem(0)
check("工作台列出 2 个条目", root.childCount() == 2, str(root.childCount()))
mw._switch_side_tab(0)
check("切回对话 Tab", mw.side_stack.currentIndex() == 0
      and mw.tab_chat.isChecked())

# _open_path（mock os.startfile）
opened = []
real_startfile = getattr(os, "startfile", None)
os.startfile = lambda p: opened.append(p)
mw._open_path(WS / "a.py")
check("双击打开文件走系统默认方式", opened == [str(WS / "a.py")], str(opened))
mw._open_path(None)
check("未挂载打开有提示", len(FakeMB.info) >= 1)
mw._open_path(WS / "nope.txt")
check("不存在路径有警告", len(FakeMB.warn) >= 1)
if real_startfile:
    os.startfile = real_startfile

# ✦ 菜单：导出 Markdown
mw.agent.messages = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "导出测试对话"},
    {"role": "user", "content": "OBSERVATION: 过程消息不应导出"},
    {"role": "assistant", "content": "这是助手的回答"},
]
mw.b_star.click()
star_menu = FakeMenu.made[-1]
check("✦ 菜单 3 个动作", len(star_menu.actions) == 3, str(len(star_menu.actions)))
star_menu.actions[2][1]()                    # 导出
out = TMP / "export.md"
check("导出文件生成", out.exists())
if out.exists():
    txt = out.read_text(encoding="utf-8")
    check("导出含用户与助手消息", "导出测试对话" in txt and "助手的回答" in txt)
    check("导出不含 OBSERVATION 过程消息", "OBSERVATION" not in txt)
star_menu.actions[1][1]()                    # 清空当前对话
check("清空对话后消息为空", len(mw.agent.messages) == 0)
mw.agent.messages = [{"role": "user", "content": "导出测试对话"},
                     {"role": "assistant", "content": "助手的回答"}]

# ＋ 菜单
mw.b_plus.click()
plus_menu = FakeMenu.made[-1]
check("＋ 菜单 3 个动作", len(plus_menu.actions) == 3)

# 模式下拉
mw.agent_combo.setCurrentText("仅提问")
check("仅提问模式生效", mw.agent.qa_mode is True)
check("qa 系统提示已重建", "list_files" not in mw.agent.messages[0]["content"])
mw.agent_combo.setCurrentText("Agent")
check("切回 Agent 模式", mw.agent.qa_mode is False
      and "list_files" in mw.agent.messages[0]["content"])

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
