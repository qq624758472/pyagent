"""PyAgent Studio 桌面界面，与控制台共用同一套 Agent 核心。"""
import html
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, QEvent, QTimer, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QFileDialog,
                             QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLayout,
                             QHeaderView, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPlainTextEdit, QPushButton,
                             QScrollArea, QMenu, QShortcut, QStackedWidget,
                             QTextEdit, QToolButton, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget, QSizePolicy)

from core.agent import Agent
from core.config import PROJECT_ROOT
from core.runtime import RESOURCE_ROOT
from core.llm_client import LLMClient
from core.log import get_logger, config_snapshot
from core.sessions import (delete_session, filter_sessions, list_sessions,
                           load_session, new_session_id, save_session,
                           set_session_pinned)
from core.workspace import Workspace, WorkspaceError
from ui.theme import THEMES, ZOOM_MAX, ZOOM_MIN, ZOOM_STEP, build_qss
from ui.workers import AgentWorker, FetchModelsWorker
from ui.icons import line_icon

log = get_logger("ui.main_window")


def _msg_pair(m):
    """兼容旧版 [role, content] 与新版完整 OpenAI 消息对象。"""
    if isinstance(m, dict):
        return m.get("role"), (m.get("content") or "")
    if isinstance(m, (list, tuple)) and len(m) >= 2:
        return m[0], (m[1] or "")
    return None, ""


# ============================================================
# 基础控件
# ============================================================
class SelectableLabel(QTextEdit):
    """只读、无边框、可选中复制、右键菜单带「复制」的文本块"""

    def __init__(self, text="", parent=None, objname=""):
        super().__init__(parent)
        self.setPlainText(text)
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if objname:
            self.setObjectName(objname)

    def _copy(self):
        self.copy()

    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Copy):
            self._copy()
            return
        super().keyPressEvent(e)

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        menu.addAction("复制", self._copy)
        menu.exec_(e.globalPos())


class AutoHeightLabel(SelectableLabel):
    """高度随文档内容自适应的只读文本块

    踩坑记录（V31.2）：QTextEdit 的 document().size().height() 在控件刚
    创建、尚未进入布局时返回 0 —— 此前按它算固定高度，得到的永远是
    max(16, 0+6)=16px，文字被裁成一条线，展开「执行过程」卡看到的
    全是空行（用户报：点开看不到文字、卡片叠在一起）。
    正确做法：监听 documentSizeChanged，等布局给出真实宽度后再同步高度；
    超长内容封顶并保留内部滚动条。"""

    def __init__(self, text="", parent=None, objname="", max_height=0):
        super().__init__(text, parent, objname=objname)
        self._max_h = int(max_height or 0)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.document().documentLayout().documentSizeChanged.connect(
            self._sync_height)
        QTimer.singleShot(0, self._sync_height)      # 布局完成后补一次

    def _sync_height(self, *_a):
        try:
            h = int(self.document().size().height())
        except Exception:
            return
        if h <= 0:
            return                        # 还没布局出宽度，等下一次信号
        h += 10                            # 边框/内边距
        if self._max_h:
            h = min(h, self._max_h)
        if abs(self.height() - h) > 1:     # 避免反复触发布局抖动
            self.setFixedHeight(max(16, h))


class InputBox(QPlainTextEdit):
    """输入框：Enter 发送、Shift+Enter 换行"""
    submitted = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)      # Shift+Enter → 换行
                return
            self.submitted.emit()
            return
        super().keyPressEvent(event)


# ============================================================
# 设置对话框
# ============================================================
class SettingsDialog(QDialog):
    """设置：API Key / Base URL / 模型（主 + kb/web 角色）/ 开关 / 协议"""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._models_worker = None
        self.setWindowTitle("设置")
        self.setMinimumWidth(620)
        self._status = QLabel("")
        self._status.setWordWrap(True)

        form = QFormLayout()
        self.e_key = QLineEdit(config.api_key)
        self.e_key.setPlaceholderText("sk-…（智谱开放平台获取）")
        self.e_key.setEchoMode(QLineEdit.Password)
        self.e_url = QLineEdit(config.base_url)
        self.e_url.setPlaceholderText("https://open.bigmodel.cn/api/paas/v4/")

        # 主模型：可编辑下拉 + 「获取模型」
        self.c_model = QComboBox()
        self.c_model.setEditable(True)
        self.c_model.addItem(config.model)
        self.c_model.setCurrentText(config.model)
        self.b_fetch = QPushButton("获取模型")
        self.b_fetch.clicked.connect(self.fetch_models)
        model_row = QHBoxLayout()
        model_row.addWidget(self.c_model, 1)
        model_row.addWidget(self.b_fetch)

        form.addRow("API Key:", self.e_key)
        form.addRow("Base URL:", self.e_url)
        form.addRow("对话模型:", model_row)

        # 角色独立配置：kb（知识库检索）/ web（联网采集）
        self.profile_widgets = {}
        head = QLabel("模型角色（独立配置，留空使用主配置）:")
        head.setStyleSheet("font-size:12px; font-weight:bold;")
        form.addRow(head)
        for key, label in (("kb", "知识库检索:"), ("web", "联网搜索:")):
            combo = QComboBox()
            combo.setEditable(True)
            prof = config.profile(key)
            combo.addItem("")
            combo.setCurrentText(prof.model or "")
            combo.setPlaceholderText("留空 = 主配置")
            self.profile_widgets[key] = combo
            form.addRow(label, combo)

        # 工具协议
        self.c_protocol = QComboBox()
        self.c_protocol.addItems(["auto", "native", "text"])
        self.c_protocol.setCurrentText(getattr(config, "protocol", "auto"))
        self.c_protocol.setToolTip(
            "工具调用协议：自动=优先原生 function calling（gpt-oss 等思考模型"
            "强烈推荐，端点不支持时自动回退文本协议）；原生=强制 function "
            "calling；文本=传统 ACTION 协议")
        form.addRow("工具协议:", self.c_protocol)

        # 各种开关
        self.chk_shell = QCheckBox("允许 Agent 执行 shell 命令（有安全风险，谨慎开启）")
        self.chk_shell.setChecked(bool(config.allow_shell))
        self.chk_python = QCheckBox("允许 Agent 直接执行本地 Python 脚本（run_python 工具）")
        self.chk_python.setChecked(bool(getattr(config, "python_enabled", True)))
        self.chk_web = QCheckBox("启用联网采集（关闭后 Agent 不使用 web_collect 工具）")
        self.chk_web.setChecked(bool(getattr(config, "web_enabled", True)))
        self.chk_light = QCheckBox("亮色主题（取消勾选为暗色）")
        self.chk_light.setChecked(config.theme == "light")
        self.chk_debug = QCheckBox(
            "调试模式（每轮显示模型原始返回与解析过程，排障用；"
            "关闭时调试卡仍存在，只是默认收起）")
        self.chk_debug.setChecked(bool(getattr(config, "debug_mode", False)))
        self.chk_confirm = QCheckBox(
            "高危操作需用户确认（递归删除目录 / shell 命令 / 本地 Python 执行；"
            "Python 同一任务内只问一次；关闭后自动执行）")
        self.chk_confirm.setChecked(bool(getattr(config, "confirm_dangerous", True)))

        # 技能开关（带已发现技能数提示）
        self.chk_skills = QCheckBox()
        try:
            from core.skills import discover
            found = discover(config, None)
            if found:
                names = "、".join(s.name for s in found[:3])
                more = f" 等 {len(found)} 个" if len(found) > 3 else ""
                self.chk_skills.setText(f"启用技能（Skill）· 已发现 {names}{more}")
            else:
                self.chk_skills.setText(
                    "启用技能（Skill）· 未发现技能（放到 <工程>/skills/<名字>/SKILL.md）")
        except Exception:
            self.chk_skills.setText("启用技能（Skill）")
        self.chk_skills.setChecked(bool(getattr(config, "skills_enabled", True)))
        self.chk_skills.setToolTip(
            "技能是可复用的操作规范包（目录内含 SKILL.md）。\n"
            "Agent 遇到匹配任务时会先加载技能规范再执行。\n"
            "技能目录：<工程>/skills/、<工作区>/.pyagent/skills/、~/.pyagent/skills/")

        for w in (self.chk_shell, self.chk_python, self.chk_web, self.chk_light,
                  self.chk_debug, self.chk_confirm, self.chk_skills):
            form.addRow(w)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)
        heading = QLabel("配置你的工作助手", self)
        heading.setObjectName("settingsTitle")
        layout.addWidget(heading)
        form.setVerticalSpacing(12)
        layout.addLayout(form)
        layout.addWidget(self._status)
        layout.addWidget(buttons)

    # ---------- 模型列表 ----------
    def fetch_models(self):
        key = self.e_key.text().strip()
        url = self.e_url.text().strip()
        if not key:
            self._show_status("请先填写 API Key 再获取模型。", error=True)
            return
        if not url:
            self._show_status("请先填写 Base URL 再获取模型。", error=True)
            return
        self.b_fetch.setEnabled(False)
        self.b_fetch.setText("获取中…")
        self._show_status(f"正在从 {url} 拉取模型列表…")
        self._models_worker = FetchModelsWorker(key, url, self)
        self._models_worker.ok.connect(self._on_models)
        self._models_worker.failed.connect(self._on_fetch_failed)
        self._models_worker.finished.connect(self._fetch_cleanup)
        self._models_worker.start()

    def _on_models(self, ids):
        cur_main = self.c_model.currentText().strip()
        cur_profiles = {k: w.currentText().strip()
                        for k, w in self.profile_widgets.items()}
        self.c_model.clear()
        self.c_model.addItems(ids or [])
        keep = cur_main or self.config.model
        self.c_model.setCurrentText(keep if keep in (ids or []) else
                                    (ids[0] if ids else keep))
        for k, w in self.profile_widgets.items():
            w.clear()
            w.addItem("")
            w.addItems(ids or [])
            w.setCurrentText(cur_profiles.get(k, ""))
        self._show_status(f"✔ 已获取 {len(ids)} 个模型，主模型与角色下拉均可选择。")

    def _on_fetch_failed(self, msg):
        self._show_status(f"✘ {msg}", error=True)

    def _fetch_cleanup(self):
        self.b_fetch.setEnabled(True)
        self.b_fetch.setText("获取模型")

    def _show_status(self, text, error=False):
        color = "#D93025" if error else "#188038"
        self._status.setStyleSheet(f"color:{color}; font-size:11px;")
        self._status.setText(text)

    def done(self, r):
        if self._models_worker is not None and self._models_worker.isRunning():
            self._models_worker.wait(1500)
        super().done(r)

    def values(self) -> dict:
        return {
            "api_key": self.e_key.text().strip(),
            "base_url": self.e_url.text().strip(),
            "model": self.c_model.currentText().strip(),
            "profiles": {k: w.currentText().strip()
                         for k, w in self.profile_widgets.items()},
            "protocol": self.c_protocol.currentText(),
            "allow_shell": self.chk_shell.isChecked(),
            "python_enabled": self.chk_python.isChecked(),
            "web_enabled": self.chk_web.isChecked(),
            "skills_enabled": self.chk_skills.isChecked(),
            "theme": "light" if self.chk_light.isChecked() else "dark",
            "debug_mode": self.chk_debug.isChecked(),
            "confirm_dangerous": self.chk_confirm.isChecked(),
        }


# ============================================================
# 消息流里的折叠卡
# ============================================================
class ToolCollapseCard(QFrame):
    """工具执行过程卡：默认收起一行摘要，可展开看每次 action/observation"""

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self._c = colors
        self._lines = 0
        self._status_msg = ""
        self._done = False
        self._spin = 0
        self.setObjectName("toolCard")
        v = QVBoxLayout(self)
        v.setContentsMargins(9, 5, 9, 6)
        v.setSpacing(3)

        self.head = QToolButton(self)
        self.head.setObjectName("toolToggle")
        self.head.setText("▸ 正在执行…")
        self.head.setCheckable(True)
        self.head.setChecked(False)
        self.head.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.head.clicked.connect(self._on_toggle)
        v.addWidget(self.head)

        self.body = QWidget(self)
        self.body.setObjectName("toolBody")
        self.body_v = QVBoxLayout(self.body)
        self.body_v.setContentsMargins(2, 2, 2, 2)
        self.body_v.setSpacing(2)
        self.body.setVisible(False)
        v.addWidget(self.body)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(400)

    def _tick(self):
        if self._done:
            return
        self._spin = (self._spin + 1) % 4
        self._refresh_head()

    def _refresh_head(self):
        arrow = "▾" if self.head.isChecked() else "▸"
        n = self._lines
        if self._done:
            suffix = f" ✔ 执行过程 · {n} 项操作" if n else " ✔ 执行过程"
        else:
            dots = "." * self._spin
            suffix = f" 正在执行{dots}"
        status = f" {self._status_msg}" if self._status_msg else ""
        self.head.setText(f"{arrow}{suffix}{status}")

    def _on_toggle(self, checked):
        self.body.setVisible(bool(checked))
        self._refresh_head()

    def set_status(self, msg):
        """progress 事件：更新折叠头部的动态状态行（用户不展开也能看到进展）"""
        txt = str(msg or "").strip()
        if txt.startswith("〔联网〕"):
            txt = txt[len("〔联网〕"):].strip()
        self._status_msg = txt
        self._refresh_head()

    def _body_line(self, text: str):
        # V31.2：用 AutoHeightLabel（高度随文档自适应，封顶 260px + 内部滚动）
        # 旧的 setFixedHeight(document().size().height()) 在控件未布局时算到 0
        lab = AutoHeightLabel(str(text), self.body, objname="toolProgress",
                              max_height=260)
        self.body_v.addWidget(lab)
        self._lines += 1
        return lab

    def finish(self):
        """任务结束：停动画，状态定格（详情仍可展开查看）"""
        self._done = True
        self._status_msg = ""
        self._timer.stop()
        self._refresh_head()

    def add_action(self, name, arguments):
        args = json.dumps(arguments or {}, ensure_ascii=False)
        if len(args) > 300:
            args = args[:300] + "  …(已截断)"
        self._body_line(f"▸ {name} {args}").setObjectName("toolAction")
        self._refresh_head()

    def add_observation(self, content):
        text = str(content or "")
        if len(text) > 1200:
            text = text[:1200] + "…(已截断)"
        self._body_line(f"  {text}").setObjectName("toolObs")
        self._refresh_head()


class DebugCard(QFrame):
    """静态折叠卡：默认收起一行摘要；展开后显示 LLM 元信息 /
    解析判定（人话 note + 参数 JSON）/ 模型原始返回全文（等宽可复制）"""

    def __init__(self, colors, round_no, llm, parse, raw, expanded=False,
                 parent=None):
        super().__init__(parent)
        self._c = colors
        self.setObjectName("debugCard")
        v = QVBoxLayout(self)
        v.setContentsMargins(9, 4, 9, 6)
        v.setSpacing(3)

        ptype = str((parse or {}).get("type") or "?")
        if ptype == "action":
            tail = f"→ 执行 {parse.get('name')}"
        elif ptype == "answer":
            tail = "→ 最终回答"
        elif ptype == "bad_action":
            tail = "→ JSON 解析失败，已要求重试"
        elif ptype == "直答":
            tail = "→ 按直答处理"
        else:
            tail = f"→ {ptype}"
        skipped = (parse or {}).get("skipped") or 0
        if skipped:
            tail += f"（跳过 {skipped} 个坏段）"

        self.head = QToolButton(self)
        self.head.setObjectName("debugToggle")
        self.head.setCheckable(True)
        self.head.setChecked(bool(expanded))
        self.head.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.head.clicked.connect(self._on_toggle)
        v.addWidget(self.head)

        self.body = QWidget(self)
        self.body.setObjectName("toolBody")
        bv = QVBoxLayout(self.body)
        bv.setContentsMargins(2, 2, 2, 2)
        bv.setSpacing(3)

        llm = llm or {}
        meta = (f"模型输出 {llm.get('secs', '?')}s · {llm.get('chunks', '?')} 片 · "
                f"上下文 {llm.get('ctx_msgs', '?')} 条 / {llm.get('ctx_chars', '?')} 字符")
        bv.addWidget(self._label(meta, "debugMeta"))

        note = str((parse or {}).get("note") or "")
        if note:
            bv.addWidget(self._label(f"判定：{note}", "debugNote"))

        args = (parse or {}).get("arguments")
        if args:
            bv.addWidget(self._label(
                "参数：" + json.dumps(args, ensure_ascii=False), "debugMeta"))

        bv.addWidget(self._label("模型原始返回（可选中复制）：", "debugMeta"))
        raw_view = SelectableLabel(str(raw or ""), self.body, objname="debugRaw")
        raw_view.setFixedHeight(180)
        raw_view.setLineWrapMode(QTextEdit.WidgetWidth)
        bv.addWidget(raw_view)

        self.body.setVisible(bool(expanded))
        v.addWidget(self.body)

        self._summ = f" 🔍 调试 · 第 {round_no} 轮 {tail}"
        self._on_toggle(bool(expanded))

    def _label(self, text, objname):
        lab = QLabel(str(text), self)
        lab.setObjectName(objname)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return lab

    def _on_toggle(self, checked):
        arrow = "▾" if checked else "▸"
        self.head.setText(arrow + self._summ)
        self.body.setVisible(bool(checked))


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    confirm_signal = pyqtSignal("PyQt_PyObject", str, str)

    def __init__(self, config, workspace=None):
        super().__init__()
        self.config = config
        self.ui_scale = float(getattr(config, "ui_scale", 1.0) or 1.0)
        self.agent = Agent(LLMClient(config), workspace,
                           max_iterations=config.max_iterations)
        self.worker = None
        self._session_id = None
        self._session_filter = "all"
        self._search_query = ""
        self._chips = {}
        self._rounds = 0
        self._stream_view = None
        self._tool_card = None
        self._busy = False
        self._confirm_pending = False
        self._scale_timer = QTimer(self)
        self._scale_timer.setSingleShot(True)
        self._scale_timer.timeout.connect(self._save_scale_soon)

        self.watchdog = QTimer(self)
        self.watchdog.setSingleShot(True)
        self.watchdog.setInterval(45000)
        self.watchdog.timeout.connect(self._on_watchdog)

        self.confirm_signal.connect(self._on_confirm_dialog)

        self.setWindowTitle("PyAgent Studio")
        self.setMinimumSize(920, 680)
        self._icon_buttons = []
        self.resize(int(1200 * min(self.ui_scale, 1.4)),
                    int(820 * min(self.ui_scale, 1.4)))
        icon_path = RESOURCE_ROOT / "icons" / "app.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._build_ui()
        self._apply_theme()
        self._refresh_tree()
        self._refresh_status()
        self._restore_session()
        self._install_zoom_shortcut()
        QApplication.instance().installEventFilter(self)
        log.info("界面已就绪: %s", config_snapshot(config))

    def _set_icon(self, button, name, primary=False):
        self._icon_buttons.append((button, name, primary))
        button.setText("")
        button.setAccessibleName(button.toolTip() or name)


    # ---------- 布局 ----------
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())
        h.addWidget(self._build_main(), 1)

    def _build_sidebar(self):
        side = QWidget(self)
        side.setObjectName("sidebar")
        self.sidebar = side
        side.setFixedWidth(self._px(272))
        v = QVBoxLayout(side)
        v.setContentsMargins(18, 24, 18, 18)
        v.setSpacing(14)

        logo_row = QHBoxLayout()
        logo_row.setSpacing(8)
        badge = QLabel("P", side)
        badge.setObjectName("brandBadge")
        badge.setFixedSize(self._px(38), self._px(38))
        badge.setAlignment(Qt.AlignCenter)
        name_box = QVBoxLayout()
        name_box.setSpacing(0)
        name = QLabel("PyAgent", side)
        name.setObjectName("logoName")
        sub = QLabel("S T U D I O", side)
        sub.setObjectName("logoSub")
        name_box.addWidget(name)
        name_box.addWidget(sub)
        logo_row.addWidget(badge)
        logo_row.addLayout(name_box)
        logo_row.addStretch(1)
        v.addLayout(logo_row)

        self.b_new = QPushButton("＋   新建会话", side)
        self.b_new.setShortcut(QKeySequence("Alt+K"))
        self.b_new.setToolTip("新建会话 · Alt+K")
        self.b_new.setObjectName("newChatBtn")
        self.b_new.setCursor(Qt.PointingHandCursor)
        self.b_new.clicked.connect(self.new_chat)
        v.addWidget(self.b_new)

        self.search = QLineEdit(side)
        self.search.setObjectName("searchBox")
        self.search.setPlaceholderText("搜索会话…      Ctrl K")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search_changed)
        self.search.returnPressed.connect(self._on_search_enter)
        v.addWidget(self.search)

        tab_row = QHBoxLayout()
        tab_row.setSpacing(6)
        self.tab_chat = QPushButton("对话", side)
        self.tab_ws = QPushButton("工作台", side)
        for i, b in enumerate((self.tab_chat, self.tab_ws)):
            b.setObjectName("sideTab")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, idx=i: self._switch_side_tab(idx))
            tab_row.addWidget(b, 1)
        self.tab_chat.setChecked(True)
        v.addLayout(tab_row)

        # 筛选 chip（V25：带实时计数，互斥）
        chip_row = QGridLayout()
        chip_row.setSpacing(4)
        for i, (mode, text, tip) in enumerate((
                ("all", "全部会话", "全部会话"),
                ("active", "进行中", "24 小时内有活动的会话"),
                ("pinned", "置顶", "置顶的会话（右键会话可置顶）"),
                ("archived", "归档", "超过 7 天无活动的会话"))):
            chip = QPushButton(text, side)
            chip.setObjectName("filterChip")
            chip.setCheckable(True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setToolTip(tip)
            chip.clicked.connect(lambda _=False, m=mode: self._set_session_filter(m))
            self._chips[mode] = chip
            chip_row.addWidget(chip, i // 2, i % 2)
        self._chips["all"].setChecked(True)
        v.addLayout(chip_row)

        self.side_stack = QStackedWidget(side)
        self.side_stack.setObjectName("sideStack")

        self.tree = QTreeWidget(side)
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.tree.setColumnWidth(1, self._px(30))
        self.tree.setIndentation(self._px(12))
        self.tree.setRootIsDecorated(True)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.itemDoubleClicked.connect(self._on_tree_double)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.side_stack.addWidget(self.tree)
        self.side_stack.addWidget(self._build_workbench())

        v.addWidget(self.side_stack, 1)

        self.workspace_button = QPushButton("选择工作区  ↗", side)
        self.workspace_button.setObjectName("workspaceCard")
        self.workspace_button.setCursor(Qt.PointingHandCursor)
        self.workspace_button.clicked.connect(self.mount_workspace)
        v.addWidget(self.workspace_button)

        user_row = QHBoxLayout()
        user_row.setSpacing(8)
        av = QLabel("P", side)
        av.setObjectName("avatar")
        av.setFixedSize(self._px(30), self._px(30))
        av.setAlignment(Qt.AlignCenter)
        ub = QVBoxLayout()
        ub.setSpacing(0)
        un = QLabel("本地工作空间", side)
        un.setObjectName("userName")
        up = QLabel("PyAgent Desktop", side)
        up.setObjectName("userPlan")
        ub.addWidget(un)
        ub.addWidget(up)
        gear = QPushButton("⚙", side)
        gear.setObjectName("iconBtn")
        gear.setCursor(Qt.PointingHandCursor)
        gear.setFixedSize(self._px(30), self._px(30))
        gear.setToolTip("设置")
        self._set_icon(gear, "settings")
        gear.clicked.connect(self.open_settings)
        user_row.addWidget(av)
        user_row.addLayout(ub, 1)
        user_row.addWidget(gear)
        v.addLayout(user_row)
        return side

    def _build_workbench(self):
        """V25 工作台页：工作区文件树 + 快捷操作（挂载/卸载/在资源管理器打开）"""
        page = QWidget(self)
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 4, 0, 0)
        v.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(4)
        b_mount = QPushButton("📂 挂载/切换", page)
        b_mount.setObjectName("wsBtn")
        b_mount.setCursor(Qt.PointingHandCursor)
        b_mount.setToolTip("挂载或切换工作区目录")
        b_mount.clicked.connect(self.mount_workspace)
        b_unmount = QPushButton("⏏ 卸载", page)
        b_unmount.setObjectName("wsBtn")
        b_unmount.setCursor(Qt.PointingHandCursor)
        b_unmount.setToolTip("卸载当前工作区")
        b_unmount.clicked.connect(self.unmount_workspace)
        row.addWidget(b_mount, 1)
        row.addWidget(b_unmount)
        v.addLayout(row)

        b_open = QPushButton("🗖 打开目录", page)
        b_open.setObjectName("wsBtn")
        b_open.setCursor(Qt.PointingHandCursor)
        b_open.setToolTip("在资源管理器中打开工作区目录")
        b_open.clicked.connect(
            lambda: self._open_path(self.agent.workspace.root
                                    if self.agent.workspace else None))
        v.addWidget(b_open)

        self.ws_tree = QTreeWidget(page)
        self.ws_tree.setColumnCount(1)
        self.ws_tree.setHeaderHidden(True)
        self.ws_tree.setIndentation(self._px(12))
        self.ws_tree.itemDoubleClicked.connect(self._on_ws_tree_double)
        v.addWidget(self.ws_tree, 1)
        return page

    def _build_main(self):
        main = QWidget(self)
        v = QVBoxLayout(main)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = QWidget(main)
        header.setObjectName("header")
        hv = QHBoxLayout(header)
        hv.setContentsMargins(28, 18, 24, 18)
        hv.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        t = QLabel("智能工作台", header)
        t.setObjectName("chatTitle")
        self.title_sub = QLabel("从想法到完成，每一步都有助手", header)
        self.title_sub.setObjectName("chatSub")
        title_box.addWidget(t)
        title_box.addWidget(self.title_sub)
        hv.addLayout(title_box)
        hv.addStretch(1)
        self.state_badge = QLabel("● 就绪", header)
        self.state_badge.setObjectName("stateBadge")
        hv.addWidget(self.state_badge)

        for text, tip, fn in (
                ("✚", "新建会话", self.new_chat),
                ("📂", "挂载工作区", self.mount_workspace),
                ("🌙", "切换主题", self.toggle_theme),
                ("⚙", "设置", self.open_settings)):
            b = QPushButton(text, header)
            b.setObjectName("iconBtn")
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.setFixedSize(self._px(34), self._px(34))
            b.clicked.connect(fn)
            if text == "🌙":
                self.b_theme = b
            elif text == "⚙":
                self.b_h_settings = b
            elif text == "✚":
                self.b_h_new = b
            else:
                self.b_h_mount = b
            self._set_icon(b, {"✚": "plus", "📂": "folder", "🌙": "moon", "⚙": "settings"}[text])
            hv.addWidget(b)
        v.addWidget(header)

        # 消息流
        self.scroll = QScrollArea(main)
        self.scroll.setObjectName("chatScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        chat_host = QWidget(self.scroll)
        self.chat_v = QVBoxLayout(chat_host)
        self.chat_v.setContentsMargins(36, 28, 36, 28)
        self.chat_v.setSpacing(20)
        self.chat_v.addStretch(1)
        self.scroll.setWidget(chat_host)
        self.content_stack = QStackedWidget(main)
        self.content_stack.addWidget(self._build_welcome())
        self.content_stack.addWidget(self.scroll)
        v.addWidget(self.content_stack, 1)

        # 输入卡
        input_wrap = QWidget(main)
        iw_v = QVBoxLayout(input_wrap)
        self.input_layout = iw_v
        iw_v.setContentsMargins(32, 8, 32, 18)
        iw_v.setSpacing(10)

        card = QFrame(input_wrap)
        card.setObjectName("inputCard")
        card_v = QVBoxLayout(card)
        card_v.setContentsMargins(18, 14, 18, 12)
        card_v.setSpacing(10)

        self.input = InputBox(card)
        self.input.setObjectName("inputBox")
        self.input.setPlaceholderText("描述你的任务，让 PyAgent 帮你完成…")
        self.input.setAccessibleName("任务输入框")
        self.input.setFixedHeight(self._px(64))
        self.input.submitted.connect(self._on_send_clicked)
        card_v.addWidget(self.input)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.b_plus = QPushButton("＋", card)
        self.b_plus.setObjectName("inputActionBtn")
        self.b_plus.setCursor(Qt.PointingHandCursor)
        self.b_plus.setToolTip("工作区快捷操作")
        self._set_icon(self.b_plus, "plus")
        self.b_plus.clicked.connect(self._show_plus_menu)
        self.b_star = QPushButton("✦", card)
        self.b_star.setObjectName("inputActionBtn")
        self.b_star.setCursor(Qt.PointingHandCursor)
        self.b_star.setToolTip("会话快捷操作")
        self._set_icon(self.b_star, "spark")
        self.b_star.clicked.connect(self._show_star_menu)
        action_row.addWidget(self.b_plus)
        action_row.addWidget(self.b_star)

        self.agent_combo = QComboBox(card)
        self.agent_combo.setObjectName("agentCombo")
        self.agent_combo.addItems(["Agent", "仅提问"])
        self.agent_combo.setToolTip(
            "Agent 模式：可调用工具执行任务；仅提问：纯对话，不调用工具")
        self.agent_combo.currentTextChanged.connect(self._on_mode_changed)
        action_row.addWidget(self.agent_combo)
        action_row.addStretch(1)

        self.model_label = QLabel("", card)
        self.model_label.setObjectName("hintLabel")
        self.model_label.setToolTip("点击打开设置")
        self.model_label.mousePressEvent = lambda _e: self.open_settings()
        action_row.addWidget(self.model_label)
        card_v.addLayout(action_row)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)
        hint = QLabel("Enter 发送  ·  Shift + Enter 换行", card)
        hint.setToolTip("Agent 可读写已挂载工作区的文件，请核对重要操作。Ctrl + 滚轮缩放界面，Ctrl+0 复位。")
        hint.setObjectName("hintLabel")
        self.hint = hint
        bottom_row.addWidget(hint)
        bottom_row.addStretch(1)
        self.b_send = QPushButton("↑", card)
        self.b_send.setObjectName("sendBtn")
        self.b_send.setCursor(Qt.PointingHandCursor)
        self.b_send.setToolTip("发送")
        self._set_icon(self.b_send, "arrow", True)
        self.b_send.setFixedSize(self._px(40), self._px(36))
        self.b_send.clicked.connect(self._on_send_clicked)
        bottom_row.addWidget(self.b_send)
        card_v.addLayout(bottom_row)

        iw_v.addWidget(card)
        self.status_label = QLabel("", input_wrap)
        self.status_label.setObjectName("statusLabel")
        iw_v.addWidget(self.status_label)
        v.addWidget(input_wrap)
        return main

    def _build_welcome(self):
        page = QWidget(self)
        outer = QVBoxLayout(page)
        outer.setSizeConstraint(QLayout.SetMinimumSize)
        outer.setContentsMargins(32, 16, 32, 16)
        outer.addStretch(1)
        panel = QWidget(page)
        panel.setMaximumWidth(760)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        mark = QLabel("P", panel)
        self.welcome_mark = mark
        mark.setObjectName("welcomeMark")
        mark.setFixedSize(56, 56)
        mark.setAlignment(Qt.AlignCenter)
        layout.addWidget(mark, 0, Qt.AlignHCenter)
        eyebrow = QLabel("你的 AI 工作伙伴", panel)
        self.welcome_eyebrow = eyebrow
        eyebrow.setObjectName("welcomeEyebrow")
        eyebrow.setAlignment(Qt.AlignCenter)
        layout.addWidget(eyebrow)
        title = QLabel("让繁杂的工作，简单一点。", panel)
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        subtitle = QLabel("整理文件、探索代码、搜集信息。\n告诉我目标，我们一起把它变成结果。", panel)
        subtitle.setObjectName("welcomeSub")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        grid = QGridLayout()
        grid.setSpacing(12)
        prompts = (
            ("folder", "整理工作区", "梳理文件结构，快速了解项目", "请先查看当前工作区的文件结构，梳理项目组成并给出整理建议，暂时不要修改文件。"),
            ("code", "读懂代码", "理解实现逻辑，发现改进方向", "请分析当前工作区的代码，说明项目入口、核心模块与运行方式，并给出改进建议。"),
            ("globe", "研究一个主题", "搜集公开信息，提炼关键结论", "我想研究一个主题，请先询问我的研究方向和范围，再帮我搜集资料并注明来源。"),
            ("file", "起草一份文档", "从零开始，写出清晰的初稿", "请帮我起草一份文档。先询问文档主题、目标读者与篇幅，再开始写作。"),
        )
        self._prompt_icons = []
        for i, (icon, heading, detail, prompt) in enumerate(prompts):
            button = QPushButton(panel)
            button.setObjectName("promptCard")
            button.setAccessibleName(heading)
            button.setMinimumHeight(82)
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip("将任务填入输入框，可编辑后发送")
            row = QHBoxLayout(button)
            row.setContentsMargins(16, 12, 16, 12)
            row.setSpacing(12)
            icon_label = QLabel(button)
            icon_label.setFixedSize(22, 22)
            self._prompt_icons.append((icon_label, icon))
            row.addWidget(icon_label)
            words = QVBoxLayout()
            words.setSpacing(5)
            name = QLabel(heading, button)
            name.setObjectName("promptTitle")
            description = QLabel(detail, button)
            description.setObjectName("promptDetail")
            words.addWidget(name)
            words.addWidget(description)
            row.addLayout(words, 1)
            for label in (icon_label, name, description):
                label.setAttribute(Qt.WA_TransparentForMouseEvents)
            button.clicked.connect(lambda _=False, p=prompt: self._use_prompt(p))
            grid.addWidget(button, i // 2, i % 2)
        layout.addLayout(grid)
        center = QHBoxLayout()
        center.addStretch(1)
        center.addWidget(panel, 8)
        center.addStretch(1)
        outer.addLayout(center)
        outer.addStretch(1)
        welcome_scroll = QScrollArea(self)
        welcome_scroll.setObjectName("welcomeScroll")
        welcome_scroll.setWidgetResizable(True)
        welcome_scroll.setFrameShape(QFrame.NoFrame)
        welcome_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        welcome_scroll.setWidget(page)
        return welcome_scroll

    def _use_prompt(self, prompt):
        self.input.setPlainText(prompt)
        self.input.setFocus()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "input_layout"):
            roomy = self.height() >= self._px(740)
            self.welcome_mark.setVisible(roomy)
            self.welcome_eyebrow.setVisible(roomy)
            width = self.width() - self.sidebar.width()
            margin = max(self._px(32), (width - self._px(960)) // 2)
            self.input_layout.setContentsMargins(margin, 8, margin, 18)
            self.chat_v.setContentsMargins(margin, 28, margin, 28)

    # ---------- 主题与缩放 ----------
    def _apply_theme(self):
        colors = THEMES.get(self.config.theme, THEMES["dark"])
        self._c = colors
        self.setStyleSheet(build_qss(colors, self.ui_scale))
        for label, name in self._prompt_icons:
            label.setPixmap(line_icon(name, colors["accent"]).pixmap(22, 22))
        for button, name, primary in self._icon_buttons:
            if button is getattr(self, "b_theme", None):
                name = "sun" if self.config.theme == "dark" else "moon"
            if button is getattr(self, "b_send", None):
                name = "stop" if self._busy else "arrow"
            button.setIcon(line_icon(name, colors["accent_text"] if primary else colors["text2"]))
            button.setIconSize(QSize(self._px(18), self._px(18)))
        if hasattr(self, "b_theme"):
            is_dark = self.config.theme == "dark"
            self.b_theme.setToolTip("切换为亮色" if is_dark else "切换为暗色")

    def _px(self, n):
        """按当前缩放系数换算像素值"""
        return max(1, int(round(n * self.ui_scale)))

    def _install_zoom_shortcut(self):
        QShortcut(QKeySequence("Ctrl+0"), self, activated=lambda: self.zoom(reset=True))
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self._focus_search)

    def _focus_search(self):
        if hasattr(self, "search"):
            self._switch_side_tab(0)
            self.search.setFocus()

    def eventFilter(self, obj, ev):
        """应用级过滤器：任何控件上按住 Ctrl 滚轮都缩放界面（事件不再传给目标控件）"""
        if ev.type() == QEvent.Wheel and (ev.modifiers() & Qt.ControlModifier):
            self.zoom(up=ev.angleDelta().y() > 0)
            return True
        if ev.type() == QEvent.KeyPress and obj is getattr(self, "search", None):
            if ev.key() == Qt.Key_Escape:
                self.search.clear()
                return True
        return super().eventFilter(obj, ev)

    def zoom(self, up=True, reset=False):
        if reset:
            self.ui_scale = 1.0       # 复位到出厂默认（100%）
        else:
            step = ZOOM_STEP if up else -ZOOM_STEP
            self.ui_scale = max(ZOOM_MIN, min(ZOOM_MAX, self.ui_scale + step))
        self.config.ui_scale = self.ui_scale
        self._apply_theme()
        self._resize_fixed()
        self._relayout_streams()
        self._flash_zoom_status()
        self._scale_timer.start(800)

    def _flash_zoom_status(self):
        if hasattr(self, "status_label"):
            self.status_label.setText(
                f"界面缩放 {int(round(self.ui_scale * 100))}% （Ctrl+0 复位）")

    def _save_scale_soon(self):
        """滚轮连续缩放时防抖 800ms 再写配置"""
        try:
            self.config.save()
        except Exception as e:
            log.warning("缩放配置保存失败: %s", e)
        self._refresh_status()

    def _resize_fixed(self):
        """缩放后重设代码里写死的固定尺寸控件"""
        if hasattr(self, "input"):
            self.input.setFixedHeight(self._px(64))
        if hasattr(self, "b_send"):
            self.b_send.setFixedSize(self._px(40), self._px(36))
        if hasattr(self, "sidebar"):
            self.sidebar.setFixedWidth(self._px(272))

    def _relayout_streams(self):
        """缩放后按新字号重算历史消息块高度"""
        for w in (getattr(self, "_stream_view", None),):
            if w is not None:
                w._sync_height()          # V31.2：AutoHeightLabel 自带同步方法
        self._scroll_bottom()

    def toggle_theme(self):
        self.config.theme = "light" if self.config.theme == "dark" else "dark"
        self._apply_theme()
        try:
            self.config.save()
        except Exception:
            pass
        self._refresh_status()

    def _scroll_bottom(self):
        def _go():
            bar = self.scroll.verticalScrollBar()
            bar.setValue(bar.maximum())
        QTimer.singleShot(0, _go)

    # ---------- 消息渲染 ----------
    def _add_widget(self, w):
        self.chat_v.insertWidget(self.chat_v.count() - 1, w)
        self._scroll_bottom()

    def _append_sys(self, text):
        lab = QLabel(str(text), self)
        lab.setObjectName("sysLine")
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._add_widget(lab)

    def _append_error(self, text):
        lab = QLabel("✘ " + str(text), self)
        lab.setObjectName("errLine")
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._add_widget(lab)

    def _append_user(self, text):
        self.content_stack.setCurrentIndex(1)
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 2, 0, 2)
        h.addStretch(1)
        bubble = QLabel(html.escape(str(text)).replace("\n", "<br>"), row)
        bubble.setObjectName("userBubble")
        bubble.setWordWrap(True)
        bubble.setTextFormat(Qt.RichText)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setMaximumWidth(self._px(620))
        h.addWidget(bubble)
        self._add_widget(row)

    def _start_assistant(self, meta=""):
        self.content_stack.setCurrentIndex(1)
        row = QWidget(self)
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(6)
        av = QLabel("P", row)
        av.setObjectName("assistAvatar")
        av.setFixedSize(self._px(26), self._px(26))
        av.setAlignment(Qt.AlignCenter)
        nm = QLabel("PyAgent", row)
        nm.setObjectName("assistName")
        head.addWidget(av)
        head.addWidget(nm)
        self._meta_label = QLabel(str(meta), row)
        self._meta_label.setObjectName("assistMeta")
        head.addWidget(self._meta_label)
        head.addStretch(1)
        v.addLayout(head)

        self._stream_view = AutoHeightLabel("", row, objname="streamText",
                                            max_height=2400)
        self._stream_view.setPlaceholderText("")
        self._stream_view.setFixedHeight(self._px(24))
        v.addWidget(self._stream_view)
        self._add_widget(row)

    def _stream(self, text):
        if self._stream_view is None:
            self._start_assistant()
        cur = self._stream_view.toPlainText() + str(text)
        self._stream_view.setPlainText(cur)
        # 高度由 AutoHeightLabel 自行同步（V31.2：手动按 document().size()
        # 算高在控件未布局时恒为 0，正文被裁成一行）
        self._scroll_bottom()

    def _finish_assistant(self):
        if self._stream_view is not None:
            self._render_markdown(self._stream_view)
        self._stream_view = None

    def _render_markdown(self, view):
        """任务完成时把正文整体重渲染为 Markdown 富文本（流式期间为纯文本）"""
        text = view.toPlainText()
        if not text.strip():
            return
        try:
            from ui.md_render import md_to_html
            body = re.sub(r"^\s*ANSWER:\s*", "", text, flags=re.M)
            view.setHtml(md_to_html(body))
            # 高度由 AutoHeightLabel 自动同步（setHtml 同样触发 documentSizeChanged）
        except Exception as e:                    # 渲染失败保留纯文本
            log.warning("Markdown 渲染失败，保留纯文本: %s", e)

    def _ensure_tool_card(self):
        if self._tool_card is None:
            self._tool_card = ToolCollapseCard(self._c, self)
            self._add_widget(self._tool_card)
        return self._tool_card

    def on_event(self, ev):
        t = ev.get("type")
        self._feed_watchdog()
        if t == "keepalive":                      # 只喂看门狗，不渲染
            return
        if t == "delta":
            self._stream(ev.get("content") or "")
            return
        if t == "debug":
            card = DebugCard(self._c, ev.get("round"), ev.get("llm"),
                             ev.get("parse"), ev.get("raw"),
                             expanded=bool(getattr(self.config, "debug_mode", False)),
                             parent=self)
            self._add_widget(card)
            return
        if t == "progress":
            self._ensure_tool_card().set_status(ev.get("content") or "")
            return
        if t == "action":
            self._finish_assistant()
            self._ensure_tool_card().add_action(ev.get("name"),
                                                ev.get("arguments"))
            return
        if t == "observation":
            self._ensure_tool_card().add_observation(ev.get("content") or "")
            return
        if t == "answer":
            self._stream(ev.get("content") or "") if self._stream_view is None \
                else self._stream(ev.get("content") or "")
            self._finish_assistant()
            if self._tool_card is not None:
                self._tool_card.finish()
                self._tool_card = None
            self.title_sub.setText("✔ 任务完成")
            return
        if t == "cancelled":
            self._finish_assistant()
            self._append_sys("已停止")
            if self._tool_card is not None:
                self._tool_card.finish()
                self._tool_card = None
            self.title_sub.setText("已停止")
            return
        if t == "redirect":
            self._append_sys("⟳ " + str(ev.get("content") or ""))
            return
        if t == "error":
            self._finish_assistant()
            self._append_error(ev.get("content") or "")
            if self._tool_card is not None:
                self._tool_card.finish()
                self._tool_card = None
            self.title_sub.setText("出错了")
            return
        if t == "format_error":
            self._append_sys("⟳ " + str(ev.get("content") or ""))
            return
        if t == "done":
            self._finish()
            return

    def _finish(self):
        self._busy = False
        self.watchdog.stop()
        if hasattr(self, "b_send"):
            self.b_send.setIcon(line_icon("arrow", self._c["accent_text"]))
            self.b_send.setToolTip("发送")
        if self._stream_view is not None:
            self._finish_assistant()
        if self._tool_card is not None:
            self._tool_card.finish()
            self._tool_card = None
        if self.title_sub.text() in ("运行中…", ""):
            self.title_sub.setText("就绪")
        self._refresh_status()
        self._refresh_tree()
        self._save_current_session()

    def _on_send_clicked(self):
        if self._busy:
            self.stop()
        else:
            self.send()

    def send(self):
        if self._busy:
            QMessageBox.information(self, "提示",
                                    "Agent 正在执行任务，请等待或点击「停止」。")
            return
        if not self.config.configured:
            QMessageBox.warning(self, "未配置 API Key",
                                "请先在「设置…」中填写 API Key。")
            self.open_settings()
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._append_user(text)
        self._rounds += 1
        self._finish_assistant()
        self._tool_card = None
        self._start_assistant(f"第 {self._rounds} 轮")
        self._busy = True
        self.title_sub.setText("运行中…")
        self.b_send.setIcon(line_icon("stop", self._c["accent_text"]))
        self.b_send.setToolTip("停止")
        self._refresh_status()

        self.worker = AgentWorker(self.agent, text, self._confirm_dangerous, self)
        self.worker.agent_event.connect(self.on_event)
        self._feed_watchdog()
        self.worker.start()

    def stop(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.title_sub.setText("正在停止…")

    # ---------- 看门狗 ----------
    def _feed_watchdog(self):
        """任何事件到达都续命；任务未运行时不启动"""
        if self._busy and not self._confirm_pending:
            self.watchdog.start(45000)

    def _on_watchdog(self):
        """45s 无任何响应 = 对话超时：提示用户并自动停止"""
        if not self._busy or self._confirm_pending:
            return
        log.warning("看门狗触发: 45s 无事件，判定对话超时")
        self._append_error("⚠ 对话超时：长时间无响应，已自动停止。"
                           "（可能网络中断或服务端无响应，请检查后重试）")
        self.stop()

    # ---------- 高危操作确认（跨线程） ----------
    def _confirm_dangerous(self, name, reason):
        """AgentWorker 线程调用：把确认请求转派主线程弹框，阻塞等待结果。

        确认框的模态事件循环期间看门狗可能触发——_confirm_pending 已豁免；
        等待上限 3 分钟，超时按拒绝处理（不阻塞 worker 线程到天荒地老）。"""
        # 回调线程与主线程共享明确的过期标记，晚到的点击不得被误记为允许。
        box = {"ok": False, "ev": threading.Event(),
               "lock": threading.Lock(), "expired": False}
        self._confirm_pending = True
        try:
            self.confirm_signal.emit(box, str(name), str(reason))
        except Exception as e:
            log.warning("确认信号发送失败，按拒绝处理: %s", e)
            self._confirm_pending = False
            return False
        got = box["ev"].wait(180)
        with box["lock"]:
            if not got:
                box["expired"] = True
            allowed = got and bool(box["ok"])
        self._confirm_pending = False
        self._feed_watchdog()
        if not got:
            log.warning("高危确认等待超时(180s)，本次未执行: %s", name)
            return False
        return allowed

    @pyqtSlot(object, str, str)
    def _on_confirm_dialog(self, box, name, reason):
        """主线程槽：弹确认框，结果写回 box 并放行 worker 线程"""
        try:
            with box["lock"]:
                if box["expired"]:
                    log.warning("丢弃已过期的高危确认弹窗: %s", name)
                    return
            dialog = QMessageBox(QMessageBox.Question, "高危操作确认",
                f"Agent 请求执行高危操作：\n\n{name}\n\n{reason}\n\n是否允许执行？",
                QMessageBox.Yes | QMessageBox.No, self)
            dialog.setDefaultButton(QMessageBox.No)
            # 弹框与工作线程使用同一个截止时间；防止过期弹窗一直挂着。
            timer = QTimer(dialog)
            timer.setSingleShot(True)
            timer.timeout.connect(dialog.reject)
            timer.start(179000)
            ans = dialog.exec_()
            timer.stop()
            with box["lock"]:
                if box["expired"]:
                    log.warning("忽略过期后才作出的高危确认结果: %s", name)
                    return
                box["ok"] = (ans == QMessageBox.Yes)
            log.info("高危操作确认: %s → %s", name,
                     "允许" if box["ok"] else "拒绝或超时")
        except Exception as e:
            with box["lock"]:
                box["ok"] = False
            log.warning("确认框异常，按拒绝处理: %s", e)
        finally:
            box["ev"].set()

    # ---------- 会话 ----------
    def _save_current_session(self):
        """当前对话归档到 sessions/<id>.json（含工作区路径）"""
        msgs = []
        for m in self.agent.messages:
            role = m.get("role")
            if role == "system":
                continue
            content = m.get("content") or ""
            if not content:
                continue
            msgs.append({"role": role, "content": content})
        if not msgs:
            return
        if self._session_id is None:
            self._session_id = new_session_id()
        title = ""
        for m in msgs:
            if m["role"] == "user" and not m["content"].startswith("OBSERVATION"):
                title = m["content"][:40]
                break
        ws = str(self.agent.workspace.root) if self.agent.workspace else ""
        try:
            save_session(self._session_id, msgs, ws, title)
            log.info("会话已保存: %s (%d 条消息)", title[:20] or "(无标题)", len(msgs))
        except Exception as e:
            log.warning("会话保存失败: %s", e)

    def _restore_session(self):
        """启动时恢复最近会话（消息+工作区）；无会话历史则恢复 config.last_workspace"""
        sid = ""
        try:
            ss = list_sessions()
            sid = ss[0]["id"] if ss else ""
        except Exception as e:
            log.warning("会话列表读取失败: %s", e)
        if not sid:
            self._restore_last_workspace()
            return
        data = load_session(sid) or {}
        msgs = data.get("messages") or []
        ws_path = data.get("workspace") or ""
        if ws_path:
            try:
                self.agent.set_workspace(Workspace(ws_path))
                log.info("已恢复会话工作区: %s", ws_path)
            except WorkspaceError as e:
                log.warning("工作区恢复失败: %s", e)
        self._session_id = sid
        for m in msgs:                 # 上下文装回内存，界面只重放对话主线
            role, content = _msg_pair(m)
            if role == "system" or not role:
                continue
            self.agent.messages.append(
                dict(m) if isinstance(m, dict)
                else {"role": role, "content": content})
        self._replay(msgs)
        rounds = sum(1 for m in msgs
                     if _msg_pair(m)[0] == "user"
                     and not str(_msg_pair(m)[1]).startswith("OBSERVATION"))
        self._append_sys(f"已恢复最近会话（{rounds} 轮，标题「"
                         f"{data.get('title', '')}」）—— 双击左侧「历史会话」可切换。")
        self._refresh_status()
        self._refresh_workbench()

    def _restore_last_workspace(self):
        """无会话历史时：恢复上一次挂载的工作区（config.last_workspace）"""
        path = getattr(self.config, "last_workspace", "") or ""
        if not path:
            return
        try:
            self.agent.set_workspace(Workspace(path))
            log.info("已恢复上次的工作区: %s", path)
            self._refresh_workbench()
        except WorkspaceError as e:
            log.warning("上次的工作区已不可访问: %s", e)

    def _replay(self, messages):
        """历史消息重放到界面（跳过 OBSERVATION 等过程消息，只保留对话主线）"""
        for m in messages or []:
            role, content = _msg_pair(m)
            if role == "user":
                if content.startswith("OBSERVATION"):
                    continue
                self._append_user(content)
            elif role == "assistant":
                self._start_assistant("历史")
                self._stream(content)
                self._finish_assistant()

    def _on_tree_double(self, item, _col):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind = data[0]
        if kind == "session":
            self._open_session_by_id(data[1])
        elif kind == "path":
            self._open_path(Path(data[1]))
        elif kind == "current":
            self.input.setFocus()
        elif kind == "workspace":
            self._open_path(self.agent.workspace.root
                            if self.agent.workspace else None)

    def _on_tree_context_menu(self, pos):
        """V25：历史会话右键菜单（打开 / 置顶 / 删除）"""
        item = self.tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data or data[0] != "session":
            return
        sid = data[1]
        cur = next((s for s in list_sessions() if s["id"] == sid), None)
        title = (cur or {}).get("title", "")
        menu = QMenu(self)
        menu.addAction("打开会话", lambda: self._open_session_by_id(sid))
        pinned = bool((cur or {}).get("pinned"))
        label = "取消置顶" if pinned else "📌 置顶"
        menu.addAction(label, lambda: self._toggle_pin(sid, not pinned))
        menu.addSeparator()
        menu.addAction("🗑 删除会话", lambda: self._delete_session(sid, title))
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def _toggle_pin(self, sid, pinned):
        if set_session_pinned(sid, pinned):
            self._refresh_tree()
            self._append_sys(f"会话「{self._title_of(sid)}」"
                             + ("已置顶" if pinned else "已取消置顶"))

    def _title_of(self, sid):
        for s in list_sessions():
            if s["id"] == sid:
                return s.get("title", "")
        return ""

    def _open_session_by_id(self, sid):
        """打开指定历史会话（双击 / 回车 / 右键菜单共用）"""
        if self._busy:
            QMessageBox.information(self, "提示", "请等待当前任务结束再切换会话。")
            return
        data = load_session(sid)
        if data is None:
            QMessageBox.warning(self, "提示", "会话文件读取失败。")
            return
        self._save_current_session()
        self.agent.reset()
        msgs = data.get("messages") or []
        ws_path = data.get("workspace") or ""
        if ws_path:
            try:
                self.agent.set_workspace(Workspace(ws_path))
            except WorkspaceError as e:
                log.warning("会话工作区不可访问: %s", e)
                self.agent.set_workspace(None)
        else:
            self.agent.set_workspace(None)
        for m in msgs:
            role, content = _msg_pair(m)
            if role == "system":
                continue
            if role == "user" and content.startswith("OBSERVATION"):
                continue
            self.agent.messages.append(
                dict(m) if isinstance(m, dict)
                else {"role": role, "content": content})
        self._clear_chat()
        self._replay(msgs)
        self._session_id = sid
        rounds = sum(1 for m in msgs
                     if _msg_pair(m)[0] == "user"
                     and not str(_msg_pair(m)[1]).startswith("OBSERVATION"))
        self._append_sys(f"已切换到会话「{data.get('title', '')}」（{rounds} 轮）。")
        self._refresh_status()
        self._refresh_tree()
        self._refresh_workbench()

    def new_chat(self):
        if self._busy:
            QMessageBox.information(self, "提示", "请等待当前任务结束再新建会话。")
            return
        self._save_current_session()
        self.agent.reset()
        self._session_id = None
        self._clear_chat()
        self._append_sys("已开启新会话。")
        self._refresh_status()
        self._refresh_tree()
        self.input.setFocus()

    def _delete_session(self, sid, title=""):
        """删除历史会话（V24）：删文件；若删的是当前会话，连内存对话一起清空。

        删除是不可恢复操作——先弹确认框；删当前会话时 agent.messages 一并
        reset（否则下一轮 _save_current_session 会把内容又写回磁盘）。"""
        if self._busy:
            QMessageBox.information(self, "提示", "请等待当前任务结束再删除会话。")
            return
        ans = QMessageBox.question(
            self, "删除会话",
            f"确定删除会话「{title}」？\n\n该会话的全部对话与历史记录将被删除，且不可恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        ok = delete_session(sid)
        if not ok:
            QMessageBox.warning(self, "提示", "会话文件删除失败，请查看运行日志。")
            return
        is_current = (sid == self._session_id)
        if is_current:
            self.agent.reset()
            self._clear_chat()
            self._session_id = None
            self._append_sys(f"会话「{title}」已删除（含全部对话与历史记录）。")
        else:
            self._append_sys(f"历史会话「{title}」已删除。")
        log.info("用户删除会话: %s（%s）", sid, title)
        self._refresh_tree()
        self._refresh_status()

    def _clear_chat(self):
        """清空对话区（保留欢迎系统行），立即从 UI 层级摘除"""
        while self.chat_v.count() > 1:
            item = self.chat_v.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._stream_view = None
        self._tool_card = None
        self.content_stack.setCurrentIndex(0)

    def _show_plus_menu(self):
        """＋ 按钮：工作区快捷操作"""
        menu = QMenu(self)
        menu.addAction("📂 挂载/切换工作区", self.mount_workspace)
        menu.addAction("⏏ 卸载工作区", self.unmount_workspace)
        menu.addAction("🗖 在资源管理器打开工作区",
                       lambda: self._open_path(self.agent.workspace.root
                                               if self.agent.workspace else None))
        menu.exec_(self.b_plus.mapToGlobal(self.b_plus.rect().topLeft()))

    def _show_star_menu(self):
        """✦ 按钮：会话快捷操作"""
        menu = QMenu(self)
        menu.addAction("✚ 新建会话", self.new_chat)
        menu.addAction("🧹 清空当前对话", self._reset_current_chat)
        menu.addAction("💾 导出对话为 Markdown…", self._export_session_md)
        menu.exec_(self.b_star.mapToGlobal(self.b_star.rect().topLeft()))

    def _reset_current_chat(self):
        """清空当前对话（不删会话文件，相当于开新会话）"""
        if self._busy:
            QMessageBox.information(self, "提示", "请等待当前任务结束再清空对话。")
            return
        self.agent.reset()
        self._clear_chat()
        self._append_sys("当前对话已清空（会话历史文件保留）。")

    def _export_session_md(self):
        """导出当前会话对话为 Markdown（跳过 OBSERVATION 过程消息）"""
        lines = []
        for m in self.agent.messages:
            role = m.get("role")
            if role == "system":
                continue
            content = m.get("content") or ""
            if not content or content.startswith("OBSERVATION"):
                continue
            if role == "user":
                lines.append(f"## 🧑 用户\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"## 🤖 助手\n\n{content}\n")
        if not lines:
            QMessageBox.information(self, "提示", "当前会话还没有可导出的对话。")
            return
        title = (self._title_of(self._session_id) if self._session_id
                 else "会话") or "会话"
        safe = re.sub(r"[/\\:*?\"<>|]", "_", title) or "导出对话"
        default = str(Path.home() / f"{safe}.md")
        path, _flt = QFileDialog.getSaveFileName(
            self, "导出对话", default, "Markdown (*.md)")
        if not path:
            return
        try:
            Path(path).write_text(f"# {title}\n\n" + "\n".join(lines),
                                  encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        log.info("会话已导出: %s", path)
        self._append_sys(f"已导出对话 → {path}")

    def _on_mode_changed(self, text):
        """Agent / 仅提问 模式切换（V25）"""
        qa = str(text).strip() == "仅提问"
        if qa == self.agent.qa_mode:
            return
        self.agent.set_qa_mode(qa)
        self._append_sys("已切换到「仅提问」模式：只对话，不调用工具。" if qa
                         else "已切换到「Agent」模式：可调用工具执行任务。")
        self._refresh_status()

    # ---------- 工作区 ----------
    def mount_workspace(self):
        if self._busy:
            QMessageBox.information(self, "提示", "请等待当前任务结束再挂载工作区。")
            return
        path = QFileDialog.getExistingDirectory(self, "选择工作区目录", "")
        if not path:
            return
        try:
            self.agent.set_workspace(Workspace(path))
        except WorkspaceError as e:
            QMessageBox.warning(self, "挂载失败", str(e))
            return
        self.config.last_workspace = str(self.agent.workspace.root)
        try:
            self.config.save()
        except Exception:
            pass
        self._append_sys(f"已挂载工作区: {self.agent.workspace.root}")
        self._refresh_status()
        self._refresh_tree()
        self._refresh_workbench()

    def unmount_workspace(self):
        if self._busy:
            QMessageBox.information(self, "提示", "请等待当前任务结束再卸载。")
            return
        self.agent.set_workspace(None)
        self.config.last_workspace = ""
        try:
            self.config.save()
        except Exception:
            pass
        self._append_sys("已卸载工作区")
        self._refresh_status()
        self._refresh_tree()
        self._refresh_workbench()

    def open_settings(self):
        dlg = SettingsDialog(self.config, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        v = dlg.values()
        self.config.api_key = v["api_key"]
        self.config.base_url = v["base_url"]
        self.config.model = v["model"]
        for key, model in (v.get("profiles") or {}).items():
            p = self.config.profile(key)
            p.model = model or ""
        self.config.protocol = v["protocol"]
        self.config.allow_shell = v["allow_shell"]
        self.config.python_enabled = v["python_enabled"]
        self.config.web_enabled = v["web_enabled"]
        self.config.skills_enabled = v["skills_enabled"]
        self.config.theme = v["theme"]
        self.config.debug_mode = v["debug_mode"]
        self.config.confirm_dangerous = v["confirm_dangerous"]
        try:
            self.config.save()
        except Exception as e:
            log.warning("配置保存失败: %s", e)
        self.agent.set_llm(LLMClient(self.config))
        self._apply_theme()
        self._refresh_status()
        self._append_sys("设置已保存。")

    # ---------- 状态与侧栏刷新 ----------
    def _refresh_status(self):
        ws = (f"工作区: {self.agent.workspace.root}"
              if self.agent.workspace else "工作区: 未挂载")
        key = "API Key 已配置" if self.config.configured else "未配置 API Key"
        run = "运行中" if self._busy else "空闲"
        mode = "仅提问" if self.agent.qa_mode else "Agent"
        proto = ("原生工具"
                 if (getattr(self.config, "protocol", "auto") != "text")
                 else "文本协议")
        model = self.config.model or "?"
        skills = ""
        if getattr(self.config, "skills_enabled", True):
            try:
                from core.skills import discover
                n = len(discover(self.config, self.agent.workspace))
                skills = f" · 技能: {n}"
            except Exception:
                skills = ""
        text = (f"▣ {ws} · {key} · {run} · 模式: {mode} · 协议: {proto}"
                f" · 模型: {model}{skills} · 主题: {self.config.theme}")
        if hasattr(self, "status_label"):
            self.status_label.setText(
                f"{'执行中' if self._busy else '准备就绪'}  ·  {mode} 模式"
                f"  ·  {'工作区已连接' if self.agent.workspace else '连接工作区以处理本地文件'}")
            self.status_label.setToolTip(text)
        if hasattr(self, "state_badge"):
            self.state_badge.setText("● 执行中" if self._busy else (
                "● 就绪" if self.config.configured else "○ 待配置"))
        if hasattr(self, "workspace_button"):
            root = self.agent.workspace.root if self.agent.workspace else None
            self.workspace_button.setText(
                f"工作区 · {Path(root).name[:20]}  ↗" if root else "连接工作区  ↗")
            self.workspace_button.setToolTip(str(root) if root else "选择一个本地文件夹，开始处理文件任务")
        if hasattr(self, "model_label"):
            self.model_label.setText(f"模型 {model}")
        if hasattr(self, "agent_combo"):
            want = "仅提问" if self.agent.qa_mode else "Agent"
            if self.agent_combo.currentText() != want:
                self.agent_combo.blockSignals(True)
                self.agent_combo.setCurrentText(want)
                self.agent_combo.blockSignals(False)

    def _refresh_tree(self):
        if not hasattr(self, "tree"):
            return
        # 先显式摘除行内按钮再 clear：QTreeWidget.clear() 与 setItemWidget
        # 的控件所有权存在竞争，直接 clear 在部分平台会崩（offscreen 可复现）
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                node = top.child(j)
                w = self.tree.itemWidget(node, 1)
                if w is not None:
                    self.tree.removeItemWidget(node, 1)
                    w.setParent(None)
                    w.deleteLater()
        self.tree.clear()
        sessions = list_sessions()
        mode = getattr(self, "_session_filter", "all")
        q = (getattr(self, "_search_query", "") or "").strip().lower()
        shown = filter_sessions(sessions, mode)
        if q:
            hit = [s for s in shown if q in str(s.get("title", "")).lower()]
            if not hit:                       # 标题未命中 → 搜内容
                hit = self._search_session_content(shown, q)
            shown = hit

        root = QTreeWidgetItem(self.tree)
        label = f"历史会话 · {len(shown)}" if shown else "历史会话"
        root.setText(0, label)
        root.setData(0, Qt.UserRole, ("root", ""))
        root.setExpanded(True)
        for s in shown:
            node = QTreeWidgetItem(root)
            mark = "📌 " if s.get("pinned") else ""
            prefix = "● " if s["id"] == self._session_id else ""
            node.setText(0, f"{mark}{prefix}{s.get('title') or s['id']}")
            node.setData(0, Qt.UserRole, ("session", s["id"]))
            node.setToolTip(0, "\n双击切换会话 · 右键置顶/删除 · 点右侧 ✕ 删除")
            btn = QPushButton("✕", self.tree)
            btn.setObjectName("delSessBtn")
            btn.setFixedSize(self._px(20), self._px(20))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip("删除该会话（对话与历史记录一并删除，不可恢复）")
            btn.clicked.connect(
                lambda _=False, sid=s["id"], t=s.get("title", ""):
                self._delete_session(sid, t))
            self.tree.setItemWidget(node, 1, btn)
        if not shown:
            empty = QTreeWidgetItem(root)
            empty.setText(0, (f"无匹配「{q}」的会话" if q else "（暂无会话）"))
            empty.setData(0, Qt.UserRole, ("none", ""))

        # 工作区节点（文件树）
        ws = self.agent.workspace
        if ws is not None:
            ws_node = QTreeWidgetItem(self.tree)
            ws_node.setText(0, f"📁 {ws.root.name}")
            ws_node.setData(0, Qt.UserRole, ("workspace", str(ws.root)))
            ws_node.setToolTip(0, "双击在资源管理器中打开")
            ws_node.setExpanded(True)
            try:
                entries = sorted(ws.root.iterdir(),
                                 key=lambda p: (p.is_file(), p.name.lower()))
            except OSError:
                entries = []
            for p in entries[:50]:
                child = QTreeWidgetItem(ws_node)
                icon = "📁" if p.is_dir() else "📄"
                child.setText(0, f"{icon} {p.name}")
                child.setData(0, Qt.UserRole, ("path", str(p)))
                child.setToolTip(0, "双击用默认应用打开")
            if len(entries) > 50:
                more = QTreeWidgetItem(ws_node)
                more.setText(0, f"… 其余 {len(entries) - 50} 项")
                more.setData(0, Qt.UserRole, ("none", ""))

        # 当前会话节点
        cur = QTreeWidgetItem(self.tree)
        rounds = sum(1 for m in self.agent.messages
                     if m.get("role") == "user"
                     and not str(m.get("content") or "").startswith("OBSERVATION"))
        cur.setText(0, f"当前会话 · {rounds} 轮" if rounds else "当前会话")
        cur.setData(0, Qt.UserRole, ("current", ""))
        cur.setToolTip(0, "双击聚焦输入框")
        self._update_chip_counts()

    def _set_session_filter(self, mode):
        """筛选 chip 切换（互斥）"""
        self._session_filter = mode if mode in self._chips else "all"
        for m, b in self._chips.items():
            b.setChecked(m == self._session_filter)
        self._refresh_tree()

    def _on_search_changed(self, text):
        self._search_query = str(text or "")
        self._refresh_tree()

    def _on_search_enter(self):
        """回车打开第一个匹配的会话"""
        root = self.tree.topLevelItem(0)
        if root is None:
            return
        for i in range(root.childCount()):
            data = root.child(i).data(0, Qt.UserRole)
            if data and data[0] == "session":
                self._open_session_by_id(data[1])
                return

    def _search_session_content(self, sessions, q):
        """标题未命中时搜消息内容（限量 20 个文件，避免卡 UI）"""
        out = []
        for s in sessions[:20]:
            data = load_session(s["id"]) or {}
            for m in data.get("messages") or []:
                _role, content = _msg_pair(m)
                if q in str(content).lower():
                    out.append(s)
                    break
        return out

    def _update_chip_counts(self):
        """chip 文案带实时计数（原来写死"全部 0"）"""
        sessions = list_sessions()
        names = {"all": "全部", "active": "进行中", "pinned": "置顶",
                 "archived": "归档"}
        for mode, b in getattr(self, "_chips", {}).items():
            n = len(filter_sessions(sessions, mode))
            b.setText(f"{names.get(mode, mode)} {n}")

    def _switch_side_tab(self, idx):
        """对话 / 工作台 Tab 切换（互斥 + 堆叠页）"""
        idx = 1 if idx else 0
        if hasattr(self, "side_stack"):
            self.side_stack.setCurrentIndex(idx)
        if hasattr(self, "tab_chat"):
            self.tab_chat.setChecked(idx == 0)
            self.tab_ws.setChecked(idx == 1)
        if idx == 1:
            self._refresh_workbench()

    def _refresh_workbench(self):
        """重建工作台文件树（未挂载显示空态）"""
        if not hasattr(self, "ws_tree"):
            return
        self.ws_tree.clear()
        ws = self.agent.workspace
        if ws is None:
            node = QTreeWidgetItem(self.ws_tree)
            node.setText(0, "未挂载工作区 —— 点上方「📂 挂载/切换」")
            node.setData(0, Qt.UserRole, ("none", ""))
            return
        node = QTreeWidgetItem(self.ws_tree)
        node.setText(0, f"📁 {ws.root.name}")
        node.setData(0, Qt.UserRole, ("workspace", str(ws.root)))
        node.setToolTip(0, "双击在资源管理器中打开")
        node.setExpanded(True)
        try:
            entries = sorted(ws.root.iterdir(),
                             key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            entries = []
        for p in entries[:200]:
            child = QTreeWidgetItem(node)
            icon = "📁" if p.is_dir() else "📄"
            child.setText(0, f"{icon} {p.name}")
            child.setData(0, Qt.UserRole, ("path", str(p)))
            child.setToolTip(0, "双击用默认应用打开")

    def _on_ws_tree_double(self, item, _col):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        if data[0] == "path":
            self._open_path(Path(data[1]))
        elif data[0] == "workspace":
            self._open_path(Path(data[1]))

    def _open_path(self, p):
        """用系统默认方式打开文件/目录（Win 资源管理器/默认应用，macOS open，Linux xdg-open）"""
        if p is None:
            QMessageBox.information(self, "提示", "尚未挂载工作区。")
            return
        path = Path(p)
        if not path.exists():
            QMessageBox.warning(self, "提示", f"路径不存在: {path}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            log.warning("打开路径失败: %s | %s", path, e)
            QMessageBox.warning(self, "打开失败", str(e))

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        self._save_current_session()
        super().closeEvent(event)


def run_gui(config, workspace=None):
    """启动图形界面（main.py 调用）"""
    app = QApplication.instance() or QApplication(sys.argv)
    if config.theme:
        app.setStyle("Fusion")
    win = MainWindow(config, workspace)
    win.show()
    return app.exec_()
