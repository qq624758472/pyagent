# simple_wechat.py
"""
一个极简的 "微信" 模拟器：
    • 左侧是联系人列表（仅作演示）
    • 右侧是聊天记录
    • 下方输入框 + 发送按钮
    • 本地 "聊天服务器" 把文字保存并立即回显
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QTextEdit,
    QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
)
from PyQt5.QtCore import Qt

# 本地 "聊天服务器"——仅仅负责把消息记录下来并返回回显
class SimpleChatServer:
    def __init__(self):
        self.history = {}      # {(from_user,to_user):[msg1, …]}

    def send_message(self, from_user: str, to_user: str, content: str) -> str:
        key = (from_user, to_user)
        self.history.setdefault(key, []).append(content)
        return f"{from_user} → {to_user}: {content}"

# 主窗口
class WeChatWindow(QMainWindow):
    def __init__(self, username: str, server: SimpleChatServer):
        super().__init__()
        self.username = username
        self.server   = server
        self.setWindowTitle(f"PyQt5 WeChat – {username}")
        self.resize(800, 500)
        self._build_ui()

    # UI 组合
    def _build_ui(self):
        central = QWidget()
        hbox = QHBoxLayout(central)

        # 左侧——联系人
        self.contact_list = QListWidget()
        for name in ("Alice", "Bob", "Charlie"):
            if name != self.username:
                self.contact_list.addItem(name)
        self.contact_list.currentRowChanged.connect(self._on_contact_changed)

        hbox.addWidget(self.contact_list, 1)

        # 右侧——聊天记录 + 输入区
        right = QWidget()
        vbox = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        vbox.addWidget(QLabel("聊天记录:"))
        vbox.addWidget(self.chat_display, 8)

        # 输入 + 发送
        input_layout = QHBoxLayout()
        self.message_edit = QLineEdit()
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self._send)

        input_layout.addWidget(self.message_edit, 8)
        input_layout.addWidget(self.send_btn, 1)
        vbox.addLayout(input_layout)

        right.setLayout(vbox)
        hbox.addWidget(right, 4)

        self.setCentralWidget(central)

        # 目前选择的联系人
        self._current_contact = None

    # 当联系人切换时
    def _on_contact_changed(self, index):
        if index is not None and index >= 0:
            self._current_contact = self.contact_list.item(index).text()
        else:
            self._current_contact = None

    # 发送按钮
    def _send(self):
        if not self._current_contact:
            # 先把自己给自己发
            self._current_contact = self.username
        content = self.message_edit.text().strip()
        if not content:
            return
        # 先把“服务器”里放一条
        echo = self.server.send_message(
            self.username, self._current_contact, content
        )
        # 将消息展示到自己的聊天窗口
        self.chat_display.append(echo)
        # 也把同一条写到被会话联系人聊天窗口——这里简单演示只更新自己的
        self.chat_display.append(echo)

if __name__ == "__main__":
    # 若未安装 PyQt5，请先执行 `pip install pyqt5`
    app = QApplication(sys.argv)
    # 为演示先自己设定一个名称，实际项目可以改为真正的账号
    username = "Me"
    server = SimpleChatServer()
    win = WeChatWindow(username, server)
    win.show()
    sys.exit(app.exec_())