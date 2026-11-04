"""界面主题：配色表 + QSS 生成

与窗口逻辑解耦：调配色/字号只需要改这里，不必碰 main_window。
所有尺寸都乘以缩放系数 s（Ctrl+滚轮），保证整体等比缩放。
"""

# ============================================================
# 产品主题：石墨蓝 / 云白，搭配鸢尾紫强调色
# ============================================================
THEMES = {
    "dark": {
        "base": "#151923",        # 石墨蓝侧栏
        "surface": "#1B202C",     # 主聊天区
        "elevated": "#232A38",    # 卡片/气泡/输入容器
        "input_bg": "#1B202C",
        "hover": "#2D3546",
        "text": "#EDF1FA",
        "text2": "#A9B3C7",
        "text3": "#8490A7",
        "accent": "#8B91F8",      # 柔和灰蓝（降低饱和度）
        "accent_text": "#FFFFFF",
        "success": "#4ADE80",
        "error": "#E05252",
        "border": "#30384A",
        "input_border": "#3C465C",
        "user_bubble": "#303A57",
        "tool_bg": "#202735",
        "scrollbar": "#414C63",
        "avatar_bg": "#34395D",   # 头像：柔蓝灰，不刺眼
        "avatar_fg": "#C4C7FF",
        "send_bg": "#6269D6",
        "send_border": "#6269D6",
    },
    "light": {
        "base": "#F3F4F8",
        "surface": "#FCFCFE",
        "elevated": "#FFFFFF",
        "input_bg": "#FFFFFF",
        "hover": "#E9EBF3",
        "text": "#111827",
        "text2": "#6B7280",
        "text3": "#788397",
        "accent": "#6468D9",
        "accent_text": "#FFFFFF",
        "success": "#16A34A",
        "error": "#DC2626",
        "border": "#E5E7EB",
        "input_border": "#D1D5DB",
        "user_bubble": "#EDEFFD",
        "tool_bg": "#F5F6F8",
        "scrollbar": "#D1D5DB",
        "avatar_bg": "#DCE4F2",
        "avatar_fg": "#4A5B80",
        "send_bg": "#6468D9",
        "send_border": "#D1D5DB",
    },
}

MONO_FAMILIES = "'Consolas','JetBrains Mono','Courier New',monospace"
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.7, 2.5, 0.1


def _msg_pair(m):
    """会话文件里 messages 存的是 [role, content] 对，内存里是 dict
    （system/tool_calls 等结构只有内存态有）——统一取成 (role, content)"""
    if isinstance(m, dict):
        return m.get("role"), (m.get("content") or "")
    if isinstance(m, (list, tuple)) and len(m) >= 2:
        return m[0], (m[1] or "")
    return None, ""


def build_qss(c, s=1.0):
    """s = 界面缩放系数（Ctrl+滚轮调整），所有字号/间距/圆角同步缩放"""
    f = lambda n: int(round(n * s))
    return f"""
    QMainWindow, QDialog {{ background: {c['surface']}; }}
    QWidget {{ font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei"; color: {c['text']}; font-size: {f(13)}px; }}

    /* ---------- 侧栏 ---------- */
    #sidebar {{ background: {c['base']}; border-right: 1px solid {c['border']}; }}
    #logoName {{ color: {c['text']}; font-size: {f(15)}px; font-weight: 700; }}
    #logoSub {{ color: {c['text3']}; font-size: {f(10)}px; letter-spacing: 2px; }}
    #newChatBtn {{
        background: {c['elevated']}; border: 1px solid {c['border']};
        border-radius: {f(9)}px; padding: {f(8)}px {f(12)}px; text-align: left;
        color: {c['text']}; font-size: {f(13)}px;
    }}
    #newChatBtn:hover {{ background: {c['hover']}; }}
    #searchBox {{
        background: {c['input_bg']}; border: 1px solid {c['border']};
        border-radius: {f(8)}px; padding: {f(7)}px {f(10)}px; color: {c['text2']};
    }}
    #sideTab {{
        background: transparent; border: none; color: {c['text2']};
        padding: {f(6)}px 2px; font-size: {f(13)}px;
    }}
    #sideTab:checked {{ color: {c['text']}; border-bottom: 2px solid {c['accent']}; }}
    #filterChip {{
        background: transparent; border: 1px solid {c['border']};
        border-radius: {f(10)}px; padding: {f(3)}px {f(10)}px; color: {c['text2']}; font-size: {f(11)}px;
    }}
    #filterChip:checked {{ background: {c['elevated']}; color: {c['text']}; }}
    QTreeWidget {{
        background: transparent; border: none; outline: none;
        color: {c['text']}; font-size: {f(12)}px;
    }}
    QTreeWidget::item {{ padding: {f(4)}px 2px; border-radius: {f(6)}px; }}
    QTreeWidget::item:hover {{ background: {c['hover']}; }}
    QTreeWidget::item:selected {{ background: {c['elevated']}; }}
    QTreeWidget::branch {{ background: transparent; }}
    #delSessBtn {{
        background: transparent; border: none; border-radius: 4px;
        color: {c['text3']}; font-size: 11px; padding: 0px; margin: 0px;
    }}
    #delSessBtn:hover {{ background: {c['error']}; color: #FFFFFF; }}
    QStackedWidget#sideStack {{ background: transparent; }}
    #wsBtn {{
        background: transparent; border: 1px solid {c['border']};
        border-radius: {f(7)}px; padding: {f(5)}px {f(8)}px; color: {c['text2']};
        font-size: {f(11)}px;
    }}
    #wsBtn:hover {{ background: {c['hover']}; }}
    QMenu {{
        background: {c['elevated']};
        border: 1px solid {c['border']}; padding: {f(4)}px;
    }}
    QMenu::item {{ padding: {f(6)}px {f(14)}px; border-radius: {f(5)}px; }}
    QMenu::item:selected {{ background: {c['hover']}; }}
    QMenu::separator {{ height: 1px; background: {c['border']}; margin: {f(4)}px; }}
    #sideGroup {{ color: {c['text3']}; font-size: {f(11)}px; }}
    #avatar {{
        background: {c['avatar_bg']}; color: {c['avatar_fg']};
        border-radius: {f(16)}px; font-weight: 700; font-size: {f(13)}px;
    }}
    #userName {{ color: {c['text']}; font-size: {f(13)}px; font-weight: 600; }}
    #userPlan {{ color: {c['text3']}; font-size: {f(11)}px; }}

    /* ---------- 主区头部 ---------- */
    #header {{ background: {c['surface']}; border-bottom: 1px solid {c['border']}; }}
    #chatTitle {{ color: {c['text']}; font-size: {f(14)}px; font-weight: 600; }}
    #chatSub {{ color: {c['text2']}; font-size: {f(11)}px; }}
    #iconBtn {{
        background: transparent; border: none; border-radius: {f(8)}px;
        padding: {f(6)}px; color: {c['text2']}; font-size: {f(14)}px;
    }}
    #iconBtn:hover {{ background: {c['hover']}; }}

    /* ---------- 消息流 ---------- */
    #chatScroll {{ background: {c['surface']}; border: none; }}
    #chatScroll > QWidget > QWidget {{ background: transparent; }}
    QLabel#userBubble {{
        background: {c['user_bubble']}; border-radius: {f(10)}px;
        padding: {f(8)}px {f(12)}px; color: {c['text']};
    }}
    #assistName {{ color: {c['text']}; font-size: {f(13)}px; font-weight: 700; }}
    #assistMeta {{ color: {c['text3']}; font-size: {f(11)}px; }}
    QToolButton#messageAction {{
        background: transparent; color: {c['text2']}; border: 1px solid {c['border']};
        border-radius: {f(6)}px; padding: {f(3)}px {f(8)}px; font-size: {f(11)}px;
    }}
    QToolButton#messageAction:hover, QToolButton#messageAction:checked {{
        background: {c['hover']}; color: {c['text']};
    }}
    #assistAvatar {{
        background: {c['avatar_bg']}; color: {c['avatar_fg']};
        border-radius: {f(13)}px; font-size: {f(12)}px; font-weight: 700;
    }}
    QTextEdit#streamText {{
        background: transparent; border: none; color: {c['text']};
        font-size: {f(13)}px; selection-background-color: {c['accent']};
    }}
    QFrame#toolCard {{
        background: {c['tool_bg']}; border: 1px solid {c['border']};
        border-radius: {f(8)}px;
    }}
    QToolButton#toolToggle {{
        background: transparent; border: none; color: {c['text2']};
        font-size: {f(12)}px; text-align: left;
    }}
    QToolButton#toolToggle:hover {{ color: {c['text']}; }}
    QFrame#debugCard {{
        background: {c['tool_bg']}; border: 1px dashed {c['input_border']};
        border-radius: {f(8)}px;
    }}
    QToolButton#debugToggle {{
        background: transparent; border: none; color: {c['text3']};
        font-size: {f(11)}px; text-align: left;
        font-family: {MONO_FAMILIES};
    }}
    QToolButton#debugToggle:hover {{ color: {c['text']}; }}
    #debugRaw {{
        color: {c['text2']}; font-size: {f(11)}px;
        font-family: {MONO_FAMILIES};
        background: transparent; border: 1px solid {c['border']};
        border-radius: {f(5)}px;
    }}
    #debugMeta {{ color: {c['text3']}; font-size: {f(11)}px; }}
    #debugNote {{
        color: {c['accent']}; font-size: {f(11)}px;
    }}
    #toolAction {{
        color: {c['text2']}; font-size: {f(11)}px; font-family: {MONO_FAMILIES};
        background: transparent; border: none;
    }}
    #toolObs {{
        color: {c['text3']}; font-size: {f(11)}px; font-family: {MONO_FAMILIES};
        background: transparent; border: none;
    }}
    #toolProgress {{
        color: {c['accent']}; font-size: {f(11)}px;
        background: transparent; border: none;
    }}
    #sysLine {{ color: {c['text3']}; font-size: {f(11)}px; }}
    #errLine {{ color: {c['error']}; font-size: {f(11)}px; }}

    /* ---------- 输入区 ---------- */
    QFrame#inputCard {{
        background: {c['elevated']}; border: 1px solid {c['input_border']};
        border-radius: {f(12)}px;
    }}
    QFrame#inputCard:hover {{ border-color: {c['accent']}; }}
    #inputBox {{ background: transparent; border: none; color: {c['text']};
        font-size: {f(13)}px; }}
    #inputActionBtn {{
        background: transparent; border: 1px solid {c['border']};
        border-radius: {f(7)}px; padding: {f(3)}px {f(7)}px; color: {c['text2']};
        font-size: {f(12)}px;
    }}
    #inputActionBtn:hover {{ background: {c['hover']}; }}
    QComboBox#agentCombo {{
        background: transparent; border: 1px solid {c['border']};
        border-radius: {f(7)}px; padding: {f(3)}px {f(7)}px; color: {c['text2']};
        font-size: {f(11)}px;
    }}
    QComboBox#agentCombo::drop-down {{ border: none; width: {f(14)}px; }}
    QComboBox#agentCombo QAbstractItemView {{
        background: {c['elevated']}; color: {c['text']};
        selection-background-color: {c['hover']};
    }}
    #sendBtn {{
        background: {c['send_bg']}; border: 1px solid {c['send_border']};
        border-radius: {f(15)}px; color: {c['text']};
        font-size: {f(13)}px; font-weight: 700;
    }}
    #sendBtn:hover {{ background: {c['accent']}; color: {c['accent_text']}; border: none; }}
    #sendBtn:disabled {{ background: {c['hover']}; color: {c['text3']}; }}
    #hintLabel {{ color: {c['text3']}; font-size: {f(10)}px; }}
    #statusLabel {{ color: {c['text3']}; font-size: {f(11)}px; }}

    QScrollBar:vertical {{
        background: transparent; width: {f(10)}px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c['scrollbar']}; border-radius: {f(5)}px;
        min-height: {f(24)}px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    /* ---------- Studio 产品组件 ---------- */
    QPushButton {{ outline: none; }}
    QPushButton:focus, QToolButton:focus {{ border: 1px solid {c['accent']}; }}
    QToolTip {{ background: {c['elevated']}; color: {c['text']};
        border: 1px solid {c['border']}; padding: 8px; }}
    #brandBadge, #welcomeMark {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {c['accent']}, stop:1 #5357B5);
        color: #FFFFFF; border-radius: {f(12)}px;
        font-size: {f(22)}px; font-weight: 700;
    }}
    #welcomeMark {{ border-radius: {f(18)}px; font-size: {f(30)}px; }}
    #welcomeScroll, #welcomeScroll > QWidget > QWidget {{ background: {c['surface']}; }}
    #settingsTitle {{ font-size: {f(22)}px; font-weight: 700; }}
    #logoName {{ font-size: {f(18)}px; }}
    #logoSub {{ font-size: {f(9)}px; color: {c['text2']}; }}
    #newChatBtn {{ background: {c['send_bg']}; color: {c['accent_text']};
        border: none; padding: {f(12)}px; border-radius: {f(9)}px; font-weight: 600; }}
    #newChatBtn:hover {{ background: #5357B5; }}
    #searchBox {{ padding: {f(10)}px; font-size: {f(12)}px; }}
    #searchBox:focus {{ border-color: {c['accent']}; color: {c['text']}; }}
    #sideTab {{ padding: {f(10)}px; font-weight: 600; }}
    #sideTab:checked {{ background: {c['elevated']}; border-bottom: none;
        border-radius: {f(7)}px; color: {c['accent']}; }}
    #filterChip {{ padding: {f(5)}px {f(8)}px; border: none; text-align: left; }}
    #filterChip:checked {{ background: {c['avatar_bg']}; color: {c['avatar_fg']}; }}
    QTreeWidget::item {{ padding: {f(8)}px 2px; }}
    #workspaceCard {{ background: {c['elevated']}; border: 1px solid {c['border']};
        border-radius: {f(10)}px; padding: {f(14)}px {f(10)}px;
        color: {c['text2']}; text-align: left; font-size: {f(12)}px; }}
    #workspaceCard:hover {{ border-color: {c['accent']}; color: {c['text']}; }}
    #chatTitle {{ font-size: {f(16)}px; }}
    #chatSub {{ color: {c['text3']}; padding-top: {f(4)}px; }}
    #stateBadge {{ background: {c['avatar_bg']}; color: {c['avatar_fg']};
        border-radius: {f(10)}px; padding: {f(5)}px {f(10)}px; font-size: {f(11)}px; }}
    #welcomeEyebrow {{ color: {c['accent']}; font-size: {f(12)}px; font-weight: 600; }}
    #welcomeTitle {{ color: {c['text']}; font-size: {f(30)}px; font-weight: 700; }}
    #welcomeSub {{ color: {c['text2']}; font-size: {f(13)}px; line-height: 1.7; }}
    #promptCard {{ background: {c['elevated']}; border: 1px solid {c['border']};
        border-radius: {f(12)}px; padding: {f(16)}px; text-align: left;
        color: {c['text2']}; font-size: {f(12)}px; }}
    #promptCard:hover {{ border-color: {c['accent']}; background: {c['hover']}; color: {c['text']}; }}
    #promptCard:focus {{ border-color: {c['accent']}; }}
    #promptTitle {{ color: {c['text']}; font-size: {f(13)}px; font-weight: 600; }}
    #promptDetail {{ color: {c['text2']}; font-size: {f(11)}px; }}
    QFrame#inputCard {{ border-radius: {f(16)}px; }}
    #inputBox {{ font-size: {f(14)}px; selection-background-color: {c['accent']}; }}
    #inputActionBtn {{ border: none; padding: {f(5)}px; border-radius: {f(6)}px; }}
    QComboBox#agentCombo {{ padding: {f(5)}px {f(10)}px; }}
    #sendBtn {{ background: {c['send_bg']}; border: none; border-radius: {f(10)}px; }}
    #sendBtn:hover {{ background: {c['accent']}; }}
    #statusLabel {{ font-size: {f(11)}px; color: {c['text3']}; }}
    QLabel#userBubble {{ padding: {f(12)}px {f(16)}px; border-radius: {f(14)}px; }}
    QDialog QLineEdit, QDialog QComboBox {{ background: {c['input_bg']};
        border: 1px solid {c['input_border']}; border-radius: {f(6)}px;
        padding: {f(8)}px; selection-background-color: {c['accent']}; }}
    QDialog QLineEdit:focus, QDialog QComboBox:focus {{ border-color: {c['accent']}; }}
    QDialog QPushButton {{ background: {c['elevated']}; border: 1px solid {c['border']};
        border-radius: {f(6)}px; padding: {f(8)}px {f(16)}px; }}
    QDialog QPushButton:hover {{ border-color: {c['accent']}; }}
    QDialog QPushButton:default {{ background: {c['accent']}; color: {c['accent_text']}; }}
    QDialog QCheckBox {{ spacing: {f(8)}px; padding: {f(4)}px 0; }}
    """
