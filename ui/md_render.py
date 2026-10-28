"""轻量 Markdown → HTML（QTextEdit 专用，无第三方依赖）

支持：代码块```、行内代码、粗体、斜体、标题 #~####、无序/有序列表、
引用 >、链接 [t](u)、水平线、换行。所有输入先做 HTML 转义——注入安全。
流式期间按纯文本渲染，任务完成时整体重渲染为富文本（见 _finish_assistant）。"""
import html
import re


def _inline(text: str) -> str:
    """行内元素：行内代码 → 粗体 → 斜体 → 链接"""
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)",
               r'<a href="\2">\1</a>', t)
    return t


_HEAD = {1: 20, 2: 17, 3: 15, 4: 13}


def md_to_html(md: str) -> str:
    """把 Markdown 文本转为 QTextEdit 可显示的 HTML 片段"""
    out = []
    lines = md.split("\n")
    i, n = 0, len(lines)
    in_code, code_buf = False, []

    while i < n:
        line = lines[i]

        # ---- 代码块 ----
        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_buf),
                                                       quote=False) + "</code></pre>")
                code_buf, in_code = [], False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        stripped = line.strip()

        # ---- 空行 ----
        if not stripped:
            out.append("<br>")
            i += 1
            continue

        # ---- 水平线 ----
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            out.append("<hr>")
            i += 1
            continue

        # ---- 标题 ----
        m = re.match(r"(#{1,4})\s+(.*)", stripped)
        if m:
            level = len(m.group(1))
            size = _HEAD[level]
            out.append(f'<p style="font-size:{size}px; font-weight:bold;'
                       f' margin:{6 if level > 1 else 10}px 0 4px 0;">'
                       f"{_inline(m.group(2))}</p>")
            i += 1
            continue

        # ---- 引用 ----
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append('<p style="color:#888; border-left:3px solid #555;'
                       f'padding-left:10px; margin:4px 0;">'
                       f"{_inline(' '.join(quote_lines))}</p>")
            continue

        # ---- 无序列表 ----
        if re.match(r"[-*+]\s+", stripped):
            items = []
            while i < n and re.match(r"[-*+]\s+", lines[i].strip()):
                text = re.sub(r"^[-*+]\s+", "", lines[i].strip())
                items.append(f"<li>{_inline(text)}</li>")
                i += 1
            out.append("<ul style='margin:4px 0 4px 22px;'>" + "".join(items) + "</ul>")
            continue

        # ---- 有序列表 ----
        if re.match(r"\d+[.)]\s+", stripped):
            items = []
            while i < n and re.match(r"\d+[.)]\s+", lines[i].strip()):
                text = re.sub(r"^\d+[.)]\s+", "", lines[i].strip())
                items.append(f"<li>{_inline(text)}</li>")
                i += 1
            out.append("<ol style='margin:4px 0 4px 26px;'>" + "".join(items) + "</ol>")
            continue

        # ---- 普通段落 ----
        out.append(f"<p style='margin:3px 0;'>{_inline(stripped)}</p>")
        i += 1

    if in_code:                                   # 未闭合的代码块兜底
        out.append("<pre><code>" + html.escape("\n".join(code_buf),
                                                quote=False) + "</code></pre>")
    return "".join(out)
