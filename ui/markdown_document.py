"""Qt 自带的 GitHub Markdown 解析与主题样式，无额外依赖。"""
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import (QColor, QFont, QTextCursor, QTextDocument, QTextFormat,
                         QTextTable, QTextLength, QTextFrameFormat)


class MarkdownDocument(QTextDocument):
    def loadResource(self, resource_type, name):
        if resource_type == QTextDocument.ImageResource:
            return QVariant()  # 不因回答中的图片地址访问网络或本地文件。
        return super().loadResource(resource_type, name)


def render_markdown(document, source, colors=None, scale=1.0):
    c = colors or {}
    foreground = QColor(c.get('text', '#202530'))
    muted = QColor(c.get('text2', '#657085'))
    background = QColor(c.get('tool_bg', '#F2F4F8'))
    border = QColor(c.get('border', '#D9DFEA'))
    font = QFont('Microsoft YaHei UI')
    font.setPixelSize(max(10, round(14 * scale)))
    document.setDefaultFont(font)
    document.setDocumentMargin(round(6 * scale))
    document.setMarkdown(source, QTextDocument.MarkdownFeatures(
        QTextDocument.MarkdownDialectGitHub | QTextDocument.MarkdownNoHTML))
    block = document.begin()
    while block.isValid():
        cursor = QTextCursor(block)
        fmt = block.blockFormat()
        fmt.setTopMargin(3 * scale)
        fmt.setBottomMargin(9 * scale)
        fmt.setLineHeight(145, fmt.ProportionalHeight)
        heading = fmt.headingLevel()
        code = fmt.hasProperty(QTextFormat.BlockCodeFence)
        quote = fmt.intProperty(QTextFormat.BlockQuoteLevel)
        if heading:
            fmt.setTopMargin(18 * scale)
            fmt.setBottomMargin(10 * scale)
        if code:
            fmt.setBackground(background)
            fmt.setLeftMargin(12 * scale)
            fmt.setRightMargin(12 * scale)
            fmt.setBottomMargin(1 * scale)
            fmt.setTopMargin(1 * scale)
        elif quote:
            fmt.setLeftMargin(18 * scale)
        cursor.setBlockFormat(fmt)
        fragment = block.begin()
        while not fragment.atEnd():
            run = fragment.fragment()
            if run.isValid():
                cur = QTextCursor(document)
                cur.setPosition(run.position())
                cur.setPosition(run.position() + run.length(), QTextCursor.KeepAnchor)
                cf = run.charFormat()
                cf.setForeground(QColor(c.get('accent', '#5965C7')) if cf.isAnchor()
                                 else muted if quote else foreground)
                if heading:
                    cf.setFontPointSize((22, 18, 15, 13, 12, 11)[min(heading, 6)-1] * scale)
                if code or cf.fontFixedPitch():
                    cf.setFontFamily('Consolas')
                    cf.setFontPointSize(10 * scale)
                    cf.setBackground(background)
                cur.setCharFormat(cf)
            fragment += 1
        block = block.next()
    frames = list(document.rootFrame().childFrames())
    while frames:
        frame = frames.pop()
        frames.extend(frame.childFrames())
        if isinstance(frame, QTextTable):
            fmt = frame.format()
            fmt.setBorder(0.7)
            fmt.setBorderBrush(border)
            fmt.setBorderStyle(QTextFrameFormat.BorderStyle_Solid)
            fmt.setCellPadding(8 * scale)
            fmt.setCellSpacing(0)
            fmt.setHeaderRowCount(1)
            fmt.setWidth(QTextLength(QTextLength.PercentageLength, 100))
            frame.setFormat(fmt)
            for col in range(frame.columns()):
                cell = frame.cellAt(0, col)
                cf = cell.format()
                cf.setBackground(background)
                cell.setFormat(cf)
