"""Qt 自带的 GitHub Markdown 解析与主题样式，无额外依赖。"""
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import (QColor, QFont, QTextCursor, QTextDocument, QTextFormat,
                         QTextTable, QTextLength, QTextFrameFormat)
from PyQt5.QtWidgets import QTextBrowser, QTextEdit
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices, QTextCharFormat, QTextListFormat
import re


def readable_markdown(text):
    """Markdown 原样解析；普通日志保留换行，避免挤成一段。"""
    text=str(text or '')
    if re.search(r'(?m)^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|```|>|\|)|\*\*|`[^`]+`|\[[^]]+\]\(',text):
        return text
    return text.replace('\n','  \n')


class MarkdownPanel(QTextBrowser):
    """独立文档窗口只使用这一处滚动，不显示 Markdown 源码。"""
    def __init__(self, colors, scale=1.0, parent=None):
        super().__init__(parent)
        self.colors,self.scale,self.source=colors,scale,''
        self.setDocument(MarkdownDocument(self))
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self.open_link)
        self.setObjectName('markdownPanel')
    def open_link(self,url):
        if url.scheme().lower() in ('http','https','mailto'):
            QDesktopServices.openUrl(url)
    def set_markdown(self,text):
        self.source=str(text)
        position=self.verticalScrollBar().value()
        render_markdown(self.document(),readable_markdown(self.source),self.colors,self.scale)
        self.verticalScrollBar().setValue(position)


class MarkdownEditor(QTextEdit):
    """技能的所见即所得编辑器，保存时仍使用标准 Markdown。"""
    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.colors=colors
        self.setDocument(MarkdownDocument(self))
        self.setObjectName('skillBody')
    def setPlainText(self,text):
        # 使用编辑器入口同步其光标/字符格式状态，不能只替换底层文档。
        QTextEdit.setMarkdown(self,str(text))
    def markdown(self):
        # Qt5 在部分主题下导出整段强调时会丢失标记；显式保留有效字符格式。
        import uuid
        document=self.document().clone()
        spans=[]; replacements={}; prefix='ZXFMT'+uuid.uuid4().hex[:8]
        block=self.document().begin()
        while block.isValid():
            fragment=block.begin()
            while not fragment.atEnd():
                run=fragment.fragment()
                if run.isValid() and not block.blockFormat().headingLevel() and not block.blockFormat().hasProperty(QTextFormat.BlockCodeFence):
                    cursor=QTextCursor(self.document());cursor.setPosition(run.position())
                    cursor.setPosition(run.position()+run.length(),QTextCursor.KeepAnchor)
                    fmt=cursor.charFormat()
                    if not fmt.fontFixedPitch():
                        markers=tuple(mark for flag,mark in ((fmt.fontWeight()>=QFont.Bold,'**'),(fmt.fontItalic(),'*'),(fmt.fontStrikeOut(),'~~')) if flag)
                        if markers:
                            if spans and spans[-1][0]+spans[-1][1]==run.position() and spans[-1][2]==markers:
                                previous=spans.pop();spans.append((previous[0],previous[1]+run.length(),markers))
                            else: spans.append((run.position(),run.length(),markers))
                fragment+=1
            block=block.next()
        for i,(position,length,markers) in enumerate(reversed(spans)):
            begin=f'{prefix}{i}B';end=f'{prefix}{i}E'
            replacements[begin]=''.join(markers);replacements[end]=''.join(reversed(markers))
            cursor=QTextCursor(document);cursor.setPosition(position);cursor.setPosition(position+length,QTextCursor.KeepAnchor)
            fmt=QTextCharFormat();fmt.setFontWeight(QFont.Normal);fmt.setFontItalic(False);fmt.setFontStrikeOut(False)
            cursor.mergeCharFormat(fmt)
            cursor.setPosition(position+length);cursor.insertText(end,fmt)
            cursor.setPosition(position);cursor.insertText(begin,fmt)
        result=document.toMarkdown()
        for token,marker in replacements.items(): result=result.replace(token,marker)
        return result
    def insertFromMimeData(self,source):
        if source.hasText():
            document=MarkdownDocument()
            render_markdown(document,source.text(),self.colors)
            self.textCursor().insertHtml(document.toHtml())
        else:
            super().insertFromMimeData(source)
    def format_heading(self):
        cursor=self.textCursor(); fmt=cursor.blockFormat(); fmt.setHeadingLevel(2)
        cursor.setBlockFormat(fmt)
        char=QTextCharFormat(); char.setFontWeight(QFont.Bold); char.setFontPointSize(16)
        cursor.mergeCharFormat(char)
    def format_bold(self):
        fmt=QTextCharFormat(); fmt.setFontWeight(QFont.Normal if self.fontWeight()==QFont.Bold else QFont.Bold)
        self.mergeCurrentCharFormat(fmt)
    def format_list(self):
        fmt=QTextListFormat(); fmt.setStyle(QTextListFormat.ListDisc); self.textCursor().createList(fmt)


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
