"""Markdown HTML compatibility helper; requires QApplication."""
from ui.markdown_document import MarkdownDocument, render_markdown


def md_to_html(md):
    document = MarkdownDocument()
    render_markdown(document, md)
    return document.toHtml()
