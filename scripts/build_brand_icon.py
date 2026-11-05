"""从仓库原生 SVG 生成 Windows EXE 图标，不依赖外部图片服务。"""
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage,QPainter
from PyQt5.QtSvg import QSvgRenderer
root=Path(__file__).resolve().parent.parent
image=QImage(256,256,QImage.Format_ARGB32)
image.fill(Qt.transparent)
painter=QPainter(image)
QSvgRenderer(str(root/'icons/zhixing.svg')).render(painter)
painter.end()
assert image.save(str(root/'icons/zhixing.png'))
