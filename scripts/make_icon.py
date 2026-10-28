#!/usr/bin/env python3
"""生成正式程序图标 icons/app.png（256×256，透明圆角 + 蓝渐变 + 白色 A）

用 PyQt5 的 QPainter 绘制（已在 requirements.txt 中，不引额外依赖）：
    python scripts/make_icon.py
生成结果与工程内 icons/app.png 同规格，可复现。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF, QRectF, Qt          # noqa: E402
from PyQt5.QtGui import (QBrush, QColor, QIcon,       # noqa: E402
                         QLinearGradient, QPainter, QPainterPath, QPixmap)
from PyQt5.QtWidgets import QApplication                # noqa: E402

SIZE = 256
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "icons", "app.png")


def draw(size=SIZE):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = size / 256.0                       # 以 256 为设计基准等比缩放

    # ---- 圆角底 + 深蓝渐变 ----
    path = QPainterPath()
    path.addRoundedRect(QRectF(4 * s, 4 * s, 248 * s, 248 * s), 62 * s, 62 * s)
    grad = QLinearGradient(QPointF(0, 0), QPointF(size, size))
    grad.setColorAt(0.0, QColor("#2E4E93"))
    grad.setColorAt(1.0, QColor("#15254C"))
    p.fillPath(path, QBrush(grad))

    # ---- 几何字形 A（两条斜腿 + 横杠 + 两个节点），不依赖系统字体 ----
    white = QColor("#FFFFFF")
    p.setPen(Qt.NoPen)
    p.setBrush(white)
    left_leg = QPainterPath()
    left_leg.addPolygon(_poly([(128, 34), (141, 58), (74, 210), (47, 210)], s))
    right_leg = QPainterPath()
    right_leg.addPolygon(_poly([(128, 34), (209, 210), (182, 210), (115, 58)], s))
    p.drawPath(left_leg)
    p.drawPath(right_leg)
    # 横杠
    p.drawRect(QRectF(72 * s, 150 * s, 112 * s, 16 * s))
    # 交点节点（用底色圆点，像工程图上的铆钉）
    p.setBrush(QColor("#1B2E5C"))
    p.drawEllipse(QPointF(90 * s, 158 * s), 5.5 * s, 5.5 * s)
    p.drawEllipse(QPointF(166 * s, 158 * s), 5.5 * s, 5.5 * s)
    p.end()
    return pm


def _poly(points, s):
    from PyQt5.QtGui import QPolygonF
    return QPolygonF([QPointF(x * s, y * s) for x, y in points])


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pm = draw()
    if not pm.save(OUT, "PNG"):
        print(f"写入失败: {OUT}", file=sys.stderr)
        return 1
    print(f"图标已生成: {OUT} ({SIZE}×{SIZE})")
    # 自检：能被 QIcon 加载
    if QIcon(OUT).isNull():
        print("自检失败：QIcon 无法加载生成的 PNG", file=sys.stderr)
        return 1
    print("自检通过：QIcon 加载正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
