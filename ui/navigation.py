"""会话列表使用标题与元信息分层，避免文件树和历史混在同一列表。"""
from PyQt5.QtCore import Qt, QSize, QRectF
from PyQt5.QtGui import QColor, QFont, QPen
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle


class SessionDelegate(QStyledItemDelegate):
    def __init__(self, owner, parent):
        super().__init__(parent)
        self.owner=owner
    def sizeHint(self,option,index):
        kind=(index.sibling(index.row(),0).data(Qt.UserRole) or ('',))[0]
        return QSize(200,self.owner._px(62 if kind=='session' else 30))
    def paint(self,painter,option,index):
        kind=(index.sibling(index.row(),0).data(Qt.UserRole) or ('',))[0]
        if index.column()==1 and kind=='session':
            return
        if kind!='session':
            return super().paint(painter,option,index)
        painter.save()
        c=self.owner._c; r=option.rect.adjusted(0,2,-2,-2)
        if option.state & (QStyle.State_Selected|QStyle.State_MouseOver):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(c['avatar_bg'] if option.state & QStyle.State_Selected else c['hover']))
            painter.drawRoundedRect(QRectF(r),8,8)
        title=QFont(option.font); title.setPixelSize(self.owner._px(13)); title.setWeight(QFont.DemiBold)
        painter.setFont(title); painter.setPen(QColor(c['text']))
        text=painter.fontMetrics().elidedText(str(index.data() or ''),Qt.ElideRight,r.width()-18)
        painter.drawText(r.adjusted(self.owner._px(10),self.owner._px(7),-self.owner._px(8),-self.owner._px(31)),Qt.AlignLeft|Qt.AlignVCenter,text)
        meta=QFont(option.font); meta.setPixelSize(self.owner._px(10)); painter.setFont(meta)
        painter.setPen(QColor(c['text3']))
        text=painter.fontMetrics().elidedText(str(index.data(Qt.UserRole+1) or ''),Qt.ElideRight,r.width()-18)
        painter.drawText(r.adjusted(self.owner._px(10),self.owner._px(32),-self.owner._px(8),-self.owner._px(8)),Qt.AlignLeft|Qt.AlignVCenter,text)
        painter.restore()
