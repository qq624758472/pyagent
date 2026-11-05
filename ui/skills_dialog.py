"""技能选择与 Markdown 编写。用户副本保存到工作区 .agent/skills。"""
import json
import re
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog,QHBoxLayout,QVBoxLayout,QListWidget,QPushButton,
                             QLineEdit,QPlainTextEdit,QLabel,QMessageBox,QFormLayout)
from core.skills import discover, parse_frontmatter
from core.runtime import DATA_ROOT
from ui.markdown_document import MarkdownEditor


class SkillsDialog(QDialog):
    def __init__(self, owner, edit=False):
        super().__init__(owner)
        self.owner=owner
        self.setWindowTitle('编写技能' if edit else '加载技能')
        self.resize(900,620)
        layout=QHBoxLayout(self)
        left=QVBoxLayout()
        self.list=QListWidget(self)
        self.list.setObjectName('skillList')
        self.list.setMinimumWidth(230)
        self.list.currentRowChanged.connect(self.select)
        left.addWidget(self.list)
        self.b_refresh=QPushButton('刷新技能',self)
        self.b_refresh.clicked.connect(self.refresh)
        left.addWidget(self.b_refresh)
        self.b_new=QPushButton('新建技能',self)
        self.b_new.clicked.connect(self.new_skill)
        left.addWidget(self.b_new)
        layout.addLayout(left,1)
        right=QVBoxLayout()
        form=QFormLayout()
        self.name=QLineEdit(self)
        self.name.setPlaceholderText('英文名称，例如 acs-design-writer')
        self.description=QLineEdit(self)
        form.addRow('技能名称',self.name)
        form.addRow('用途说明',self.description)
        right.addLayout(form)
        toolbar=QHBoxLayout()
        self.body=MarkdownEditor(owner._c,self)
        for title,fn in [('标题',self.body.format_heading),('加粗',self.body.format_bold),('列表',self.body.format_list)]:
            button=QPushButton(title,self); button.setObjectName('taskAction'); button.clicked.connect(fn); toolbar.addWidget(button)
        toolbar.addStretch(1)
        right.addLayout(toolbar)
        self.body.setPlaceholderText('编写技能的适用场景、执行步骤和验收条件…')
        right.addWidget(self.body,1)
        self.location=QLabel(self)
        self.location.setWordWrap(True)
        self.location.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right.addWidget(self.location)
        row=QHBoxLayout()
        self.b_save=QPushButton('保存到工作区',self)
        self.b_save.clicked.connect(self.save_skill)
        self.b_load=QPushButton('加载到当前会话',self)
        self.b_load.clicked.connect(self.load_skill)
        row.addWidget(self.b_save)
        row.addWidget(self.b_load)
        for button in (self.b_refresh,self.b_new,self.b_save,self.b_load):
            button.setObjectName('taskAction')
            button.setCursor(Qt.PointingHandCursor)
        right.addLayout(row)
        layout.addLayout(right,3)
        self.refresh()
        if edit: self.new_skill()

    def root(self):
        ws=self.owner.agent.workspace
        if ws:
            return ws.resolve('.agent/skills')
        return Path(DATA_ROOT)/'skills'

    def refresh(self):
        self.list.blockSignals(True)
        # 管理界面始终允许查看技能，即使自动加载开关关闭。
        self.skills=discover(None,self.owner.agent.workspace)
        self.list.clear()
        for skill in self.skills:
            self.list.addItem(skill.name+' · '+skill.source)
        self.list.blockSignals(False)
        if self.skills:
            self.list.setCurrentRow(0)
            self.select(0)

    def select(self,index):
        if not 0<=index<len(self.skills): return
        skill=self.skills[index]
        self.name.setText(skill.name)
        self.description.setText(skill.description)
        self.body.setPlainText(skill.body)
        self.location.setText(str(skill.path)+'\n保存会创建工作区副本；加载使用工作区版本。')

    def new_skill(self):
        self.list.setCurrentRow(-1)
        self.name.clear()
        self.description.clear()
        self.body.setPlainText('## 适用场景\n\n## 执行步骤\n\n1. 检查当前任务与已有产物。\n2. 按小任务执行并保存进度。\n3. 根据实际结果修复并重新验证。\n\n## 验收条件\n\n')
        self.location.setText('保存位置：'+str(self.root()))
        self.name.setFocus()

    def write_skill(self):
        name=self.name.text().strip()
        description=self.description.text().strip()
        body=self.body.markdown().strip()
        if not re.fullmatch(r'[a-z][a-z0-9-]{0,63}',name):
            raise ValueError('名称请使用小写英文字母、数字和连字符，且以字母开头。')
        if not description or len(body)<30:
            raise ValueError('请填写用途说明和至少30字符的执行规范。')
        root=self.root()
        target=root/name/'SKILL.md'
        ws=self.owner.agent.workspace
        if ws: target=ws.resolve(str(target))
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('技能路径越界')
        header='name: '+name+'\ndescription: '+json.dumps(description,ensure_ascii=False)
        row=self.list.currentRow()
        if 0<=row<len(self.skills) and self.skills[row].name==name:
            original=self.skills[row].path.read_text(encoding='utf-8')
            match=re.match(r'\A---\s*\n(.*?)\n---',original,re.S)
            if match:
                header=re.sub(r'(?m)^name:.*$', 'name: '+name, match.group(1))
                header=re.sub(r'(?m)^description:.*$',lambda _: 'description: '+json.dumps(description,ensure_ascii=False),header)
                if not re.search(r'(?m)^description:',header): header+='\ndescription: '+json.dumps(description,ensure_ascii=False)
        text='---\n'+header+'\n---\n\n'+body+'\n'
        meta,parsed=parse_frontmatter(text)
        if meta.get('name')!=name or not parsed.strip():
            raise ValueError('技能格式校验未通过')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.md.tmp')
        temporary.write_text(text,encoding='utf-8')
        temporary.replace(target)
        self.location.setText('已保存：'+str(target))
        return name,target

    def save_skill(self):
        try:
            name,_=self.write_skill()
            self.refresh()
            for i,s in enumerate(self.skills):
                if s.name==name:
                    self.list.setCurrentRow(i)
                    break
        except (OSError,ValueError,RuntimeError) as exc:
            QMessageBox.warning(self,'保存失败',str(exc))

    def load_skill(self):
        if self.owner._busy:
            QMessageBox.information(self,'正在执行','请在当前任务停止后加载技能，避免中途改变执行规范。')
            return
        try:
            name=self.name.text().strip()
            # 编写内容先保存，避免界面显示与实际加载的版本不一致。
            row=self.list.currentRow()
            skill=self.skills[row] if 0<=row<len(self.skills) else None
            unchanged=bool(skill and skill.name==name and skill.description==self.description.text().strip()
                           and skill.body.strip()==self.body.markdown().strip())
            if not unchanged:
                self.write_skill()
            if not self.owner.config.skills_enabled:
                self.owner.config.skills_enabled=True
                self.owner.config.save()
            self.owner.agent.select_skill(name)
            self.owner._append_sys('已选择技能「'+name+'」，下次发送任务时自动加载。')
            self.owner._save_current_session()
            self.owner._refresh_status()
            self.accept()
        except (OSError,ValueError,RuntimeError) as exc:
            QMessageBox.warning(self,'加载失败',str(exc))
