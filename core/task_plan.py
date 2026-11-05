"""工作区内可恢复的串行任务计划。计划状态只能由实际文件和执行记录推进。"""
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from core.workspace import WorkspaceError

PLAN_TOOLS = {'set_plan', 'update_task', 'task_status', 'assemble_document'}
SOURCE_EXT = {'.c', '.h', '.cpp', '.hpp', '.py', '.js', '.ts', '.java', '.go', '.rs'}


def tool_specs():
    def spec(name, description, properties, required=()):
        return {'type': 'function', 'function': {'name': name, 'description': description,
                'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}}
    return [
        spec('set_plan', '开始前细化小任务计划。按依赖顺序执行，每步包含标题、类别和验收标准；已有完成项不能丢弃。', {
            'steps': {'type': 'array', 'items': {'type': 'object', 'properties': {
                'id': {'type': 'string'}, 'title': {'type': 'string'},
                'kind': {'type': 'string', 'enum': ['analyze', 'implement', 'assemble', 'verify']},
                'criteria': {'type': 'string'}, 'source': {'type': 'string'}},
                'required': ['title', 'kind', 'criteria']}}}, ['steps']),
        spec('update_task', '完成当前小任务或记录失败。完成须有实际笔记/文件/执行证据，不能跳步。', {
            'id': {'type': 'string'}, 'status': {'type': 'string', 'enum': ['completed', 'failed', 'in_progress']},
            'summary': {'type': 'string'}}, ['id', 'status']),
        spec('task_status', '读取已保存的小任务计划、当前步骤、验收条件与临时目录。', {}),
        spec('assemble_document', '全部分析小任务完成后，把已验证的 Markdown 笔记按顺序汇总成目标 DOCX，无需重新生成整篇参数。', {}),
    ]


def inventory(ws):
    files = []
    ignored = {'.agent', '.agents', '.pyagent', '.git', '.venv', 'venv', 'node_modules', 'build', 'dist', '__pycache__', 'utf8_tmp'}
    # os.walk 可剪枝，避免为了规划扫描依赖目录或自身临时产物。
    for root, dirs, names in os.walk(ws.root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ignored and not Path(root, d).is_symlink())
        for name in sorted(names):
            path = Path(root, name)
            if path.suffix.lower() in SOURCE_EXT and not path.is_symlink():
                files.append((path.relative_to(ws.root).as_posix(), path.stat().st_size))
    return files


class TaskPlan:
    def __init__(self, ws, data):
        self.ws, self.data = ws, data
        self.directory = ws.resolve('.agent/tasks/' + data['id'])

    @classmethod
    def start(cls, ws, goal, document=False, resume_id=None):
        if resume_id and re.fullmatch(r'[a-f0-9]{16}', str(resume_id)):
            path = ws.resolve('.agent/tasks/' + resume_id + '/plan.json')
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
                if data.get('id') != resume_id:
                    raise WorkspaceError('任务计划编号不匹配')
                obj = cls(ws, data)
                obj.revalidate()
                obj.save()
                return obj
        sources = inventory(ws)
        brief=bool(re.search(r'简短|极简|一页|概览|\d{1,3}字以内',goal)) and not re.search(r'详细|全面|完整',goal)
        strict = ((document and len(sources) >= 4 and not brief)
                  or bool(re.search(r'复杂|大型|多模块|完整项目|详细拆分|(?:开发|构建|实现).{0,16}(?:系统|平台|网站|应用)', goal))
                  or (len(sources)>=4 and bool(re.search(r'修复|重构|开发|新增',goal))))
        task_id = uuid.uuid4().hex[:16]
        quoted=re.search(r'[`"“]([^`"”\n]+\.docx)[`"”]',goal,re.I)
        target = quoted or re.search(r'((?:[A-Za-z]:)?[\w\u4e00-\u9fff./\\-]+\.docx)', goal, re.I)
        final_path = target.group(1) if target else '设计方案.docx'
        if not quoted and not re.match(r'[A-Za-z]:|[./\\]',final_path):
            final_path=re.sub(r'^.*?(?:命名为|保存为|名为|叫|输出为|(?:生成|编写|撰写|创建|写)(?:一个|一份)?)','',final_path)
        # 原任务路径来自用户；仍须在工作区内。
        if document:
            final_path=ws.resolve(final_path).relative_to(ws.root).as_posix()
        data = dict(schema=1, id=task_id, goal=goal, strict=strict, document=document,
                    target=final_path, steps=[], receipts=[], source_count=len(sources), completed=False,
                    needs_refinement=strict and not document, inventory=sources)
        obj = cls(ws, data)
        if strict and document:
            seeds = [('总体架构与源码清单', 'analyze', '', 700)]
            for path, size in sources:
                if size>=80000:
                    seeds.extend([(f'{path} 接口与核心数据结构','analyze',path,700),
                                  (f'{path} 执行流程与关键分支','analyze',path,900),
                                  (f'{path} 异常边界与验证依据','analyze',path,700)])
                else:
                    seeds.append((f'分析模块 {path}', 'analyze', path, 500))
            seeds += [('跨模块调用 数据流 状态切换 异常与验证策略', 'analyze', '', 900),
                      ('汇总完整设计文档', 'assemble', '', 0), ('检查覆盖范围和最终文件', 'verify', '', 0)]
        else:
            seeds = [('明确需求和验收条件', 'analyze', '', 250),
                     ('实现并保存产物', 'implement', '', 0), ('实际检查和验证', 'verify', '', 0)]
        for i, (title, kind, source, minimum) in enumerate(seeds, 1):
            data['steps'].append(obj.make_step(f's{i:03}', title, kind, source, minimum))
        obj.save()
        return obj

    def make_step(self, step_id, title, kind, source='', minimum=300, criteria=''):
        return dict(id=step_id, title=title, kind=kind, source=source, minimum=minimum,
                    artifact=f'.agent/tasks/{self.data["id"]}/notes/{step_id}.md' if kind=='analyze' else '',
                    criteria=criteria or ('职责、接口输入输出、核心数据、执行流程、边界异常、源码位置；不确定项明确标注' if kind=='analyze'
                                          else '依据真实产物和执行结果验收'), status='pending', proofs={}, reads=[], writes=[], runs=[])

    def current(self):
        return next((s for s in self.data['steps'] if s['status']!='completed'), None)

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ws.resolve('.agent/tasks/' + self.data['id'])
        self.ws.resolve('.agent/tasks/'+self.data['id']+'/tmp').mkdir(exist_ok=True)
        if os.name == 'nt':
            import ctypes
            folder = str(self.ws.resolve('.agent'))
            flags = ctypes.windll.kernel32.GetFileAttributesW(folder)
            if flags != -1:
                ctypes.windll.kernel32.SetFileAttributesW(folder, flags | 2)
        current = self.current()
        if current and current['status']=='pending':
            current['status']='in_progress'
        self.data['completed'] = current is None
        path = self.directory/'plan.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.data,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(path)
        (self.directory/'progress.md').write_text(self.markdown(),encoding='utf-8')

    def markdown(self):
        def cell(value): return str(value).replace('|','\\|').replace('\n',' ')
        statuses={'pending':'待处理','in_progress':'进行中','completed':'已完成','failed':'需修复'}
        done=sum(s['status']=='completed' for s in self.data['steps'])
        lines=['# 任务计划',self.data['goal'],f'**进度：{done} / {len(self.data["steps"])}**',
               '| 编号 | 状态 | 子任务 |','| --- | --- | --- |']
        for step in self.data['steps']:
            lines.append(f'| {cell(step["id"])} | {statuses.get(step["status"],step["status"])} | {cell(step["title"])} |')
        current=self.current()
        if current:
            lines+=['','## 当前步骤',current['title'],'','### 验收要求',current['criteria']]
            if current['artifact']: lines+=['','**分析笔记**：`'+current['artifact']+'`']
            if current.get('error'): lines+=['','### 待解决的问题',current['error']]
        lines+=['','## 任务资料','临时目录：`.agent/tasks/'+self.data['id']+'/tmp/`']
        if self.data['document']: lines+=['','交付文件：`'+self.data['target']+'`']
        return '\n\n'.join(lines[:3])+ '\n\n'+'\n'.join(lines[3:])

    def describe(self):
        lines = [f'任务 {self.data["id"]}', self.data['goal'][:2000],
                 f'进度：{sum(s["status"]=="completed" for s in self.data["steps"])}/{len(self.data["steps"])}',
                 f'临时目录：.agent/tasks/{self.data["id"]}/tmp/']
        if self.data['document']: lines.append('最终交付：'+self.data['target'])
        for step in self.data['steps']:
            lines.append(f'{step["id"]} [{step["status"]}] {step["title"]}')
        current = self.current()
        if current:
            lines += ['当前只处理：'+current['id']+' '+current['title'], '验收：'+current['criteria']]
            if current['artifact']:
                lines += [f'笔记保存到：{current["artifact"]}，至少 {current["minimum"]} 字符；需要源码依据，禁止用重复段落凑字数。']
            if current.get('error'):
                lines.append('最近失败：'+current['error'][:500])
        return '\n'.join(lines)

    def context(self):
        description=self.describe()
        if len(description)>5000:
            current=self.current()
            description=description[:2000]+'\n当前小任务：'+json.dumps(current,ensure_ascii=False)[:2200]
        extra=''
        if self.data.get('needs_refinement'):
            extra=('开始前必须先调用 set_plan，把实现工作拆分成可独立验证的小任务。自动草案不是详细实现计划。'
                   '已发现文件：'+str(self.data.get('inventory',[]))[:2500])
        if self.data['completed']:
            extra+='所有小任务已通过文件或执行证据验收。最终仅简短告知交付位置和验证范围，不要重复全文。'
        return ('先按已保存的计划逐项处理，不要重建已完成项。\n'+description+'\n'+extra
                +'\n所有临时脚本、提取结果和草稿必须写入当前任务的 .agent 目录。最终交付文件按用户指定位置保存。'
                +'\n可用 set_plan 在开始前细化计划，update_task 完成当前步骤后再进入下一步。'
                +'set_plan 参数 steps=[{title,kind,criteria,source可选}]；update_task 参数 id,status=completed|failed|in_progress,summary；'
                +'task_status 和 assemble_document 使用空参数对象。'
                +'分析笔记使用标题分节，说明职责、接口、数据、流程、异常、依据；不要只写模块名称。')

    def allowed_tools(self):
        if not self.data['strict'] or not self.data['document']:
            return None
        current=self.current()
        if not current:
            return {'task_status'}
        if current['kind']=='assemble':
            return {'assemble_document','task_status','read_file','update_task'}
        if current['kind']=='verify':
            return {'read_file','task_status','update_task','write_file','assemble_document'}
        allowed={'list_files','read_file','search_files','write_file','append_file','task_status','update_task'}
        if not self.data['receipts']:
            allowed.update(('set_plan','load_skill'))
        return allowed

    def fingerprint(self, rel):
        path = self.ws.resolve(rel, must_exist=True)
        if not path.is_file():
            raise WorkspaceError('证据必须是文件')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def revalidate(self):
        invalid = False
        for step in self.data['steps']:
            if step['status']=='completed' and not invalid:
                if self.data['strict'] and step['kind'] in ('analyze','implement','assemble') and not step.get('proofs'):
                    invalid=True
                for rel, digest in step.get('proofs',{}).items():
                    try: valid = self.fingerprint(rel)==digest
                    except (OSError,WorkspaceError): valid=False
                    if not valid:
                        invalid=True
                        break
            if invalid:
                step['status']='pending'
                step['reads'],step['writes'],step['runs'],step['proofs']=[],[],[],{}
                step['error']='此前证据已变更或丢失，需要重新检查此步及后续步骤。'
        self.data['completed']=self.current() is None

    def guard(self, name, arguments):
        if name in ('write_file','append_file','delete_file'):
            target=self.ws.resolve(str(arguments.get('path') or ''))
            relative=target.relative_to(self.ws.root).as_posix()
            if relative.startswith('.agent/tasks/'):
                if not target.is_relative_to(self.directory) or target.name in ('plan.json','progress.md'):
                    return '错误: 任务状态由计划工具维护；不能直接覆盖计划文件或其他任务的资料。'
        if not self.data['strict']:
            return ''
        step = self.current()
        if not step:
            return ''
        if self.data.get('needs_refinement') and name not in ('list_files','load_skill') and not name.startswith('skill_'):
            return '错误: 复杂任务须先 set_plan 细化小任务和验收条件，再逐项执行。\n'+self.context()
        if self.data['document'] and name in ('write_file','append_file'):
            path = self.ws.resolve(str(arguments.get('path') or ''))
            internal = self.ws.resolve('.agent')
            if path.is_relative_to(internal):
                return ''
            if path == self.ws.resolve(self.data['target']) and step['kind'] in ('assemble','verify'):
                return ''
            return '错误: 当前先完成小任务 '+step['id']+'。中间产物必须保存在 .agent 下；最终文档请在汇总步骤生成。\n'+self.describe()
        return ''

    def observe(self, name, arguments, result, failed=False):
        step = self.current()
        if not step:
            return
        if failed:
            step['error']=result[-500:]
        else:
            path = str(arguments.get('path') or '').replace('\\','/')
            if path:
                try: path=self.ws.resolve(path).relative_to(self.ws.root).as_posix()
                except (WorkspaceError,ValueError): pass
            if name in ('read_file','search_files') and path and path not in step['reads']:
                step['reads'].append(path)
            if name in ('write_file','append_file') and path and path not in step['writes']:
                step['writes'].append(path)
            if name in ('run_python','run_command','tcp_echo_test'):
                step['runs'].append(result[-1000:])
            if not self.data['strict']:
                # 简单任务仍保存步骤，但不要求额外的计划工具往返。
                if name in ('write_file','append_file'):
                    self.data['steps'][0]['status']='completed'
                    self.data['steps'][1]['status']='completed'
                if name in ('run_python','run_command','tcp_echo_test') or (name=='read_file' and path.endswith('.docx')):
                    if self.data['steps'][1]['status']=='completed':
                        self.data['steps'][2]['status']='completed'
        self.data['receipts'].append(dict(step=step['id'],tool=name,path=str(arguments.get('path') or ''),failed=failed,result=result[-400:]))
        del self.data['receipts'][:-100]
        self.save()

    def complete_step(self, step):
        kind=step['kind']
        proofs={}
        if kind=='analyze':
            rel=step['artifact']
            text=self.ws.read_file(rel)
            if len(text.strip()) < step['minimum']:
                raise WorkspaceError(f'分析笔记过短，至少 {step["minimum"]} 字符，并应覆盖接口、流程和异常依据')
            if len(set(text.splitlines())) < 5:
                raise WorkspaceError('分析笔记必须分节说明，不能用重复段落凑字数')
            topics=['职责','功能','接口','输入','输出','数据','结构','流程','算法','异常','边界','依据','来源']
            if sum(topic in text for topic in topics)<5:
                raise WorkspaceError('分析笔记须说明职责/接口/数据/流程/异常/依据等内容，不能只列标题或模块名称')
            if rel not in step['writes']:
                raise WorkspaceError('本步骤尚未实际写入指定分析笔记')
            if step['source']:
                source=step['source']
                if source not in step['reads'] or Path(source).name not in text:
                    raise WorkspaceError('必须实际读取目标源码，并在笔记中引用该源码的位置')
                proofs[source]=self.fingerprint(source)
            elif self.data['source_count'] and not step['reads']:
                raise WorkspaceError('此分析步骤尚无实际读取或搜索源码的证据')
            for source in step['reads']:
                if source and not source.startswith('.agent/'):
                    path=self.ws.resolve(source)
                    if path.is_file(): proofs[source]=self.fingerprint(source)
            proofs[rel]=self.fingerprint(rel)
        elif kind=='implement':
            for path in step['writes']:
                proofs[path]=self.fingerprint(path)
            if not proofs:
                raise WorkspaceError('尚未实际保存实现产物')
        elif kind in ('assemble','verify') and self.data['document']:
            from core.word_documents import inspect_docx
            target=self.data['target']
            info=inspect_docx(self.ws.resolve(target,must_exist=True))
            if info['characters'] < self.minimum_document():
                raise WorkspaceError('文档正文不足以覆盖计划中的模块分析，需汇总所有已完成笔记')
            if kind=='verify' and target not in step['reads']:
                raise WorkspaceError('验证步骤须实际 read_file 检查最终 DOCX 正文与覆盖范围')
            for item in self.data['steps']:
                if item['source'] and Path(item['source']).name not in info['text']:
                    raise WorkspaceError('文档缺少模块依据：'+item['source'])
            proofs[target]=self.fingerprint(target)
        elif kind=='verify' and not step['runs']:
            raise WorkspaceError('尚无真实执行成功记录')
        step['proofs']=proofs
        step['status']='completed'
        step.pop('error',None)

    def minimum_document(self):
        return max(120, sum(s['minimum'] for s in self.data['steps'] if s['kind']=='analyze'))

    def execute(self, name, arguments):
        try:
            self.revalidate()
            if name=='set_plan':
                if self.data.get('customized') or self.data['receipts']:
                    raise WorkspaceError('计划已开始执行，不要重建；请继续当前未完成小任务')
                steps=arguments.get('steps')
                if not isinstance(steps,list) or not 3<=len(steps)<=240:
                    raise WorkspaceError('计划必须包含3到240个小任务')
                completed=[s for s in self.data['steps'] if s['status']=='completed']
                if completed:
                    raise WorkspaceError('已有完成步骤，不能重建计划；请继续当前步骤')
                if self.data['strict'] and self.data['document']:
                    # 保留自动发现的模块覆盖约束，允许补充细化而不能删掉模块。
                    sources={s['source'] for s in self.data['steps'] if s['source']}
                    if not sources.issubset({s.get('source') for s in steps if s.get('kind')=='analyze'}):
                        raise WorkspaceError('细化计划不能遗漏已发现的源文件模块')
                    for source in sources:
                        required=sum(s['source']==source for s in self.data['steps'])
                        if sum(s.get('source')==source and s.get('kind')=='analyze' for s in steps)<required:
                            raise WorkspaceError('大型模块的接口、流程、异常分析不能被合并成一个概述：'+source)
                    global_count=sum(s['kind']=='analyze' and not s['source'] for s in self.data['steps'])
                    if sum(s.get('kind')=='analyze' and not s.get('source') for s in steps)<global_count:
                        raise WorkspaceError('总体架构和跨模块流程不能在细化计划时被遗漏')
                proposed=[]
                identifiers=set()
                global_minimum=[s['minimum'] for s in self.data['steps'] if s['kind']=='analyze' and not s['source']]
                global_index=0
                for i,s in enumerate(steps,1):
                    kind=s.get('kind')
                    if kind not in ('analyze','implement','assemble','verify') or not str(s.get('title') or '').strip() or not s.get('criteria'):
                        raise WorkspaceError('每步必须有标题、合法类别和明确验收条件')
                    source=str(s.get('source') or '')
                    if source: self.ws.resolve(source,must_exist=True)
                    minimum=max([s['minimum'] for s in self.data['steps'] if source and s['source']==source] or [500 if source else 300])
                    if kind=='analyze' and not source:
                        if global_index<len(global_minimum): minimum=max(minimum,global_minimum[global_index])
                        global_index+=1
                    step_id=str(s.get('id') or f's{i:03}')
                    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',step_id) or step_id in identifiers:
                        raise WorkspaceError('小任务编号必须唯一，且只能包含字母、数字、下划线或连字符')
                    identifiers.add(step_id)
                    proposed.append(self.make_step(step_id,str(s['title'])[:160],kind,source,minimum,str(s['criteria'])[:800]))
                if proposed[-1]['kind']!='verify' or (self.data['document'] and not any(s['kind']=='assemble' for s in proposed)):
                    raise WorkspaceError('最后一步必须验收，文档任务还须有汇总步骤')
                if self.data['document']:
                    assemble=next(i for i,s in enumerate(proposed) if s['kind']=='assemble')
                    if any(s['kind']=='analyze' for s in proposed[assemble+1:]):
                        raise WorkspaceError('所有分析步骤必须在汇总文档之前完成')
                self.data['steps']=proposed
                self.data['customized']=True
                self.data['needs_refinement']=False
            elif name=='update_task':
                step=self.current()
                if not step or step['id']!=arguments.get('id'):
                    raise WorkspaceError('只能更新当前未完成小任务，不能跳过依赖或重置已完成项')
                status=arguments.get('status')
                if status=='completed': self.complete_step(step)
                elif status in ('failed','in_progress'):
                    step['status']=status
                    step['error']=str(arguments.get('summary') or '')[:800]
                else: raise WorkspaceError('不支持的任务状态')
            elif name=='assemble_document':
                step=self.current()
                if not self.data['document'] or not step or step['kind']!='assemble':
                    raise WorkspaceError('必须先完成分析小任务，再进入文档汇总步骤')
                content=[]
                for item in self.data['steps']:
                    if item['kind']=='analyze':
                        if item['status']!='completed': raise WorkspaceError('仍有分析任务未完成')
                        content.append(self.ws.read_file(item['artifact']))
                self.ws.write_file(self.data['target'],'\n\n'.join(content))
                self.complete_step(step)
            elif name!='task_status':
                raise WorkspaceError('未知计划工具')
            self.save()
            return self.describe()
        except (OSError,ValueError,KeyError,TypeError,WorkspaceError) as exc:
            try: self.save()
            except OSError: pass
            return '错误: '+str(exc)

    def completion_issue(self):
        if not self.data['strict']:
            return ''
        self.revalidate()
        self.save()
        step=self.current()
        return ('计划验收未通过：'+self.describe()) if step else ''
