"""可选真实模型验收：只完成一个已拆分的小任务，不发送用户源码。"""
import json
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from core.agent import Agent
from core.config import Config
from core.llm_client import LLMClient
from core.workspace import Workspace


def main():
    cfg=Config.load()
    cfg.protocol='native'
    cfg.temperature=.2
    cfg.web_enabled=cfg.skills_enabled=cfg.allow_shell=False
    client=LLMClient(cfg)
    client.RETRY_ON_CONN=0
    start=time.monotonic()
    counts={}
    error=''
    with tempfile.TemporaryDirectory(prefix='pyagent_real_plan_') as tmp:
        ws=Workspace(tmp)
        for i in range(4):
            ws.write_file(f'module{i}.c',f'/* Module {i}: bounded counter, caller owns state; returns unchanged value at limit. */\nint step{i}(int value, int limit) {{ return value < limit ? value + 1 : value; }}\n')
        agent=Agent(client,ws)
        def stopped():
            return time.monotonic()-start>420 or (agent._plan is not None and agent._plan.data['steps'][0]['status']=='completed')
        for event in agent.run('读取工程四个模块并编写详细设计方案 report.docx。现在先按已保存的计划完成第一个小任务；笔记应具体说明职责、接口输入输出、数据、流程、边界和依据。用 update_task 提交本步骤验收。',should_stop=stopped):
            if event['type']=='action':
                name=event.get('name') or event.get('tool')
                counts[name]=counts.get(name,0)+1
                print('ACTION',name,flush=True)
            if event['type']=='error': error=event.get('content','')
        step=agent._plan.data['steps'][0]
        passed=step['status']=='completed' and bool(step['proofs'])
        result=dict(scope='first_subtask_only',status='PASS' if passed else 'FAIL',
                    seconds=round(time.monotonic()-start,1),counts=counts,
                    step=step['title'],step_status=step['status'],proof_count=len(step['proofs']),
                    note_characters=len(ws.read_file(step['artifact'])) if ws.resolve(step['artifact']).exists() else 0,error=error)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        path=Path(__file__).resolve().parent.parent/'build'/'real-plan-step.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return passed


if __name__=='__main__':
    sys.exit(0 if main() else 1)
