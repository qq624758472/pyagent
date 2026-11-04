"""可选真实模型验收：只发送临时合成源码，不读取用户工程或改配置。"""
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.agent import Agent
from core.config import Config
from core.llm_client import LLMClient
from core.workspace import Workspace
from core.word_documents import inspect_docx


def main():
    cfg = Config.load()
    cfg.skills_enabled = cfg.web_enabled = cfg.allow_shell = False
    cfg.protocol = 'native'
    cfg.temperature = .2
    cfg.llm_read_timeout = 60
    client = LLMClient(cfg)
    client.RETRY_ON_CONN = 0
    started = time.monotonic()
    counts, phases = Counter(), []
    last, error = '', ''
    with tempfile.TemporaryDirectory(prefix='pyagent_real_research_') as tmp:
        ws = Workspace(tmp)
        ws.write_file('counter.h', 'typedef struct { int value; int limit; } Counter;\nvoid counter_init(Counter *c, int limit);\nint counter_step(Counter *c);\n')
        ws.write_file('counter.c', '#include "counter.h"\nvoid counter_init(Counter *c, int limit) { c->value=0; c->limit=limit; }\nint counter_step(Counter *c) { if(c->value >= c->limit) return 0; c->value++; return 1; }\n')
        ws.write_file('main.c', '#include "counter.h"\nint main(void) { Counter c; counter_init(&c, 3); while(counter_step(&c)) {} return c.value == 3 ? 0 : 1; }\n')
        agent = Agent(client, ws)
        for event in agent.run('读取现有源码并生成300至500字的中文设计方案 report.docx，说明Counter结构、counter_init和counter_step接口、main调用关系及边界行为。直接写入真正的Word文件并检查，不要修改源码。',
                               should_stop=lambda: time.monotonic()-started > 360,
                               confirm=lambda *_: False):
            last = event['type']
            if last == 'action':
                counts[event.get('name') or event.get('tool') or 'unknown'] += 1
                print('ACTION', event.get('name') or event.get('tool'), flush=True)
            if last == 'checkpoint': phases.append(event['content'])
            if last == 'error': error = event.get('content','')
        path = Path(tmp)/'report.docx'
        valid = False
        characters = 0
        if path.exists():
            info = inspect_docx(path)
            characters = info['characters']
            valid = characters >= 120 and all(name in info['text'] for name in ('Counter','counter_init','counter_step'))
        result = dict(status='PASS' if last=='answer' and valid else 'FAIL',
                      last_event=last, counts=dict(counts), phases=phases,
                      characters=characters, seconds=round(time.monotonic()-started,1), error=error)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        target = Path(__file__).resolve().parent.parent/'build'/'real-research-eval.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result['status']=='PASS'


if __name__=='__main__':
    sys.exit(0 if main() else 1)
