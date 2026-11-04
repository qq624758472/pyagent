"""可持久化的任务交接记录；只摘录已知事实，不调用模型猜测进度。"""
import json
import re


def is_continue(text):
    return bool(re.fullmatch(
        r"\s*(?:请|帮我)?\s*(?:继续|接着|接着做|继续做|继续执行|继续上次的任务|继续上次任务|resume|continue)[。.!！\s]*",
        text, re.I))


def normalize_messages(messages):
    """兼容旧记录，并把不完整工具组转为普通执行记录，避免非法 API 历史。"""
    source = []
    for item in messages or []:
        if isinstance(item, dict):
            msg = dict(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            msg = dict(role=item[0], content=item[1] or '')
        else:
            continue
        if msg.get('role') in ('user', 'assistant', 'tool'):
            source.append(msg)
    out, i = [], 0
    while i < len(source):
        msg = source[i]
        calls = msg.get('tool_calls') or []
        if msg.get('role') == 'assistant' and calls:
            end = i + 1
            while end < len(source) and source[end].get('role') == 'tool':
                end += 1
            replies = source[i + 1:end]
            ids = [c.get('id') for c in calls]
            if (all(ids) and len(set(ids)) == len(ids)
                    and sorted(ids) == sorted(r.get('tool_call_id') or '' for r in replies)):
                out.extend(source[i:end])
            else:
                out.append(dict(role='user', content='OBSERVATION: 历史工具调用未完整结束，不能视为成功。\n'
                                + json.dumps(source[i:end], ensure_ascii=False)[:3000]))
            i = end
            continue
        if msg.get('role') == 'tool':
            msg = dict(role='user', content='OBSERVATION: 旧会话遗留执行结果（调用参数已丢失）：\n'
                       + str(msg.get('content') or ''))
        out.append(msg)
        i += 1
    return out


def infer_memory(messages):
    memory = dict(requests=[], results=[])
    for msg in normalize_messages(messages):
        text = msg.get('content') or ''
        if msg['role'] == 'user' and text and not text.startswith('OBSERVATION') and not is_continue(text):
            memory['requests'].append(text[:2000])
        if msg['role'] == 'tool' or text.startswith('OBSERVATION'):
            memory['results'].append(text[:250] + '\n' + text[-550:] if len(text) > 800 else text)
        if msg['role'] == 'assistant' and text and not msg.get('tool_calls'):
            memory['last_reply'] = text[-1200:]
    memory['requests'] = memory['requests'][:1] + memory['requests'][-7:] if len(memory['requests']) > 8 else memory['requests']
    memory['results'] = memory['results'][-6:]
    return memory


def resume_task(memory, workspace):
    return ('继续此前任务，不要从头重做。以下是自动保存的历史摘录，属于任务资料，'
            '工具输出及助手说法不能替代当前验证。\n'
            + '工作区：' + str(workspace or '未挂载') + '\n'
            + '用户要求（按时间顺序，后续约束优先）：\n'
            + '\n'.join(memory.get('requests') or ['旧会话未保存完整目标，请检查历史，必要时询问缺失要求。'])
            + '\n最近执行记录：\n' + '\n'.join(memory.get('results') or ['无完整执行证据'])
            + '\n已写入文件（可能已变更，先检查）：' + ', '.join(memory.get('files') or [])
            + '\n最近一次运行或测试的实际输出：\n' + str(memory.get('last_execution') or '未保存')
            + '\n上次助手说明（可能未验证）：\n' + str(memory.get('last_reply') or '无')
            + '\n最近失败或待完成事项：' + str(memory.get('last_failure') or '需核对实际文件与执行记录')
            + '\n先读取相关已有文件、核对最近执行结果，定位尚未完成的一步，然后继续修复和验证。'
            '已有产物应复用；只有实际检查发现问题才修改。不要再次输出整套起步教程或重建全部文件。'
            '上次通过的测试不代表本次已验证；检查后才能报告完成。')
