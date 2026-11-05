"""Size the actual API prompt while retaining the current task and tool evidence."""
import json
from dataclasses import dataclass


def estimate_tokens(value):
    # Providers do not all expose their tokenizer. Include JSON escaping and use
    # a conservative estimate for Chinese, code and tool schemas.
    encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return (len(encoded) + 1) // 2


def context_size(value):
    try:
        size = int(str(value))
    except (TypeError, ValueError):
        return None
    return size if 512 <= size <= 10_000_000 else None


@dataclass(frozen=True)
class ContextBudget:
    window: int
    source: str
    output: int
    input_cap: int | None = None

    @property
    def input_limit(self):
        margin = max(64, min(1024, self.window // 16))
        limit = self.window - self.output - margin
        return min(limit, self.input_cap) if self.input_cap is not None else limit


def message_groups(messages):
    groups = []
    for message in messages:
        if message.get('role') == 'tool' and groups and groups[-1][0].get('tool_calls'):
            groups[-1].append(message)
        else:
            groups.append([message])
    return groups


def _wire_message(message):
    return {key: value for key, value in message.items() if not key.startswith('_context_')}


def _compact_tools(tools, descriptions=True):
    def omit_descriptions(value):
        if isinstance(value, list):
            return [omit_descriptions(item) for item in value]
        if isinstance(value, dict):
            return {key: omit_descriptions(item) for key, item in value.items() if key != 'description'}
        return value
    if not tools:
        return tools
    compact = []
    for tool in tools:
        fn = tool['function']
        schema = {'name': fn['name'], 'parameters': omit_descriptions(fn.get('parameters', {}))}
        if descriptions:
            schema['description'] = fn.get('description', '').split('。', 1)[0][:52]
        compact.append(dict(tool, function=schema))
    return compact


def fit_context(messages, tools, budget):
    """Return wire messages, tool schemas and a byte budget for the next file page."""
    groups = message_groups(messages)
    explicit_task = any(message.get('_context_required') for message in messages)
    last_user = next((group for group in reversed(groups) if group[0].get('role') == 'user'), None)
    latest_result = next((group for group in reversed(groups)
                          if group[0].get('tool_calls') or group[0].get('_context_observation')), None)

    def required(group):
        return (group[0].get('role') == 'system' or any(m.get('_context_required') for m in group)
                or (not explicit_task and group is last_user))

    def compact(group):
        return [dict(message, content=message['_context_short']) if '_context_short' in message else message
                for message in group]

    def wire(items):
        return [_wire_message(message) for group in items for message in group]

    def cost(items, schemas):
        return estimate_tokens({'messages': items, 'tools': schemas or []}) + 8 * len(items) + 24

    short_tools = _compact_tools(tools)
    essential = wire(compact(group) for group in groups if required(group))
    # Leave room for the call/result envelope and the next-page cursor.
    page_bytes = max(128, min(3600, 2 * (budget.input_limit - cost(essential, short_tools) - 280)))
    fitted = wire(groups)
    if cost(fitted, tools) <= budget.input_limit:
        return fitted, tools, page_bytes

    # Keep the latest actual tool evidence as one unit, including native call IDs.
    mandatory = [group for group in groups if required(group) or group is latest_result]
    kept = {id(group): compact(group) for group in mandatory}

    def ordered():
        return wire(kept[id(group)] for group in groups if id(group) in kept)

    if cost(ordered(), short_tools) > budget.input_limit:
        short_tools = _compact_tools(tools, descriptions=False)

    if cost(ordered(), short_tools) > budget.input_limit and latest_result is not None:
        reduced = []
        for message in kept[id(latest_result)]:
            content = message.get('content') or ''
            if (message.get('role') == 'tool' or message.get('_context_observation')) and len(content) > 380:
                if '下一页：' in content or '起始列' in content[:100]:
                    # A shortened page must never retain a cursor past unseen text.
                    content = content[:140] + '\n本页未完整送达，请从本页起始行和列重新读取更小范围。'
                else:
                    content = content[:120] + '\n…结果节选…\n' + content[-230:]
                message = dict(message, content=content)
            if message.get('tool_calls'):
                calls = []
                for call in message['tool_calls']:
                    function = call['function']
                    arguments = function.get('arguments') or '{}'
                    if len(arguments) > 360:
                        try:
                            fields = json.loads(arguments)
                        except ValueError:
                            fields = {}
                        if not isinstance(fields, dict):
                            fields = {}
                        brief = {key: fields[key] for key in ('path', 'id', 'line_start', 'line_end', 'column_start')
                                 if key in fields and len(str(fields[key])) <= 200}
                        call = dict(call, function=dict(function, arguments=json.dumps(brief, ensure_ascii=False)))
                    calls.append(call)
                message = dict(message, tool_calls=calls)
            reduced.append(message)
        kept[id(latest_result)] = reduced

    if short_tools and cost(ordered(), short_tools) > budget.input_limit:
        # On very small windows, favor the tools for the current step and
        # leave space for its evidence. Full schemas remain available next turn.
        task = '\n'.join(m.get('content') or '' for m in essential).lower()
        preferred = (
            'set_plan' if '先 set_plan' in task else 'update_task',
            'assemble_document', 'read_file',
            'tcp_echo_test' if 'tcp' in task or '联调' in task else
            'run_python' if 'python' in task or '.py' in task else 'run_command',
            'web_collect' if '联网' in task or '网页' in task else 'search_files',
            'load_skill' if '技能' in task else 'write_file',
            'read_file', 'write_file', 'append_file', 'list_files', 'search_files',
        )
        rank = {name: index for index, name in reversed(list(enumerate(preferred)))}
        candidates = sorted(short_tools, key=lambda tool: rank.get(tool['function']['name'], len(rank)))
        short_tools = []
        for tool in candidates:
            if cost(ordered(), short_tools + [tool]) <= budget.input_limit - 80:
                short_tools.append(tool)
        if not short_tools:
            raise ValueError(f'{budget.window} token 的窗口无法容纳当前任务和工具。请缩小当前子任务或核对模型上下文上限。')

    for group in reversed(groups):
        if id(group) in kept:
            continue
        kept[id(group)] = compact(group)
        if all(not message.get('content') and not message.get('tool_calls') for message in kept[id(group)]):
            del kept[id(group)]
            continue
        if cost(ordered(), short_tools) > budget.input_limit:
            del kept[id(group)]

    fitted = ordered()
    if cost(fitted, short_tools) > budget.input_limit:
        raise ValueError(f'{budget.window} token 的窗口无法同时容纳当前任务、工具和必要证据。请缩小当前子任务或核对模型上下文上限。')
    page_bytes = max(128, min(3600, 2 * (budget.input_limit - cost(essential, short_tools) - 280)))
    return fitted, short_tools, page_bytes
