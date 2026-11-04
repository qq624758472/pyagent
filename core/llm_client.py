"""LLM 客户端：OpenAI 兼容 /chat/completions，SSE 流式"""
import json
import queue
import re
import threading
import time
from contextlib import contextmanager
from urllib.parse import urlsplit

import requests

from core.config import ModelProfile
from core.log import get_logger, brief, mask

log = get_logger("core.llm_client")


class LLMError(Exception):
    pass


class LLMStreamInterrupted(LLMError):
    """已收到部分响应后网络中断；调用方必须丢弃本轮片段后才能重新请求。"""
    pass


# embedding 类模型名特征（无 chat 能力，不能用来对话/判定相关性）
_EMBEDDING_RE = re.compile(
    r"bge|text-embedding|embedding|gte-|e5-|jina-emb|nomic-embed|minilm", re.I)

def is_embedding_model(name: str) -> bool:
    """按模型名识别 embedding 类模型。kb_search / web_collect 的筛选角色
    误配此类模型时（如 bge-m3 当 chat 用）提前给出明确提示，避免静默降级。"""
    return bool(name and _EMBEDDING_RE.search(str(name)))


class LLMClient:
    # 连接类网络错误重试（本地 LLM 服务器偶发连接重置）；仅在本轮尚未
    # 产出任何内容时重试——流式中断后重发会导致内容重复
    RETRY_ON_CONN = 2
    RETRY_BACKOFF = 1.5
    REASONING_BEAT = 1.0           # 思考期心跳间隔（秒）——喂 UI 看门狗

    def __init__(self, cfg, profile=None):
        """profile: 角色名（"main"/"kb"/"web"），按 config.profiles 独立取
        模型/Base URL/Key，空值回落主配置；None = 纯主配置"""
        self.cfg = cfg
        p = cfg.profile(profile) if profile else ModelProfile()
        self.model = p.model or cfg.model
        self.base_url = p.base_url or cfg.base_url
        self.api_key = p.api_key or cfg.api_key
        self.last_reasoning = ""    # 最近一次流式的思考内容（无正文时的诊断用）
        self.reasoning_fallback_used = False   # 最近一次是否用思考过程兜底当回复
        self._request_lock = threading.Lock()
        self._active_requests = set()

    def _timeouts(self, timeout=None):
        if timeout is not None:
            return timeout
        try:
            seconds = int(getattr(self.cfg, 'llm_read_timeout', 300))
        except (TypeError, ValueError):
            seconds = 300
        return 10, max(30, min(1800, seconds))

    @contextmanager
    def _api_request(self, method, url, **kwargs):
        # 不修改操作系统代理；一个请求一个 Session，避免取消后的读取线程影响新任务。
        with requests.Session() as session:
            session.trust_env = bool(getattr(self.cfg, 'llm_use_system_proxy', True))
            with session.request(method, url, **kwargs) as response:
                yield response

    def _route_label(self, url):
        if not getattr(self.cfg, 'llm_use_system_proxy', True):
            return '直连接口'
        proxies = requests.utils.get_environ_proxies(url)
        proxy = proxies.get(urlsplit(url).scheme) or proxies.get('all')
        if proxy:
            parsed = urlsplit(proxy)
            return f'系统代理 {parsed.hostname}:{parsed.port or 80}'
        return '系统路由（当前无代理）'

    @staticmethod
    def _is_read_timeout(error):
        return isinstance(error, requests.ReadTimeout) or 'read timed out' in str(error).lower()

    @staticmethod
    def _compact_retry_payload(payload):
        """只压缩旧对话；保留任务及最近完整的 tool_calls/tool 组，不改调用方历史。"""
        messages = payload.get('messages') or []
        units, i = [], 2
        while i < len(messages):
            start = i
            i += 1
            if messages[start].get('role') == 'assistant' and messages[start].get('tool_calls'):
                while i < len(messages) and messages[i].get('role') == 'tool':
                    i += 1
            units.append(messages[start:i])
        if len(units) <= 2:
            return payload
        reduced = messages[:2] + [message for unit in units[-2:] for message in unit]
        old_size = len(json.dumps(messages, ensure_ascii=False))
        new_size = len(json.dumps(reduced, ensure_ascii=False))
        if new_size >= old_size * 0.8:
            return payload
        reduced.append({'role': 'user', 'content': (
            '上一次模型请求读取超时，已缩短较早的工具历史。保留原任务、最近结果和已生成文件，'
            '继续未完成工作。缺失资料时重新读取相关文件，不要重建已完成产物；实际验证后再总结。')})
        return dict(payload, messages=reduced)

    def _payload(self, messages, model=None):
        name = model or self.model
        payload = {"model": name, "messages": messages,
                   "temperature": self.cfg.temperature, "stream": True}
        # gpt-oss 单轮生成预算，避免推理长时间占据工具执行循环。
        if "gpt-oss" in name.lower():
            payload["max_tokens"] = getattr(self.cfg, "max_output_tokens", 3072)
            payload["reasoning_effort"] = getattr(self.cfg, "reasoning_effort", "low")
        return payload

    def _iter_sse(self, payload, m_desc, timeout=None):
        """将阻塞网络读取与事件流分开：等待首字节时也可取消和发送等待状态。"""
        cancelled = threading.Event()
        events = queue.Queue(maxsize=128)
        with self._request_lock:
            self._active_requests.add(cancelled)

        def send(item):
            while not cancelled.is_set():
                try:
                    events.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def read():
            try:
                for item in self._read_sse(payload, m_desc, timeout, cancelled):
                    if cancelled.is_set():
                        return
                    send(item)
            except Exception as exc:
                send(("error", exc))
            finally:
                send(("done", None))

        reader = threading.Thread(target=read, name="llm-network", daemon=True)
        reader.start()
        connect, read_timeout = self._timeouts(timeout)
        # 既保留已有重试预算，也限制一直发送无效/心跳数据的服务器。
        budget = connect + read_timeout * (1 + self.RETRY_ON_CONN) + self.RETRY_BACKOFF * 3
        deadline = time.monotonic() + budget
        last_wait = time.monotonic()
        try:
            while True:
                if cancelled.is_set():
                    raise LLMError("模型请求已取消")
                if time.monotonic() >= deadline:
                    raise LLMError("模型请求超过网络等待与重试预算，任务未完成；已保留工作区文件。")
                try:
                    kind, data = events.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - last_wait >= self.REASONING_BEAT:
                        last_wait = time.monotonic()
                        yield ("waiting", "")
                    continue
                if cancelled.is_set():
                    raise LLMError("模型请求已取消")
                if kind == "done":
                    break
                if kind == "error":
                    raise data
                if kind in ('content', 'reasoning', 'tool_delta'):
                    deadline = time.monotonic() + budget  # 有真实数据时继续，不限制整条回复总时长。
                yield kind, data
        finally:
            cancelled.set()
            with self._request_lock:
                self._active_requests.discard(cancelled)

    def cancel_pending(self):
        """立即结束消费端；正在阻塞的 HTTP 读取按其超时释放，禁止再重试。"""
        with self._request_lock:
            for request in self._active_requests:
                request.set()

    def _read_sse(self, payload, m_desc, timeout=None, cancelled=None):
        """底层 SSE 迭代：产出 ("content"|"reasoning", 文本) 与 ("tool_delta", 片段列表)

        连接中断或可恢复的 HTTP 5xx 在尚未产出正文/工具调用时重试；其他 HTTP 错误不重试。
        timeout: 覆盖默认 (10,180)——长链路里的子步骤（如 web_collect 筛选）
        用更短超时，避免单次请求把整条任务拖死。"""
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        req_timeout = self._timeouts(timeout)
        chunks = 0
        for attempt in range(1 + self.RETRY_ON_CONN):
            if cancelled is not None and cancelled.is_set():
                return
            try:
                with self._api_request('POST', url, headers=headers, json=payload,
                                       stream=True, timeout=req_timeout) as r:
                    r.raise_for_status()
                    r.encoding = "utf-8"
                    for line in r.iter_lines(chunk_size=1, decode_unicode=True):
                        if cancelled is not None and cancelled.is_set():
                            return
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            log.debug("LLM 无法解析的 SSE 行: %s", brief(data, 120))
                            continue
                        d = (chunk.get("choices") or [{}])[0].get("delta") or {}
                        # 思考模型的推理内容在独立字段（部分网关如此）——
                        # 此前只读 content，遇到只吐 reasoning 的模型会
                        # 连续拿到空回复（日志 chunks=0 连环实证，V23 修复）
                        rc = d.get("reasoning_content") or d.get("reasoning")
                        if rc:
                            yield ("reasoning", rc)
                        content = d.get("content")
                        if content:
                            chunks += 1
                            yield ("content", content)
                        tc = d.get("tool_calls")
                        if tc:
                            chunks += 1
                            yield ("tool_delta", tc)
                break                   # 正常完成，跳出重试循环
            except requests.HTTPError as e:
                if cancelled is not None and cancelled.is_set():
                    return
                code = e.response.status_code if e.response is not None else "?"
                detail = ""
                if e.response is not None:
                    try:
                        raw_detail = e.response.content or b""
                        if isinstance(raw_detail, bytes):
                            raw_detail = raw_detail.decode("utf-8", errors="replace")
                        detail = str(raw_detail).strip()[:240]
                    except Exception:
                        detail = ""
                # 推理服务/FRP 网关偶发 500/502/503/504：未收到正文或工具调用时安全重试。
                # 已流出工具调用片段时不可重试，否则可能拼接/重复工具命令。
                if (code in (500, 502, 503, 504) and chunks == 0
                        and attempt < self.RETRY_ON_CONN):
                    wait = self.RETRY_BACKOFF * (attempt + 1)
                    log.warning("LLM HTTP %s，%.1fs 后重试(%d/%d)",
                                code, wait, attempt + 1, self.RETRY_ON_CONN)
                    if cancelled is not None:
                        if cancelled.wait(wait):
                            return
                    else:
                        time.sleep(wait)
                    continue
                log.error("LLM 请求失败 %s HTTP %s", m_desc, code)
                suffix = f"；服务端信息: {detail}" if detail else ""
                raise LLMError(
                    f"API 请求失败(HTTP {code})，请检查 base_url/api_key/模型名。{suffix}") from e
            except requests.RequestException as e:
                if cancelled is not None and cancelled.is_set():
                    return
                if chunks > 0 or attempt >= self.RETRY_ON_CONN:
                    log.error("LLM 网络错误 %s: %s", m_desc, brief(e, 200))
                    if self._is_read_timeout(e):
                        error_type = LLMStreamInterrupted if chunks else LLMError
                        raise error_type(
                            f"模型读取超时（{self._route_label(url)}，单次等待 {req_timeout[1]} 秒）。"
                            + ("响应中途断开，未重复提交这轮请求。" if chunks else f"已重试 {attempt} 次仍未收到完整响应。")
                            + "任务未完成，文件已保留。可在设置中调整等待时间或选择直连接口，并检查模型服务负载。") from e
                    error_type = LLMStreamInterrupted if chunks else LLMError
                    raise error_type(f"网络错误: {e}") from e
                wait = self.RETRY_BACKOFF * (attempt + 1)
                if self._is_read_timeout(e):
                    compact = self._compact_retry_payload(payload)
                    changed = compact is not payload
                    payload = compact
                    yield ('retry', f"模型读取超时，{'已缩短历史，' if changed else ''}准备重试 {attempt + 1}/{self.RETRY_ON_CONN}（{self._route_label(url)}）")
                log.warning("LLM 网络错误，%.1fs 后重试(%d/%d): %s",
                            wait, attempt + 1, self.RETRY_ON_CONN, brief(e, 150))
                if cancelled is not None:
                    if cancelled.wait(wait):
                        return
                else:
                    time.sleep(wait)

    def chat_stream(self, messages, model=None, timeout=None, on_activity=None):
        """文本协议流：产出正文增量（str）。reasoning 单独收集，空正文时兜底。
        timeout: 可选覆盖默认 (10,180)，用于长链路里的子步骤（如 web_collect 筛选）。
        on_activity: 可选回调，reasoning 分片到达时调用一次（V31.2：思考期
        静默会让 UI 看门狗误判超时，调用方据此抛心跳）。"""
        if not self.api_key:
            raise LLMError("未配置 API Key。请设置环境变量 AGENT_API_KEY，"
                           "或通过界面「设置…」/控制台 /key 配置。")
        payload = self._payload(messages, model)
        m_desc = (f"model={payload['model']} msgs={len(messages)}"
                  f" total_chars={sum(len(m.get('content') or '') for m in messages)}")
        log.info("LLM 请求开始 %s", m_desc)
        log.debug("LLM 完整 messages: %s", mask(json.dumps(
            messages, ensure_ascii=False)[:4000]))
        t0 = time.monotonic()
        chunks = 0
        reasoning_buf = []           # reasoning_content 累积（gpt-oss/deepseek-r1 等）
        last_activity_beat = t0
        self.last_reasoning = ""
        self.reasoning_fallback_used = False
        for kind, data in self._iter_sse(payload, m_desc, timeout=timeout):
            if kind in ("waiting", "retry"):
                yield ""  # Agent 将其转换成 keepalive，不计作正文或进展。
            elif kind == "content":
                chunks += 1
                yield data
            elif kind == "reasoning":
                reasoning_buf.append(data)
                if on_activity is not None:
                    try:
                        on_activity()
                    except Exception:
                        pass
                if time.monotonic() - last_activity_beat >= self.REASONING_BEAT:
                    last_activity_beat = time.monotonic()
                    yield ""
        self.last_reasoning = "".join(reasoning_buf)
        log.info("LLM 响应完成 %s chunks=%d 用时 %.1fs",
                 m_desc, chunks, time.monotonic() - t0)
        if chunks == 0 and self.last_reasoning:
            # 端点把全部输出都放进了 reasoning_content（gpt-oss 经 llama.cpp 等
            # 网关的已知现象：harmony 格式的 final 通道没被分离到 content）——
            # 模型的协议文本/正文就在思考过程里，直接当回复输出，保证 Agent 可用
            log.warning("正文为空，将思考过程(%d字)作为回复输出（端点未分离 "
                        "reasoning_content）: %s",
                        len(self.last_reasoning),
                        brief(self.last_reasoning, 300))
            self.reasoning_fallback_used = True
            yield self.last_reasoning
        elif chunks == 0:
            log.warning("模型返回空内容（无正文也无思考过程）")

    def chat_turn(self, messages, tools=None, model=None, timeout=None):
        """原生 function calling 一轮（V28）：产出事件字典

        {"type": "content", "text": ...}      正文增量
        {"type": "tool_call", "id","name","arguments"}   完整的一次工具调用
        {"type": "keepalive"}                 思考期心跳（V31.2）
        （思考内容不外抛，只进 last_reasoning 供诊断）

        V31.2：gpt-oss 这类推理模型在出正文前会长时间只吐 reasoning——
        此前这段"静默期"不产生任何事件，UI 看门狗 45s 无事件直接判超时
        （日志 18:18 实证）。这里按 1s 间隔抛 keepalive 喂看门狗。"""
        if not self.api_key:
            raise LLMError("未配置 API Key。请设置环境变量 AGENT_API_KEY，"
                           "或通过界面「设置…」/控制台 /key 配置。")
        payload = self._payload(messages, model)
        if tools:
            self.reasoning_fallback_used = False
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        m_desc = (f"model={payload['model']} msgs={len(messages)}"
                  f" tools={len(tools or [])}"
                  f" total_chars={sum(len(m.get('content', '') or '') for m in messages)}")
        log.info("LLM 请求开始 %s", m_desc)
        t0 = time.monotonic()
        calls = {}                 # index -> {id, name, arguments}
        content_parts, reasoning_parts = [], []
        last_beat = t0
        for kind, data in self._iter_sse(payload, m_desc, timeout=timeout):
            if kind == 'retry':
                yield {'type': 'progress', 'content': data}
                continue
            now = time.monotonic()
            if kind == "waiting" or now - last_beat >= self.REASONING_BEAT:
                last_beat = now
                yield {"type": "keepalive", "phase": "waiting" if kind == "waiting" else "receiving"}
            if kind == "content":
                content_parts.append(data)
                yield {"type": "content", "text": data}
            elif kind == "reasoning":
                reasoning_parts.append(data)
                now = time.monotonic()
                if now - last_beat >= self.REASONING_BEAT:
                    last_beat = now
                    yield {"type": "keepalive"}      # 思考期心跳：喂看门狗
            elif kind == "tool_delta":
                for tc in data:
                    idx = tc.get("index", 0)
                    slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
        self.last_reasoning = "".join(reasoning_parts)
        text = "".join(content_parts)
        for idx in sorted(calls):
            slot = calls[idx]
            if slot["name"]:
                log.info("原生工具调用: %s(%s)", slot["name"], brief(slot["arguments"], 200))
                yield {"type": "tool_call", "id": slot["id"] or f"call_{idx}",
                       "name": slot["name"], "arguments": slot["arguments"]}
        if not content_parts and not calls and self.last_reasoning:
            # 原生协议不能把 analysis 当最终答案；由 Agent 空轮重试。
            log.warning("原生协议：只有推理(%d字)无最终内容/工具调用，按空轮处理",
                        len(self.last_reasoning))
        log.info("LLM 响应完成 %s content=%d字 tools=%d 用时 %.1fs",
                 m_desc, len(text), len(calls), time.monotonic() - t0)

    def list_models(self) -> list:
        """拉取端点支持的模型列表（OpenAI 兼容 GET /models，智谱/OpenAI 等通用）"""
        if not self.api_key:
            raise LLMError("未配置 API Key，无法获取模型列表。")
        url = self.base_url.rstrip("/") + "/models"
        log.info("拉取模型列表 %s", mask(url))
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = None
        for attempt in range(2):                 # 连接类错误重试一次
            try:
                with self._api_request('GET', url, headers=headers, timeout=(10, 30)) as r:
                    r.raise_for_status()
                    data = r.json()
                break
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else "?"
                if code == 404:
                    raise LLMError("该地址不支持 /models 接口(HTTP 404)，请手动填写模型名。") from e
                if code == 401:
                    raise LLMError("API Key 无效或未授权(HTTP 401)。") from e
                raise LLMError(f"获取模型失败(HTTP {code})。") from e
            except requests.RequestException as e:
                if attempt == 0:
                    log.warning("拉取模型列表网络错误，1s 后重试: %s", brief(e, 150))
                    time.sleep(1.0)
                    continue
                raise LLMError(f"网络错误: {e}") from e
            except ValueError:
                raise LLMError("返回内容不是合法 JSON，请检查 Base URL 是否正确。")

        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise LLMError("返回结构不是预期的模型列表（缺少 data 数组）。")
        ids = sorted({it.get("id") for it in items
                      if isinstance(it, dict) and it.get("id")})
        if not ids:
            raise LLMError("模型列表为空。")
        return ids
