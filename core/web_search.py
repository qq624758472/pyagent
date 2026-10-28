"""联网数据采集：多引擎搜索 + 筛选模型符合性过滤 + 自动切换补爬

引擎链：百度 → 必应 → 搜狗 → 360。某个引擎触发人机验证/风控时
（EngineBlocked）自动切换下一个引擎继续，直到凑满 count 条有效数据
或尝试上限。DuckDuckGo 未纳入默认链（国内网络普遍不可达）。

流程（每轮）：当前引擎取一页 10 条 → URL 去重 → 筛选模型批量判定
→ 有效数据累计；不足则该引擎翻页；引擎被 ban 则换下一个从头爬。"""
import base64
import json
import re
import time

import requests

from core.llm_client import LLMClient, LLMError, is_embedding_model
from core.log import get_logger, brief

log = get_logger("core.web_search")

MAX_ATTEMPTS = 10      # 总尝试上限（含引擎切换/失败），防死循环
PAGE_SIZE = 10         # 每页条数
REQUEST_TIMEOUT = (10, 20)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}

_TAG = re.compile(r"<[^>]+>")
_H3A = re.compile(r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_AD = re.compile(r'ec_tuiguang|data-tuiguang|"ec-')
_JUNK = ('"}]', 'component_content', '{"clam')


class WebSearchError(Exception):
    pass


class EngineBlocked(WebSearchError):
    """触发搜索引擎人机验证/风控，应切换下一个引擎"""


def _clean(text, limit=260):
    """去标签/压缩空白/剔除页面内嵌 JSON 残渣"""
    t = _TAG.sub(" ", text)
    t = re.sub(r"\s+", " ", t).strip()
    for junk in _JUNK:
        i = t.find(junk)
        if i > 20:
            t = t[:i]
    return t[:limit]


def _fetch(url, params):
    try:
        r = requests.get(url, params=params, headers=_HEADERS,
                         timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("引擎请求失败 %s: %s", url, brief(e, 150))
        raise WebSearchError(f"请求失败: {e}")
    r.encoding = "utf-8"
    if r.status_code != 200:
        log.warning("引擎响应异常 %s: HTTP %d (len=%d)",
                    url, r.status_code, len(r.text))
        raise WebSearchError(f"HTTP {r.status_code}")
    log.info("引擎响应 %s (len=%d)", url, len(r.text))
    return r.text


def _check_page(html, engine):
    """通用页面健康检查：验证页 / 异常短页"""
    if len(html) < 20000:                 # 正常 SERP 都在几十 KB 以上
        log.warning("%s 疑似风控空页 (len=%d)", engine, len(html))
        raise EngineBlocked(
            f"{engine}返回异常短页({len(html)//1024}KB，疑似风控空页)")
    return html


def _parse_h3_blocks(html, engine, ad_filter=False):
    """通用宽松解析：以 h3-a 为锚点取标题，其后到下一个 h3 的文本作摘要"""
    blocks = re.split(r"(?=<h3[^>]*>\s*<a )", html)[1:]
    items, seen = [], set()
    for b in blocks:
        m = _H3A.search(b)
        if not m:
            continue
        if ad_filter and _AD.search(b):
            continue
        title = _clean(m.group(2), 80)
        url = m.group(1)
        if not title or url in seen:
            continue
        seen.add(url)
        if url.startswith("/"):            # 站内相对链接 → 拼引擎域名
            base = {"搜狗": "https://www.sogou.com",
                    "360": "https://www.so.com"}.get(engine, "")
            url = base + url
        snippet = _clean(b[m.end():])
        items.append({"title": title, "snippet": snippet, "url": url,
                      "engine": engine})
        if len(items) >= PAGE_SIZE:
            break
    return items


class _Engine:
    name = ""
    # 触发 EngineBlocked 的页面特征（子类覆盖）
    blocked_marks = ()

    def __init__(self):
        self.page = 0                      # 该引擎已爬页数（切换后从头开始）

    def fetch(self, query):
        raise NotImplementedError

    def _guarded(self, html):
        for mark in self.blocked_marks:
            if mark in html:
                log.warning("%s 触发人机验证: %s", self.name, mark)
                raise EngineBlocked(f"{self.name}触发人机验证({mark})")
        return _check_page(html, self.name)

    def search(self, query):
        """取下一页结果；引擎级翻页。无结果返回空列表"""
        html = self._guarded(self.fetch(query))
        return self.parse(html)

    def parse(self, html):
        raise NotImplementedError


class BaiduEngine(_Engine):
    name = "百度"
    blocked_marks = ("百度安全验证", "wappass.baidu.com")

    def fetch(self, query):
        return _fetch("https://www.baidu.com/s",
                      {"wd": query, "pn": self.page * PAGE_SIZE,
                       "rn": PAGE_SIZE})

    def parse(self, html):
        return _parse_h3_blocks(html, self.name, ad_filter=True)


class BingEngine(_Engine):
    name = "必应"
    blocked_marks = ()

    def fetch(self, query):
        return _fetch("https://cn.bing.com/search",
                      {"q": query, "first": self.page * PAGE_SIZE + 1,
                       "count": PAGE_SIZE})

    _ALGO = re.compile(r'<li class="b_algo".*?<h2[^>]*><a[^>]*href="([^"]+)"'
                       r'[^>]*>(.*?)</a>(.*?)</li>', re.S)
    # 必应把外链包成 /ck/a?...&u=a1<base64url> 跳转
    _CK_U = re.compile(r"[?&]u=a1([A-Za-z0-9_\-]+)")

    @classmethod
    def _unwrap(cls, url: str) -> str:
        """解码必应 /ck/a 跳转链接，还原真实目标 URL

        不还原的话所有结果 url 都是 cn.bing.com/ck/a...：去重失真、
        交给用户的「来源」也没法直接点开。"""
        if "/ck/a" not in url:
            return url
        m = cls._CK_U.search(url)
        if not m:
            return url
        tok = m.group(1)
        try:
            raw = base64.urlsafe_b64decode(tok + "=" * (-len(tok) % 4))
            decoded = raw.decode("utf-8", "replace")
        except (ValueError, TypeError):
            return url
        return decoded if decoded.startswith(("http://", "https://")) else url

    def parse(self, html):
        items, seen = [], set()
        for m in self._ALGO.finditer(html):
            url, title_html, rest = m.groups()
            title = _clean(title_html, 80)
            url = self._unwrap(url)
            if not title or url in seen or url.startswith("/"):
                continue
            seen.add(url)
            # 摘要取 b_algo 块内的正文段落
            seg = re.search(r'<p[^>]*>(.*?)</p>', rest, re.S)
            snippet = _clean(seg.group(1)) if seg else ""
            items.append({"title": title, "snippet": snippet, "url": url,
                          "engine": self.name})
            if len(items) >= PAGE_SIZE:
                break
        return items


class SogouEngine(_Engine):
    name = "搜狗"
    blocked_marks = ("antispider", "搜狗验证", "seccode")

    def fetch(self, query):
        return _fetch("https://www.sogou.com/web",
                      {"query": query, "page": self.page + 1})

    def parse(self, html):
        return _parse_h3_blocks(html, self.name)


class So360Engine(_Engine):
    name = "360"
    blocked_marks = ("360搜索安全中心", "verify.so.com")

    def fetch(self, query):
        return _fetch("https://www.so.com/s",
                      {"q": query, "pn": self.page + 1})

    def parse(self, html):
        return _parse_h3_blocks(html, self.name)


DEFAULT_ENGINES = (BaiduEngine, BingEngine, SogouEngine, So360Engine)


# ============================================================
# 筛选模型：批量符合性判定
# ============================================================
def screen_items(items, criteria, llm, model, progress=None, should_stop=None):
    """用筛选模型批量判定；解析失败时保守放行并注明

    progress: 可选心跳回调——筛选是阻塞式模型调用，期间若模型慢会长时间
    无任何事件，导致 UI 看门狗误报「对话超时」。逐块上报进度使其持续续命。
    should_stop: 可选停止信号，流式期间每片检查，用户点停止可即时中断。"""
    if not criteria:
        return list(items), "未设置筛选标准，全部保留"
    emb_hint = ("（该模型为 embedding 类，无对话能力，请更换「联网搜索」角色模型）"
                if is_embedding_model(model) else "")
    lines = "\n".join(f"{i + 1}. {it['title']} — {it['snippet'][:80]}"
                      for i, it in enumerate(items))
    messages = [
        {"role": "user", "content":
            f"你是数据筛选器。筛选标准：{criteria}\n"
            f"下面是 {len(items)} 条搜索结果，判断每条是否符合标准。\n"
            f"只输出 JSON 数组，格式：[{{\"i\": 1, \"ok\": true}}, "
            f"{{\"i\": 2, \"ok\": false}}]\n"
            f"i 是序号，ok 是是否符合。不要输出其他内容。\n\n{lines}"}
    ]
    try:
        full = ""
        for d in llm.chat_stream(messages, model=model, timeout=(10, 60)):
            if should_stop and should_stop():
                log.info("筛选判定被用户中断")
                break
            full += d
            # 心跳：每收到一块就上报进度，持续喂看门狗——筛选是阻塞式模型
            # 调用，模型慢时可能几十秒不出 token，必须靠逐块心跳续命，否则
            # UI 误报「对话超时」。筛选回复通常很短，逐块上报开销可忽略。
            if progress:
                progress("〔联网〕筛选模型判定中…")
    except LLMError as e:
        return list(items), f"筛选模型调用失败({e}){emb_hint}，本批全部保留"
    m = re.search(r"\[.*\]", full, re.S)
    if not m:
        return list(items), f"筛选模型输出无法解析{emb_hint}，本批全部保留"
    try:
        verdicts = {int(v["i"]): bool(v.get("ok"))
                    for v in json.loads(m.group(0))}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return list(items), "筛选结果解析失败，本批全部保留"
    ok_idx = {i for i, ok in verdicts.items() if ok}
    passed = [it for i, it in enumerate(items, 1) if i in ok_idx]
    return passed, f"本批 {len(items)} 条，筛选通过 {len(passed)} 条"


# ============================================================
# 相关性硬校验：搜索引擎分词失败时兜底（如 Bing 把专有名词拆成单字）
# ============================================================
_SPLIT = re.compile(r"[\s，,。；;、！!？?/\\·\-—()（）\[\]{}\"'`~@#$%^&*+=|:]+")
_ASCII = re.compile(r"^[\x00-\x7f]+$")


def _keywords(query):
    """查询分段：段内 ≥2 个汉字（或 ≥3 个 ASCII 字符）视为关键词"""
    kws = []
    for seg in _SPLIT.split(query):
        if _ASCII.match(seg):
            if len(seg) >= 3:
                kws.append(seg)
        elif len(seg) >= 2:                # 中文段：≥2 字，如「西光肆号01星」
            kws.append(seg)
    return kws or [query]                  # 极端情况整串当关键词


def _relevant(item, kws):
    """条目与查询相关：整段关键词命中，或长中文段（≥4 字）相邻 2-gram 至少命中 2 个

    后者是降级通道——如「卫星发射时间」整串未必出现在标题里，但真结果
    会同时含「卫星」「发射」等片段，而字典/泛义词页只可能零星撞到 1 个。"""
    text = item["title"] + item["snippet"]
    for k in kws:
        if k in text:
            return True
    for k in kws:
        if len(k) >= 4 and not _ASCII.match(k):
            grams = {k[i:i + 2] for i in range(len(k) - 1)}
            if sum(1 for g in grams if g in text) >= 2:
                return True
    return False


# ============================================================
# 采集主循环：多引擎 + 筛选 + 自动切换 + 实时进度
# ============================================================
def collect_until(query, criteria, count, cfg, progress=None, should_stop=None):
    """爬取→相关性硬校验→筛选→翻页/换引擎，直到凑满 count 条有效数据

    progress: 可选回调 callable(str)，采集过程实时推送（跨线程安全由
    调用方保证——GUI 用 Qt 信号、CLI 用 print）。超时预算 cfg.web_timeout
    秒后返回已采集部分。
    should_stop: 可选停止信号 callable()->bool（用户点停止）。采集期间持续
    检查，命中即中断并返回已采集部分——此前 web_collect 是同步阻塞调用，
    看门狗触发 stop 也无法中断，会一直跑到时间预算用完。"""
    llm = LLMClient(cfg, profile="web")     # 联网角色：独立模型/URL/Key，空则回落主配置
    model = llm.model
    kws = _keywords(query)
    budget = max(30, int(getattr(cfg, "web_timeout", 90) or 90))   # 总时间预算(秒)
    t0 = time.monotonic()

    def _p(msg):
        if progress:
            try:
                progress(msg)               # 进度回调异常不影响采集本身
            except Exception:
                pass

    # 未显式给筛选标准时，用「与查询直接相关」做默认语义筛选——
    # 避免"无标准=全保留"把引擎的分词垃圾（字典页/泛义词）当成果交差
    if not criteria:
        criteria = f"内容与「{query}」直接相关（标题或摘要围绕该主题，而非仅字面撞到个别字）"
        default_std = True
    else:
        default_std = False
    engines = [E() for E in DEFAULT_ENGINES]
    valid, seen = [], set()
    notes, rounds, total, attempts = [], 0, 0, 0
    timed_out = False
    stopped = False

    log.info("联网采集开始: query=%r count=%d budget=%ds 引擎链=%s",
             query, count, budget, "/".join(E.name for E in DEFAULT_ENGINES))
    _p(f"〔联网〕开始采集「{query}」（目标 {count} 条，时间预算 {budget}s）")
    while engines and len(valid) < count and attempts < MAX_ATTEMPTS:
        if should_stop and should_stop():
            stopped = True
            notes.append("用户已停止采集，返回已获取部分")
            _p("〔联网〕用户已停止，返回已采集部分")
            break
        left = budget - (time.monotonic() - t0)
        if left <= 0:
            timed_out = True
            notes.append("已达时间预算，停止采集并返回已获取部分")
            _p("〔联网〕已达时间预算，返回已采集部分")
            break
        eng = engines[0]
        attempts += 1
        _p(f"〔联网〕{eng.name} 第 {eng.page + 1} 页爬取中…")
        try:
            items = eng.search(query)
        except EngineBlocked as e:
            log.warning("引擎切换(blocked): %s", e)
            notes.append(f"⚠ {e}，已自动切换")
            _p(f"〔联网〕⚠ {e}，切换下一引擎")
            engines.pop(0)
            continue
        except WebSearchError as e:
            notes.append(f"⚠ 「{eng.name}」{e}，已自动切换")
            _p(f"〔联网〕⚠ 「{eng.name}」{e}，切换下一引擎")
            engines.pop(0)
            continue
        eng.page += 1
        rounds += 1
        items = [it for it in items if it["url"] not in seen]
        seen.update(it["url"] for it in items)
        total += len(items)
        if not items:
            notes.append(f"「{eng.name}」第 {eng.page} 页无新结果，换引擎")
            _p(f"〔联网〕{eng.name} 第 {eng.page} 页无新结果，换引擎")
            engines.pop(0)
            continue
        # 硬校验：关键词零命中的条目直接丢弃（零成本，先于模型筛选）
        hits = [it for it in items if _relevant(it, kws)]
        dropped = len(items) - len(hits)
        if not hits:
            # 整页无一条命中 → 该引擎把查询拆碎了（如 Bing 长中文词分词失败），
            # 翻页无意义，直接换引擎
            notes.append(f"⚠ 「{eng.name}」返回 {len(items)} 条均与查询无字面关联"
                         f"（疑似分词失败），已自动切换")
            _p(f"〔联网〕{eng.name} 返回 {len(items)} 条均与查询无关联（分词失败），换引擎")
            engines.pop(0)
            continue
        _p(f"〔联网〕{eng.name} 第 {eng.page} 页: 爬到 {len(items)} 条，"
           f"正在筛选判定…")
        passed, note = screen_items(hits, criteria, llm, model,
                                    progress=progress, should_stop=should_stop)
        note += f"，硬校验丢弃无关 {dropped} 条"
        log.info("采集轮次: 引擎=%s 页=%d 爬=%d 硬校验留=%d 筛过=%d 累计=%d/%d",
                 eng.name, eng.page, len(items), len(hits), len(passed),
                 len(valid), count)
        notes.append(f"「{eng.name}」第 {eng.page} 页: 爬到 {len(items)} 条，{note}")
        valid.extend(passed)
        if passed:
            # 应答先显示：把本页筛出的条目标题实时推给用户，不等凑满
            preview = "；".join(f"[{i}] {it['title'][:24]}"
                                for i, it in enumerate(passed[:3], len(valid) - len(passed) + 1))
            _p(f"〔联网〕已筛出 {len(valid)}/{count} 条: {preview}"
               + ("…" if len(passed) > 3 else ""))
        # 单轮（含一次阻塞式筛选调用）可能已耗时很久——再核一次预算，
        # 否则 web_timeout 只在轮次间生效、对慢筛选调用形同虚设
        if should_stop and should_stop():
            stopped = True
            notes.append("用户已停止采集，返回已获取部分")
            _p("〔联网〕用户已停止，返回已采集部分")
            break
        if budget - (time.monotonic() - t0) <= 0:
            timed_out = True
            notes.append("已达时间预算，停止采集并返回已获取部分")
            _p("〔联网〕已达时间预算，返回已采集部分")
            break
        time.sleep(1.2)                       # 温和爬取，降低风控概率

    valid = valid[:count]
    used = {it["engine"] for it in valid}
    std_show = "默认相关性" if default_std else criteria
    tail = ("（已达时间预算，提前结束）" if timed_out
            else "（用户已停止，提前结束）" if stopped else "")
    out = [f"联网采集完成（关键词「{query}」，筛选标准「{std_show}」，"
           f"筛选模型 {model}）",
           f"共尝试 {attempts} 轮 / 爬到 {total} 条，筛选后有效 {len(valid)} 条"
           f"（要求 {count} 条），数据来源: {'、'.join(sorted(used)) or '无'}"
           + tail]
    out += [f"  · {n}" for n in notes]
    if valid:
        out.append("")
        for i, it in enumerate(valid, 1):
            out.append(f"[{i}] {it['title']}\n    {it['snippet']}\n"
                       f"    来源: {it['url']} [{it['engine']}]")
    else:
        out.append("未获取到有效数据。建议：换关键词（更短/加引号精确匹配）/ "
                   "检查网络（各搜索引擎可能均已触发验证，稍后再试）。")
    log.info("联网采集结束: 有效 %d/%d 尝试 %d 轮 total=%d timed_out=%s stopped=%s",
             len(valid), count, attempts, total, timed_out, stopped)
    _p(f"〔联网〕采集结束: 有效 {len(valid)}/{count} 条")
    return "\n".join(out)
