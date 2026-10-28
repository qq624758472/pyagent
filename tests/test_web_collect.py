"""web_collect 长任务健壮性（离线，不连模型/搜索引擎）

针对 V28 后暴露的真实问题：web_collect 是同步阻塞调用，期间若筛选模型慢
会长时间无事件 → UI 看门狗误报「对话超时」；且看门狗触发 stop() 也无法
中断它。本测试验证：① 筛选调用期间持续心跳（喂看门狗）；② 用户停止信号
能即时中断采集并返回已获取部分；③ 单轮耗时超预算时该轮结束即中断。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import web_search as ws              # noqa: E402
from core.config import Config                # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name
          + (f"  | {detail}" if detail and not cond else ""))


class _Patch:
    """极简 monkeypatch：保存属性，退出还原"""
    def __init__(self, target, attr, value):
        self.t, self.a, self.v = target, attr, value
        self._old = getattr(target, attr, None)
    def __enter__(self):
        setattr(self.t, self.a, self.v)
        return self
    def __exit__(self, *e):
        setattr(self.t, self.a, self._old)


# ---------- 假 LLM：可控慢 / 可控停止 ----------
class FakeLLM:
    def __init__(self, cfg=None, profile=None):
        self.model = "fake-screen"
        self.chat_calls = 0
        self.stop = None          # 可选 callable()->bool，模拟用户停止
        self.chunk_sleep = 0.0    # 每块休眠，模拟慢模型

    def chat_stream(self, messages, model=None, timeout=None):
        self.chat_calls += 1
        text = '[{"i": 1, "ok": true}, {"i": 2, "ok": false}]'
        for i in range(0, len(text), 4):
            if self.stop is not None and self.stop():
                return                      # 模拟中断：不再产出
            if self.chunk_sleep:
                time.sleep(self.chunk_sleep)
            yield text[i:i + 4]


# ---------- 假引擎：不联网，返回含关键词的结果（每页 URL 唯一，dedup 不耗尽） ----------
class FakeEngine(ws._Engine):
    name = "测试引擎"
    blocked_marks = ()
    _n = 0
    def fetch(self, query):
        FakeEngine._n += 1
        return "<html>" + ("x" * 25000) + "</html>"   # 过 _check_page 长度门槛
    def parse(self, html):
        k = FakeEngine._n
        return [{"title": "僵尸毁灭工程攻略", "snippet": "关于僵尸毁灭工程的生存指南",
                 "url": f"http://example.com/{k}/1", "engine": self.name},
                {"title": "无关词条", "snippet": "完全不相关的内容",
                 "url": f"http://example.com/{k}/2", "engine": self.name}]


def _install(llm):
    pm = [_Patch(ws, "LLMClient", lambda cfg=None, profile=None: llm),
          _Patch(ws, "DEFAULT_ENGINES", (FakeEngine,)),
          _Patch(ws, "_check_page", lambda html, engine: html)]
    for p in pm:
        p.__enter__()
    return pm


def _uninstall(pm):
    for p in reversed(pm):
        p.__exit__()


def test_screen_heartbeat():
    """筛选调用期间必须持续上报进度（心跳），否则慢模型会让看门狗误杀"""
    beats = []
    llm = FakeLLM()
    llm.chunk_sleep = 0.05           # 人为拉长，放大静默窗口
    items = [{"title": "僵尸毁灭工程", "snippet": "x", "url": "u1", "engine": "测试"}]
    passed, note = ws.screen_items(items, "与僵尸毁灭工程相关", llm, llm.model,
                                   progress=beats.append)
    check("筛选心跳进度", bool(passed) and any("判定中" in b for b in beats),
          f"passed={passed} beats={beats}")


def test_collect_respects_stop():
    """用户停止时 collect_until 立即中断并返回已采集部分（不跑到预算耗尽）"""
    llm = FakeLLM()
    llm.chunk_sleep = 0.03
    stop_flag = {"v": False}
    llm.stop = lambda: stop_flag["v"]
    cfg = Config()
    cfg.web_timeout = 300           # 预算设大，证明不是预算兜底而是 stop 生效

    class StopAfterOne(FakeEngine):
        _pages = 0
        def search(self, query):
            StopAfterOne._pages += 1
            if StopAfterOne._pages > 1:        # 第 1 页筛出后，第 2 页置停
                stop_flag["v"] = True
            return super().search(query)

    pm = [_Patch(ws, "LLMClient", lambda cfg=None, profile=None: llm),
          _Patch(ws, "DEFAULT_ENGINES", (StopAfterOne,)),
          _Patch(ws, "_check_page", lambda html, engine: html)]
    for p in pm:
        p.__enter__()
    try:
        out = ws.collect_until("僵尸毁灭工程", "与僵尸毁灭工程相关", 10, cfg,
                               should_stop=lambda: stop_flag["v"])
    finally:
        _uninstall(pm)
    check("停止信号被识别", "用户已停止" in out, out[:200])
    check("停止后返回已采集部分", ("有效 1 条" in out or "有效 2 条" in out), out[:120])


def test_collect_budget_inside_iteration():
    """单轮（含一次阻塞筛选）耗超预算时，应在该轮结束后即中断，而非拖到下一轮

    web_timeout 有 30s 下限（防止误配过小），故用假时钟把「单轮耗时」推过
    预算，验证预算检查在轮次内（筛选调用之后）确实生效。"""
    llm = FakeLLM()
    llm.chunk_sleep = 0.0
    cfg = Config()
    cfg.web_timeout = 90            # 默认预算
    # 假时钟：前几次调用给小值，筛选调用后的预算核查那次直接跳到 1000s，
    # 模拟「单轮因慢模型已超预算」——此时必须在该轮后中断，而非再爬一页
    clk = {"n": 0}
    def fake_monotonic():
        clk["n"] += 1
        return 1.0 if clk["n"] <= 3 else 1000.0
    pm = _install(llm) + [_Patch(ws.time, "monotonic", fake_monotonic)]
    for p in pm:
        p.__enter__()
    try:
        out = ws.collect_until("僵尸毁灭工程", "相关", 10, cfg, should_stop=lambda: False)
    finally:
        _uninstall(pm)
    check("单轮超预算即中断", "已达时间预算" in out, out[:200])
    check("只跑了首轮（未被慢筛选拖入下一轮）", "共尝试 1 轮" in out, out[:120])


if __name__ == "__main__":
    test_screen_heartbeat()
    test_collect_respects_stop()
    test_collect_budget_inside_iteration()
    print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败"
          + (f"  ✘ {FAIL}" if FAIL else "  全部通过"))
    raise SystemExit(1 if FAIL else 0)
