"""日志：统一初始化 + 模块 logger 获取

日志文件 <工程目录>/logs/pyagent.log（按代码所在位置定位，不随启动
目录变化）。1MB × 3 自动轮转。默认 INFO 级即可完整重建运行现场
（配置快照/每轮模型交互/工具调用/引擎切换/异常堆栈），用户可直接把
文件发给开发者分析。DEBUG 级追加完整 messages 与流式细节。

API Key 在日志中自动脱敏（仅尾 4 位可见）。"""
import json
import logging
import logging.handlers
import os
import re
from dataclasses import asdict, fields
from pathlib import Path
from core.runtime import DATA_ROOT

# 工程根目录 = core/ 的上级（放工程目录下，用户可直接找到）
PROJECT_ROOT = DATA_ROOT
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "pyagent.log"
MAX_BYTES = 1024 * 1024          # 单文件 1MB
BACKUP_COUNT = 3                 # 保留 3 个历史 → 最多 4MB

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """初始化根日志（幂等）。main.py / cli.py 入口各调一次"""
    global _configured
    root = logging.getLogger()
    if _configured:
        root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        return root
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
        encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(handler)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    _configured = True
    root.info("=" * 60)
    root.info("pyagent 启动 | 日志级别 %s | PID %d | Python %s",
              str(level).upper(), os.getpid(),
              ".".join(str(v) for v in __import__("sys").version_info[:3]))
    root.info("日志文件: %s（超过 %dKB 自动轮转，保留 %d 份历史）",
              LOG_FILE, MAX_BYTES // 1024, BACKUP_COUNT)
    return root


def get_logger(name: str) -> logging.Logger:
    """业务模块取 logger：core.agent → 'agent'，core.web_search → 'web'"""
    short = {"core.agent": "agent", "core.llm_client": "llm",
             "core.tools": "tools", "core.web_search": "web",
             "core.workspace": "workspace", "ui.main_window": "ui",
             "cli": "cli"}.get(name, name.split(".")[-1])
    return logging.getLogger(short)


# ---------------- Key / 敏感信息脱敏 ----------------
_KEY_RE = re.compile(r'(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9_\-\.]{6,})')


def mask(text: str) -> str:
    """API Key 脱敏：保留前 3 后 4，中间折叠为 ****"""
    return _KEY_RE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[-4:],
                       str(text))


def config_snapshot(cfg) -> str:
    """配置快照（脱敏）——启动时记一条，分析环境差异用。

    用 dataclasses.asdict 递归转换（profiles 里是嵌套 dataclass，
    直接 json.dumps 会 TypeError——V14 起快照一直失败的根因）。"""
    try:
        d = {}
        for f in fields(cfg):
            v = getattr(cfg, f.name)
            if f.name == "profiles" and isinstance(v, dict):
                # dict[str, ModelProfile] → dict[str, dict]（脱敏需要可变 dict）
                d[f.name] = {k: dict(model=pv.model, base_url=pv.base_url,
                                     api_key=pv.api_key)
                             for k, pv in v.items()}
            else:
                try:
                    d[f.name] = asdict(v) if hasattr(v, "__dataclass_fields__") else v
                except TypeError:
                    d[f.name] = repr(v)
    except Exception:
        return "配置快照生成失败"
    if d.get("api_key"):
        d["api_key"] = d["api_key"][:3] + "****" + d["api_key"][-4:]
    for p in (d.get("profiles") or {}).values():
        if isinstance(p, dict) and p.get("api_key"):
            p["api_key"] = p["api_key"][:3] + "****" + p["api_key"][-4:]
    return " ".join(f"{k}={v!r}" for k, v in sorted(d.items()))


def brief(text: str, limit: int = 200) -> str:
    """单行化+截断：多行内容压成一行日志字段"""
    s = " ".join(str(text).split())
    return s[:limit] + ("…" if len(s) > limit else "")


def oneline_json(obj) -> str:
    """JSON 单行化（日志友好）"""
    try:
        return json.dumps(obj, ensure_ascii=False)[:2000]
    except (TypeError, ValueError):
        return brief(repr(obj))
