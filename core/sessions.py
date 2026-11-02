"""会话持久化：工程目录 sessions/ 下每会话一个 json 文件

文件格式（sessions/<id>.json）：
{
  "id": "20261021-135603-ab12",
  "title": "首条用户消息截断 40 字",
  "created": "2026-10-21 13:56:03",
  "workspace": "E:/docs/my_kb",            # 该会话挂载的工作区（空=无）
  "messages": [["user", "..."], ["assistant", "..."], ...]   # 不含 system（重载时重建）
}

OBSERVATION 类 user 消息照存（恢复时折叠为"历史执行过程"）。
系统提示词与工具清单按当前配置重建，保证升级后语义一致。
"""
import json
import re
import secrets
import time
from pathlib import Path

from core.config import PROJECT_ROOT
from core.log import get_logger

log = get_logger("core.sessions")

SESSIONS_DIR = PROJECT_ROOT / "sessions"
MAX_SESSIONS = 50            # 最多保留会话数（超出删最旧）

_SID_RE = re.compile(r"^[\w-]{1,64}$")     # 会话 id 合法字符（防路径逃逸）


def new_session_id() -> str:
    """时间戳 + 随机后缀：同进程/同秒多次创建不碰撞"""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def session_path(sid: str) -> Path:
    return SESSIONS_DIR / f"{sid}.json"


def save_session(sid: str, messages, workspace: str, title: str):
    """messages: Agent.messages（system 消息自动剔除）"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 2,
        "id": sid,
        "title": (title or "未命名会话")[:40],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace": workspace or "",
        # 保留 OpenAI 对话协议的完整字段（tool_calls/tool_call_id/name 等）。
        # 旧版 [role, content] 格式仍可读取，由 UI 层兼容迁移。
        "messages": [dict(m) for m in messages
                     if isinstance(m, dict) and m.get("role") != "system"],
    }
    # 置顶标记跨自动归档保留（每轮结束都会重写文件，不能丢）
    prev = load_session(sid) or {}
    if prev.get("pinned"):
        data["pinned"] = True
    path = session_path(sid)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    _prune()
    log.info("会话已保存: %s (%d 条消息) → %s",
             data["title"][:30], len(data["messages"]), path.name)


def load_session(sid: str):
    """返回 dict 或 None"""
    path = session_path(sid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning("会话文件损坏，已跳过: %s", path.name)
        return None


def list_sessions() -> list:
    """[{id, title, workspace, pinned, mtime}] 按修改时间降序（置顶优先）"""
    if not SESSIONS_DIR.exists():
        return []
    out = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"id": d.get("id", p.stem),
                        "title": d.get("title", p.stem),
                        "workspace": d.get("workspace", ""),
                        "pinned": bool(d.get("pinned")),
                        "mtime": p.stat().st_mtime})
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return sorted(out, key=lambda s: (s["pinned"], s["mtime"]), reverse=True)


def set_session_pinned(sid: str, pinned: bool) -> bool:
    """置顶/取消置顶（V25）。保留 messages 等其余字段。"""
    if not _SID_RE.match(str(sid or "")):
        return False
    path = session_path(sid)
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if pinned:
            d["pinned"] = True
        else:
            d.pop("pinned", None)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        log.info("会话%s: %s", "置顶" if pinned else "取消置顶", sid)
        return True
    except (OSError, json.JSONDecodeError) as e:
        log.warning("会话置顶写入失败: %s | %s", sid, e)
        return False


def filter_sessions(sessions: list, mode: str) -> list:
    """侧栏筛选 chip 的数据层（V25）：
    all=全部 / active=24h 内有活动 / pinned=置顶 / archived=超过 7 天无活动"""
    now = time.time()
    if mode == "active":
        return [s for s in sessions if now - s["mtime"] <= 86400]
    if mode == "pinned":
        return [s for s in sessions if s.get("pinned")]
    if mode == "archived":
        return [s for s in sessions if now - s["mtime"] > 7 * 86400]
    return list(sessions)


def latest_session_id() -> str:
    ss = list_sessions()
    return ss[0]["id"] if ss else ""


def delete_session(sid: str) -> bool:
    """删除会话文件（V24，不可恢复）。id 非法/文件不存在/删除失败返回 False。"""
    if not _SID_RE.match(str(sid or "")):
        log.warning("拒绝删除：非法会话 id %r", sid)
        return False
    path = session_path(sid)
    try:
        if path.exists():
            path.unlink()
        log.info("会话已删除: %s", sid)
        return True
    except OSError as e:
        log.warning("会话删除失败: %s | %s", sid, e)
        return False


def _prune():
    """超出 MAX_SESSIONS 删除最旧（置顶会话豁免）"""
    files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for p in files[:-MAX_SESSIONS] if len(files) > MAX_SESSIONS else []:
        if _is_pinned(p):
            continue        # 置顶会话不参与自动清理
        try:
            p.unlink()
            log.info("清理旧会话: %s", p.name)
        except OSError:
            pass


def _is_pinned(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("pinned"))
    except (json.JSONDecodeError, OSError):
        return False
