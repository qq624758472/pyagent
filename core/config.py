"""配置管理：优先级 环境变量 > 工程目录 config.json > 默认值

配置/会话/工作区全部保存在工程目录下（便携式，拷走工程=带走全部状态）。
旧版 ~/.pyagent/config.json 首次启动自动迁移（迁移后原文件保留不动）。

profiles：按角色独立配置模型（名称/Base URL/API Key），空值回落主配置
  - main  思考模型（ReAct 主循环）
  - kb    知识库检索模型（kb_search 工具）
  - web   联网搜索模型（web_collect 采集与筛选）
"""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.log import get_logger
from core.runtime import DATA_ROOT

log = get_logger("core.config")

# 工程根目录 = core/ 的上级（配置随工程走，不随启动目录/用户变化）
PROJECT_ROOT = DATA_ROOT
CONFIG_DIR = PROJECT_ROOT
CONFIG_FILE = PROJECT_ROOT / "config.json"
_LEGACY_FILE = Path.home() / ".pyagent" / "config.json"   # 旧版位置（迁移源）

PROFILE_KEYS = ("main", "kb", "web")

# 手改 config.json 时常把 true 写成 "true"、2.5 写成 "2.5"——按字段类型归一
_NUMERIC_FIELDS = {"temperature": float, "max_iterations": int,
                   "web_timeout": int, "ui_scale": float, "llm_read_timeout": int}
_BOOL_FIELDS = {"allow_shell", "python_enabled", "web_enabled",
                "debug_mode", "confirm_dangerous", "skills_enabled", "llm_use_system_proxy"}


@dataclass
class ModelProfile:
    model: str = ""       # 留空 = 使用主配置的 model
    base_url: str = ""    # 留空 = 使用主配置的 base_url
    api_key: str = ""     # 留空 = 使用主配置的 api_key


def _set_field(cfg: "Config", key: str, value) -> None:
    """按字段类型归一后写入（数值/布尔字段容忍字符串写法）"""
    if key in _NUMERIC_FIELDS:
        try:
            setattr(cfg, key, _NUMERIC_FIELDS[key](value))
        except (TypeError, ValueError):
            log.warning("配置项 %s 值非法(%r)，保持默认值", key, value)
        return
    if key in _BOOL_FIELDS:
        if isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on")
        setattr(cfg, key, bool(value))
        return
    setattr(cfg, key, value)


@dataclass
class Config:
    api_key: str = ""
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"   # 智谱 GLM，OpenAI 兼容
    model: str = "glm-4-flash"
    screen_model: str = ""       # [兼容字段] 迁移时转入 profiles.web.model
    temperature: float = 0.7
    llm_read_timeout: int = 300      # 模型单次读取等待秒数，实际设置限制30~1800
    llm_use_system_proxy: bool = True  # 保持系统路由；可在设置中为模型接口单独选择直连
    max_iterations: int = 8       # 进度检查间隔；不限制总轮数，无进展预算至少16轮
    allow_shell: bool = False
    python_enabled: bool = True     # run_python 工具开关（默认启用：本地直接执行脚本）
    web_enabled: bool = True       # 联网采集总开关（关闭后 web_collect 工具隐藏）
    web_timeout: int = 90          # 联网采集总时间预算（秒，下限 30）
    log_level: str = "INFO"        # 日志级别：INFO（默认，可完整重建现场）/ DEBUG（含完整消息）
    theme: str = "dark"          # dark / light（界面主题，仿 Reasonix）
    ui_scale: float = 1.0        # 界面缩放（Ctrl+滚轮 0.7~2.5，默认 100%）
    debug_mode: bool = False     # 调试模式：每轮显示模型原始返回+解析过程（默认收起在消息卡内）
    confirm_dangerous: bool = True   # 高危操作（递归删除/shell/本地 Python）执行前需用户确认
    protocol: str = "auto"           # 工具协议：auto（优先原生 function calling，端点不支持自动回退）/ native / text
    skills_enabled: bool = True      # 技能系统总开关（V30）：关闭后不暴露 skill_* 工具
    last_workspace: str = ""     # 上次挂载的工作区（启动恢复用，挂载/卸载即时写）
    profiles: dict = field(default_factory=lambda: {
        k: ModelProfile() for k in PROFILE_KEYS})

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        cls._migrate_legacy()
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if k == "profiles" and isinstance(v, dict):
                        for pk, pv in v.items():
                            if pk in PROFILE_KEYS and isinstance(pv, dict):
                                cfg.profiles[pk] = ModelProfile(
                                    model=str(pv.get("model", "") or ""),
                                    base_url=str(pv.get("base_url", "") or ""),
                                    api_key=str(pv.get("api_key", "") or ""))
                        continue
                    if k == "screen_model":        # 兼容：转为 web 角色
                        cfg.screen_model = v or ""
                        continue
                    if hasattr(cfg, k):
                        _set_field(cfg, k, v)
            except Exception as e:
                # 不再静默吞掉：配置损坏时用户至少能从日志知道为什么 Key/模型丢了
                log.warning("配置文件解析失败，本次使用默认值/环境变量: %s | %s: %s",
                            CONFIG_FILE, type(e).__name__, e)
        # 旧 screen_model → profiles.web.model（仅当 web 角色未单独配置）
        if cfg.screen_model and not cfg.profiles["web"].model:
            cfg.profiles["web"].model = cfg.screen_model
        env_map = {"AGENT_API_KEY": "api_key", "AGENT_BASE_URL": "base_url",
                   "AGENT_MODEL": "model", "AGENT_MAX_ITERATIONS": "max_iterations",
                   "AGENT_ALLOW_SHELL": "allow_shell"}
        for env, attr in env_map.items():
            val = os.environ.get(env)
            if not val:
                continue
            if attr == "max_iterations":            # 环境变量是字符串，做类型归一
                try:
                    val = int(val)
                except ValueError:
                    continue
            elif attr == "allow_shell":
                val = val.lower() in ("1", "true", "yes", "on")
            setattr(cfg, attr, val)
        return cfg

    @classmethod
    def _migrate_legacy(cls):
        """首次启动：旧 ~/.pyagent/config.json 自动复制到工程目录（原文件保留）"""
        if not CONFIG_FILE.exists() and _LEGACY_FILE.exists():
            try:
                CONFIG_FILE.write_text(
                    _LEGACY_FILE.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

    def save(self):
        CONFIG_FILE.write_text(json.dumps(asdict(self), ensure_ascii=False,
                                          indent=2), encoding="utf-8")

    def profile(self, key: str) -> ModelProfile:
        p = self.profiles.get(key)
        return p if p is not None else ModelProfile()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)
