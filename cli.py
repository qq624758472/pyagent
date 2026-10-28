"""控制台交互模式（与 GUI 共用同一 Agent 核心）"""
import json
import sys

from core.agent import Agent
from core.config import CONFIG_FILE, Config
from core.llm_client import LLMClient, LLMError
from core.tools import TOOLS
from core.workspace import Workspace, WorkspaceError

BANNER = """
========================================
  PyAgent · 工作区智能助手（控制台模式）
  命令: /mount <路径>  /unmount  /reset
        /tools  /skills  /key <API_KEY>
        /models（拉取模型列表） /model <名称>
        /web [on|off]  /exit
========================================
"""


def render_event(ev):
    t = ev["type"]
    if t == "action":
        args = json.dumps(ev["arguments"], ensure_ascii=False)
        print(f"  [工具] {ev['name']} {args}")
    elif t == "observation":
        print(f"  [观察] {ev['content'][:600]}")
    elif t == "answer":
        print(f"\n[回答]\n{ev['content']}\n")
    elif t == "error":
        print(f"\n[错误] {ev['content']}\n")
    elif t == "format_error":
        print(f"  [重试] {ev['content']}")
    elif t == "redirect":
        print(f"  [纠偏] {ev['content']}")
    elif t == "cancelled":
        print("\n[已停止]\n")
    # keepalive / debug / delta：控制台不渲染（delta 已由 answer 汇总）


def _console_confirm(name, reason):
    """控制台高危操作确认（V23）。--task 单次模式不传 confirm，默认拒绝。"""
    try:
        ans = input(f"  [高危确认] {name}: {reason}\n  允许执行？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def run_task(agent, text, confirm=None):
    try:
        for ev in agent.run(text, progress=lambda m: print(f"  {m}", flush=True),
                            confirm=confirm):
            render_event(ev)
    except KeyboardInterrupt:
        print("\n[中断] 本次任务已取消")
    finally:
        sys.stdout.flush()


def run(config: Config, workspace=None, task=None):
    agent = Agent(LLMClient(config), workspace, max_iterations=config.max_iterations)
    print(BANNER)
    if workspace is not None:
        print(f"已挂载工作区: {workspace.root}")
    if not config.configured:
        print("[提示] 未配置 API Key：使用 /key <你的KEY>，或设置环境变量 AGENT_API_KEY")

    # 高危操作确认：confirm_dangerous 开启时 REPL 内联交互；--task 单次
    # 模式无交互，confirm=None → execute_tool 默认拒绝（受信自动化可关配置）
    confirm = _console_confirm if getattr(config, "confirm_dangerous", True) else None

    if task:                       # 单次任务模式（适合脚本/定时调用）
        run_task(agent, task, confirm=None)
        return

    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not text:
            continue
        if text.startswith("/"):
            _handle_command(text, agent, config)
            continue
        run_task(agent, text, confirm=confirm)


def _handle_command(cmd_line, agent, config):
    parts = cmd_line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if cmd == "/mount":
        if not arg:
            print("用法: /mount <目录路径>")
            return
        try:
            agent.set_workspace(Workspace(arg))
            print(f"已挂载工作区: {agent.workspace.root}")
        except WorkspaceError as e:
            print(f"[错误] {e}")
    elif cmd == "/unmount":
        agent.set_workspace(None)
        print("已卸载工作区")
    elif cmd == "/reset":
        agent.reset()
        print("对话历史已清空")
    elif cmd == "/tools":
        print("可用工具: " + ", ".join(TOOLS))
    elif cmd == "/skills":
        # V30 技能清单（对应 Jarvis 的 jarvis-rules-index）
        try:
            from core.skills import discover, bundle_files, tool_name
            found = discover(config, agent.workspace)
            if not found:
                print("未发现技能。技能 = 含 SKILL.md 的目录，放在：")
                print("  <工程>/skills/、<工作区>/.pyagent/skills/、~/.pyagent/skills/")
                print("  也兼容 .agents/skills/ 与 .claude/skills/")
                return
            print(f"可用技能（{len(found)} 个）:")
            for s in found:
                extra = bundle_files(s)
                print(f"  - {s.name}  [来源 {s.source}]  工具 {tool_name(s)}")
                print(f"      {s.description or '(无描述)'}")
                if extra:
                    print(f"      附带: {', '.join(extra[:6])}")
        except Exception as e:
            print(f"[错误] {e}")
    elif cmd == "/key":
        if not arg:
            print("用法: /key <API_KEY>")
            return
        config.api_key = arg
        config.save()
        agent.set_llm(LLMClient(config))
        print(f"API Key 已保存到 {CONFIG_FILE}")
    elif cmd == "/models":
        if not config.configured:
            print("[提示] 未配置 API Key：使用 /key <你的KEY>，或设置环境变量 AGENT_API_KEY")
            return
        print("正在拉取模型列表…")
        try:
            ids = LLMClient(config).list_models()
        except LLMError as e:
            print(f"[错误] {e}")
            return
        print(f"可用模型（{len(ids)} 个）:")
        for i in ids:
            mark = "  ← 当前" if i == config.model else ""
            print(f"  - {i}{mark}")
        if config.model not in ids:
            print(f"[提示] 当前模型 {config.model} 不在列表中，可用 /model <名称> 切换")
    elif cmd == "/model":
        if not arg:
            print("用法: /model <模型名>（先用 /models 查看可用模型）")
            return
        config.model = arg
        config.save()
        agent.set_llm(LLMClient(config))
        print(f"模型已切换为 {arg}，已保存")
    elif cmd == "/web":
        if arg in ("on", "off", "1", "0", "true", "false"):
            config.web_enabled = arg in ("on", "1", "true")
        elif not arg:
            config.web_enabled = not config.web_enabled     # 无参数=切换
        else:
            print("用法: /web [on|off]")
            return
        config.save()
        agent.set_llm(LLMClient(config))          # 重刷 system prompt 工具清单
        print(f"联网采集已{'启用' if config.web_enabled else '关闭'}，已保存")
    elif cmd in ("/exit", "/quit"):
        raise SystemExit(0)
    else:
        print("未知命令。可用: /mount /unmount /reset /tools /skills /key "
              "/models /model /web /exit")
