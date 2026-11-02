"""真实模型端到端 TCP 联调验收。

源文件取自用户现有 agent测试目录，复制到全新临时工作区，
让 GPT-OSS-20B 调用工具自主修复并验证，绝不修改原始文件。
用法: python -X utf8 -u tests/real_tcp_eval.py --keep
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.agent import Agent
from core.config import Config
from core.llm_client import LLMClient
from core.tools import execute_tool
from core.workspace import Workspace
from core import process_tools

TASK = (
    "你自己打开 TCP 服务器接收，并用 TCP 客户端进行发送，然后进行调试。"
    "工作区里已有 tcp_server.py 和 tcp_client.py；必须使用它们完成回环通信。"
    "如果有错误，请修改这两个文件并用 tcp_echo_test 真正验证发送 hello、"
    "收到 hello 并确认客户端本身能正确接收。完成后汇报实际工具测试结果，"
    "不能只给手动运行步骤。"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "pyagent_real_tcp_eval.json"))
    parser.add_argument("--iterations", type=int, default=17)
    args = parser.parse_args()

    cfg = Config.load()
    cfg.protocol = "native"
    cfg.temperature = 0.3
    cfg.max_iterations = args.iterations
    cfg.web_enabled = False
    cfg.skills_enabled = False
    temp = Path(tempfile.mkdtemp(prefix="pyagent_tcp_eval_"))
    src = ROOT / "agent测试目录"
    for name in ("tcp_server.py", "tcp_client.py"):
        if (src / name).is_file():
            shutil.copy2(src / name, temp / name)
    ws = Workspace(temp)
    print("REAL_TCP_MODEL", cfg.model, "WORKSPACE", temp, flush=True)
    start = time.monotonic()
    actions = []
    observations = []
    terminal = []
    agent = Agent(LLMClient(cfg), ws, max_iterations=cfg.max_iterations)
    try:
        for ev in agent.run(TASK, confirm=lambda name, why: name in (
                "tcp_echo_test", "start_process", "run_python")):
            t = ev["type"]
            if t in ("action", "observation", "answer", "error", "redirect"):
                summary = str(ev.get("content", ev.get("arguments", "")))[:420]
                print("EVENT", t, ev.get("name", ""), summary.replace("\n", " "), flush=True)
                if t == "action":
                    actions.append(ev.get("name", ""))
                elif t == "observation":
                    observations.append(summary)
                elif t in ("answer", "error"):
                    terminal.append((t, summary))
    except Exception as exc:
        terminal.append(("harness_error", repr(exc)))
    finally:
        # 回收任何模型启动后未停止的服务，避免污染后续端口验收。
        for key, item in list(process_tools._processes.items()):
            if item[0] == str(ws.root):
                execute_tool("stop_process", ws, cfg, {"process_id": key})
    external = execute_tool("tcp_echo_test", ws, cfg,
        {"server_path": "tcp_server.py", "client_path": "tcp_client.py",
         "message": "hello", "port": 12345}, confirm=lambda name, why: True)
    verdict = {
        "model": cfg.model, "workspace": str(temp),
        "elapsed_s": round(time.monotonic()-start,1),
        "tools": actions,
        "agent_verified_tcp": any("TCP联调通过" in x for x in observations),
        "agent_answer": bool(terminal and terminal[-1][0] == "answer"),
        "terminal": terminal[-3:],
        "external_tcp": external[:800],
        "external_pass": "TCP联调通过" in external,
        "files": {name: (temp/name).is_file() for name in ("tcp_server.py", "tcp_client.py")}
    }
    verdict["pass"] = (verdict["agent_verified_tcp"] and verdict["agent_answer"]
                       and verdict["external_pass"] and all(verdict["files"].values()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(verdict, ensure_ascii=False), flush=True)
    if not args.keep:
        shutil.rmtree(temp, ignore_errors=True)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
