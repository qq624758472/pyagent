"""真实端点评测：跑一组任务，统计完成率 / 轮数 / 工具正确性 / 答案质量

用法（工程根目录执行）：
    python tests/real_eval.py                 # 用 config.json 的端点与配置
    python tests/real_eval.py --temp 0.3      # 临时覆盖 temperature（不改配置）
    python tests/real_eval.py --protocol text # 强制文本协议（对比用）
    python tests/real_eval.py --tasks T1,T2   # 只跑部分任务

评测产物：控制台表格 + tests/eval_report.json（便于横向比较不同参数）
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent                       # noqa: E402
from core.config import Config                     # noqa: E402
from core.llm_client import LLMClient              # noqa: E402
from core.log import setup_logging                 # noqa: E402
from core.workspace import Workspace               # noqa: E402

TASKS = [
    {"id": "T1", "name": "列目录",
     "task": "列出工作区里的文件，告诉我有哪些",
     "expect_tools": ["list_files"], "expect_answer": True},
    {"id": "T2", "name": "写脚本并运行",
     "task": "写一个 Python 脚本 sum.py 计算 1 到 100 的和并运行它，把结果告诉我",
     "expect_tools": ["write_file", "run_python"], "expect_answer": True},
    {"id": "T3", "name": "统计行数写文件",
     "task": "统计工作区所有 .py 文件的总行数，把结果写入 report.txt",
     "expect_tools": ["write_file"], "expect_answer": True},
    {"id": "T4", "name": "常识问答",
     "task": "你知道僵尸毁灭工程吗？简单介绍一下",
     "expect_tools": [], "expect_answer": True},
    # V30 技能：任务描述里含明显触发词（"总结报告"），模型应主动加载
    # report-writer 技能再按规范写——考察原生协议下模型会不会选 skill_* 工具
    {"id": "T5", "name": "技能触发",
     "task": "帮我写一份关于本周项目进展的总结报告",
     "expect_tools": ["skill_report_writer"], "expect_answer": True},
]

# 答案里泄漏模型思考的典型特征（英文自述式）
THINK_LEAK = re.compile(
    r"The user asks|We need to|I should|Let'?s |we must|according to the protocol"
    r"|we need to answer|I will (call|answer)|probably|let me", re.I)


def make_workspace(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "demo.py").write_text(
        "def add(a, b):\n    return a + b\n\n\nprint(add(1, 2))\n", encoding="utf-8")
    (root / "notes.txt").write_text("这是一份测试笔记。\n", encoding="utf-8")
    (root / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return root


def score_answer(text):
    """答案质量启发式评分"""
    if not text:
        return {"has_answer": False, "chinese_ratio": 0.0, "think_leak": False,
                "len": 0}
    cn = sum(1 for ch in text if "一" <= ch <= "鿿")
    return {"has_answer": True,
            "chinese_ratio": round(cn / max(len(text), 1), 2),
            "think_leak": bool(THINK_LEAK.search(text)),
            "len": len(text)}


def run_task(cfg, task, idx):
    ws_dir = make_workspace(Path(tempfile.mkdtemp()) / "ws")
    ws = Workspace(ws_dir)
    agent = Agent(LLMClient(cfg), ws, max_iterations=cfg.max_iterations)
    tools_used, answer, rounds = [], "", 0
    t0 = time.monotonic()
    events = list(agent.run(task["task"]))
    dt = time.monotonic() - t0
    for ev in events:
        if ev["type"] == "action":
            tools_used.append(ev["name"])
        elif ev["type"] == "answer":
            answer = ev["content"]
    rounds = len([1 for ev in events if ev["type"] in ("action", "answer")]) or 1
    q = score_answer(answer)
    exp = task["expect_tools"]
    tools_ok = all(t in tools_used for t in exp)
    return {
        "id": task["id"], "name": task["name"], "task": task["task"],
        "tools_used": tools_used, "tools_expected": exp, "tools_ok": tools_ok,
        "answer": answer[:300], "answer_score": q,
        "events": [e["type"] for e in events],
        "seconds": round(dt, 1), "rounds": rounds,
        "protocol": "native" if agent._native_mode() else "text",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--protocol", default=None, choices=["auto", "native", "text"])
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default=str(ROOT / "tests" / "eval_report.json"))
    args = ap.parse_args()

    cfg = Config.load()
    if args.temp is not None:
        cfg.temperature = args.temp
    if args.protocol:
        cfg.protocol = args.protocol
    setup_logging(cfg.log_level)
    if not cfg.configured:
        print("未配置 API Key", file=sys.stderr)
        return 1

    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks = [t for t in TASKS if not wanted or t["id"] in wanted]
    print(f"端点 {cfg.base_url}  模型 {cfg.model}  temp={cfg.temperature}  "
          f"protocol={cfg.protocol}  任务数={len(tasks)}")
    print("-" * 96)
    results = []
    for t in tasks:
        r = run_task(cfg, t, 0)
        results.append(r)
        sc = r["answer_score"]
        print(f"[{r['id']}] {r['name']:<12} {r['seconds']:>6.1f}s 轮次≈{r['rounds']} "
              f"工具={','.join(r['tools_used']) or '（无）':<22} "
              f"工具符合={'✔' if r['tools_ok'] else '✘'} "
              f"有答案={'✔' if sc['has_answer'] else '✘'} "
              f"中文={sc['chinese_ratio']:.0%} 思考泄漏={'⚠' if sc['think_leak'] else '无'}")
        if sc["has_answer"]:
            print(f"      答案: {r['answer'][:120]!r}")
    print("-" * 96)
    n = len(results)
    done = sum(1 for r in results if r["answer_score"]["has_answer"])
    tools_ok = sum(1 for r in results if r["tools_ok"])
    leak = sum(1 for r in results if r["answer_score"]["think_leak"])
    print(f"完成率(有答案) {done}/{n}   工具符合预期 {tools_ok}/{n}   "
          f"思考泄漏 {leak}/{n}   平均耗时 {sum(r['seconds'] for r in results)/max(n,1):.1f}s")
    Path(args.out).write_text(json.dumps(
        {"config": {"model": cfg.model, "temperature": cfg.temperature,
                    "protocol": cfg.protocol}, "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
