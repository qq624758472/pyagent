"""真实小项目验收：让本地 Agent 自主创建、测试任务管理 CLI，然后独立验收。

运行: python -X utf8 tests/real_project_eval.py --out C:/Temp/project_eval.json
不保存临时工作区；使用 --keep 保留产物以排查失败。
此脚本不会在任务工作区预先放入项目文件或测试代码。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.agent import Agent
from core.config import Config
from core.llm_client import LLMClient
from core.workspace import Workspace

TASK = (
    "请你独立实现并验证一个小型 Python 项目：无第三方依赖的任务管理命令行工具。"
    "在当前全新工作区创建 app.py、test_app.py 和 README.md 三个文件。"
    "运行形式为 python app.py add 任务内容、python app.py list、"
    "python app.py done 数字ID；支持跨进程持久化到工作区 tasks.json，"
    "新增任务 ID 从 1 递增，list 输出任务ID、内容和完成状态，"
    "done 能将任务标记为完成，非法 ID 必须非零退出。"
    "使用环境变量 TASK_DB 可以重定向 JSON 数据文件路径，方便自动测试。"
    "test_app.py 用 unittest 和 subprocess 测试 add/list/done、持久化、非法 ID。"
    "注意 unittest.TestCase.run 是框架保留方法，不得覆盖；"
    "测试中的子进程辅助方法命名 run_cli，各测试必须隔离 JSON 文件。"
    "README.md 必须包含完整可执行的 add/list/done 使用示例，不能写占位符。"
    "请按步骤实际创建三个文件，"
    "必须使用 run_python 执行 test_app.py；如果失败，修改代码并重新运行，"
    "直到测试通过后再总结真实测试结果。不要只给我代码块或运行方法。"
)

def _check(workdir):
    app = workdir / "app.py"
    test_file = workdir / "test_app.py"
    readme = workdir / "README.md"
    verdict = {"app_exists": app.is_file(), "tests_exist": test_file.is_file(),
               "readme_exists": readme.is_file()}
    if not app.is_file():
        return verdict
    readme_text = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else ""
    verdict["readme_ok"] = (len(readme_text) >= 120 and
                           all(w in readme_text for w in ("add", "list", "done")))
    db = workdir / "grader_data.json"
    env = dict(os.environ, TASK_DB=str(db), PYTHONUTF8="1")
    def call(*args):
        try:
            r = subprocess.run([sys.executable, str(app), *args], cwd=workdir,
                               env=env, capture_output=True, text=True, timeout=12,
                               encoding="utf-8", errors="replace")
            return {"code": r.returncode, "out": (r.stdout + r.stderr)[:450]}
        except Exception as exc:
            return {"code": -999, "out": str(exc)[:250]}
    first = call("add", "buy milk")
    second = call("add", "read book")
    listed = call("list")
    done = call("done", "1")
    after = call("list")
    invalid = call("done", "99999")
    verdict["external_calls"] = {"add1": first, "add2": second, "list": listed,
                                 "done": done, "after": after, "invalid": invalid}
    before_lines = [ln for ln in listed["out"].splitlines() if "buy milk" in ln]
    after_lines = [ln for ln in after["out"].splitlines() if "buy milk" in ln]
    verdict["external_pass"] = (
        first["code"] == 0 and second["code"] == 0 and listed["code"] == 0
        and "buy milk" in listed["out"] and "read book" in listed["out"]
        and done["code"] == 0 and after["code"] == 0
        and "buy milk" in after["out"] and "read book" in after["out"]
        and before_lines and after_lines and before_lines[0] != after_lines[0]
        and invalid["code"] != 0 and db.is_file()
    )
    if test_file.is_file():
        try:
            r = subprocess.run([sys.executable, str(test_file)], cwd=workdir,
                               capture_output=True, text=True, timeout=35,
                               encoding="utf-8", errors="replace", env=env)
            verdict["test_file_exit"] = r.returncode
            verdict["test_file_output"] = (r.stdout + r.stderr)[-1200:]
        except Exception as exc:
            verdict["test_file_exit"] = -999
            verdict["test_file_output"] = str(exc)
    return verdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "pyagent_project_eval.json"))
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--iterations", type=int, default=36)
    args = parser.parse_args()
    cfg = Config.load()
    cfg.protocol = "native"
    cfg.temperature = 0.3
    cfg.skills_enabled = False
    cfg.web_enabled = False
    cfg.max_iterations = args.iterations
    workdir = Path(tempfile.mkdtemp(prefix="pyagent_project_"))
    print("WORKSPACE", workdir, "MODEL", cfg.model, flush=True)
    events = []
    start = time.monotonic()
    agent = Agent(LLMClient(cfg), Workspace(workdir), max_iterations=cfg.max_iterations)
    try:
        for ev in agent.run(TASK, confirm=lambda name, reason: name == "run_python"):
            typ = ev["type"]
            if typ in ("action", "observation", "answer", "error", "format_error", "redirect"):
                summary = (str(ev.get("name", "")) + " " +
                           str(ev.get("content", ev.get("arguments", ""))))
                print("EVENT", typ, summary[:350].replace("\n", " "), flush=True)
                events.append({"type": typ, "name": ev.get("name"), "summary": summary[:650]})
    except Exception as exc:
        print("HARNESS_ERROR", repr(exc), flush=True)
        events.append({"type": "harness_error", "summary": repr(exc)})
    verdict = _check(workdir)
    action_names = [e["name"] for e in events if e["type"] == "action"]
    observations = [e["summary"] for e in events if e["type"] == "observation"]
    verdict["agent_ran_tests"] = ("run_python" in action_names
        and any("test_app.py" in e["summary"] for e in events if e["type"] == "action"))
    verdict["agent_test_pass_observed"] = any(
        "OK" in x and "FAILED" not in x and "错误:" not in x
        for x in observations if "Ran " in x or "OK" in x)
    verdict["agent_answer"] = any(e["type"] == "answer" for e in events)
    verdict["agent_errors"] = [e["summary"] for e in events if e["type"] == "error"]
    verdict["tools"] = action_names
    verdict["duration_seconds"] = round(time.monotonic() - start, 1)
    verdict["model"] = cfg.model
    verdict["workspace"] = str(workdir)
    verdict["pass"] = all((verdict.get("app_exists"), verdict.get("tests_exist"),
                           verdict.get("readme_ok"), verdict.get("external_pass"),
                           verdict.get("test_file_exit") == 0,
                           verdict["agent_ran_tests"],
                           verdict["agent_test_pass_observed"], verdict["agent_answer"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"verdict": verdict, "events": events},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT", json.dumps({k: v for k, v in verdict.items()
        if k not in ("external_calls", "test_file_output")}, ensure_ascii=False),
        "REPORT", out, flush=True)
    if not args.keep:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if verdict["pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
