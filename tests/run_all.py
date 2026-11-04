"""一键跑全部测试套件（桩 / 离屏 Qt，不需要真实模型）

    python tests/run_all.py

真实端点评测（会调用你的 LLM 服务器，耗时数分钟）需单独运行：
    python tests/real_eval.py

退出码：0 = 全部通过；1 = 有失败套件。
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

# (脚本, 是否需要离屏 Qt)
SUITES = [
    ("test_robustness.py", False),          # 空回复终止 / 高危确认 / 参数校验
    ("test_reasoning_fallback.py", False),  # reasoning_content 兜底（文本协议）
    ("test_protocol_nudge.py", False),      # 协议追问闭环（文本协议）
    ("test_native_protocol.py", False),     # 原生 function calling 协议
    ("test_agent_quality.py", False),        # 小模型：历史/工具调用完整性与回退
    ("test_autonomous_loop.py", False),      # 小模型自主编码：失败去重/测试闭环
    ("test_progress_loop.py", False),        # 超过8轮继续修复/双协议验收/无进展识别
    ("test_research_progress.py", False),    # 重复分页读取/资料记录/文档阶段推进/搜索范围
    ("test_word_documents.py", False),       # 真正DOCX生成/结构检查/防止伪文档交付
    ("test_tcp_execution.py", False),       # TCP 服务器生命周期/客户端联调/完成门禁
    ("test_confirm_flow.py", True),        # 高危操作确认：迟到点击拒绝/正常放行
    ("test_skills.py", False),              # 技能系统：发现/披露/工具/安全/集成
    ("test_skill_recovery.py", False),      # 技能名/工具名归一/重复加载/反馈回传/停滞恢复
    ("test_web_collect.py", False),         # 联网采集：长任务心跳 / 可中断 / 预算
    ("test_session_delete.py", True),       # 会话删除（离屏 Qt）
    ("test_session_resume.py", True),       # 完整工具历史/交接记录/重启续做/旧格式迁移
    ("test_ui_features.py", True),          # 界面功能补全（离屏 Qt）
    ("test_markdown_ui.py", True),          # Markdown流式/表格/嵌套列表/去重/滚动/状态
    ("test_request_lifecycle.py", True),    # 慢SSE心跳/及时取消/真实超时/看门狗/检查点
    ("test_network_settings.py", True),     # 网络超时设置/代理隔离/缩短历史安全重试
    ("test_stream_recovery.py", True),      # 响应中途断开/丢弃残片/自动续做/不重复写入
    ("test_v31_fixes.py", False),           # V31：内容搜索 + 重复调用熔断
    ("test_v311_native.py", False),         # V31.1：同工具刷屏熔断 / 空回合不烧迭代
    ("test_v312_fixes.py", True),           # V31.2：参数容错 / 思考期心跳 / 文本块高度
]

# 末行形如 "结果: 25 通过, 0 失败"
_RESULT_HINT = "结果:"


def _parse_counts(stdout: str):
    """从测试脚本的最后一行统计里取 (通过, 失败)；解析不出来返回 None

    只认 "结果: N 通过, M 失败" 这一种格式；用 returncode 做主判据，
    这里的计数仅用于展示，解析失败不影响判定。"""
    for line in reversed(stdout.strip().splitlines()):
        if _RESULT_HINT not in line:
            continue
        nums = line.replace(",", " ").split()
        passed = failed = None
        for i, tok in enumerate(nums):
            if tok == "通过" and i > 0:
                try:
                    passed = int(nums[i - 1])
                except ValueError:
                    pass
            elif tok == "失败" and i > 0:
                try:
                    failed = int(nums[i - 1])
                except ValueError:
                    pass
        if passed is not None and failed is not None:
            return passed, failed
    return None


def main():
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONUTF8"] = "1"          # 子进程统一 UTF-8，避免 Windows GBK 解码炸
    env["PYTHONIOENCODING"] = "utf-8"
    total_pass = total_fail = 0
    failed = []
    for script, needs_qt in SUITES:
        path = HERE / script
        if not path.exists():
            print(f"[跳过] {script} 不存在")
            continue
        r = subprocess.run([PY, str(path)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env=env, cwd=str(ROOT))
        counts = _parse_counts(r.stdout or "")
        passed, fails = counts if counts else (0, 0)
        # 主判据是退出码：脚本里 sys.exit(1) 表示有断言失败或崩溃
        ok = (r.returncode == 0) and (counts is None or fails == 0)
        total_pass += passed
        total_fail += fails
        if not ok:
            failed.append(script)
        mark = "[OK]  " if ok else "[FAIL]"
        print(f"{mark} {script:<32} 通过 {passed} / 失败 {fails}")
        if not ok:
            out = (r.stdout or "").strip().splitlines()
            err = (r.stderr or "").strip().splitlines()
            for ln in out[-12:]:
                print("    " + ln)
            for ln in err[-6:]:
                print("    " + ln)
            if r.returncode != 0:
                print(f"    (退出码 {r.returncode})")
    print("-" * 64)
    print(f"合计：通过 {total_pass}  失败 {total_fail}"
          + (f"   失败套件: {', '.join(failed)}" if failed else "   全部通过"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
