"""Python 常驻进程和 TCP 联调工具。

执行均限定工作区内的 Python 文件；不与普通 run_python 的同步执行混淆。
用于开发/调试的服务进程有显式的启动、状态读取和停止操作。
"""
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from core.tools import ConfirmRequired, tool
from core.workspace import WorkspaceError

_lock = threading.RLock()
_processes = {}


def _script(ws, path):
    if ws is None:
        raise WorkspaceError("未挂载工作区")
    target = ws.resolve(path, must_exist=True)
    if target.suffix.lower() != ".py":
        raise WorkspaceError("仅允许执行工作区内 .py 文件")
    return target


def _argv(path, args):
    try:
        return [sys.executable, "-u", str(path)] + shlex.split(str(args or ""), posix=os.name != "nt")
    except ValueError as exc:
        raise ValueError(f"参数格式错误: {exc}") from exc


def _spawn(ws, target, args=""):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    output = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            _argv(target, args), cwd=str(ws.root), env=env,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if os.name == "nt" else 0)
    except Exception:
        output.close()
        raise
    return proc, output


def _tail(output, max_chars=1800):
    try:
        output.flush()
        output.seek(0, os.SEEK_END)
        size = output.tell()
        output.seek(max(0, size - max_chars * 4))
        return output.read()[-max_chars:]
    except (OSError, ValueError):
        return ""


def _stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


@tool("start_process",
      "启动工作区内常驻 Python 服务进程（不等待服务退出），返回 process_id。"
      "TCP/HTTP 服务器请用本工具，不要用 run_python 阻塞等待。",
      {"path": "要启动的 Python 服务脚本，如 tcp_server.py",
       "args": "传给脚本的参数，可省略"})
def start_process(ws, cfg, path, args="", _confirmed=False):
    if not getattr(cfg, "python_enabled", True):
        return "错误: Python 执行已关闭"
    if not _confirmed:
        raise ConfirmRequired(f"后台启动 Python 服务: {path}")
    try:
        target = _script(ws, path)
        proc, output = _spawn(ws, target, args)
        time.sleep(0.3)
        if proc.poll() is not None:
            detail = _tail(output)
            output.close()
            return f"错误: 服务启动后立即退出，退出码 {proc.returncode}; {detail}"
        identity = uuid.uuid4().hex[:12]
        with _lock:
            _processes[identity] = (str(ws.root), proc, output)
        return f"已启动服务 process_id={identity}, pid={proc.pid}。可用 process_status 查看，stop_process 停止。"
    except (WorkspaceError, OSError, ValueError) as exc:
        return f"错误: 启动进程失败: {exc}"


def _get_entry(ws, process_id):
    if ws is None:
        return None
    with _lock:
        item = _processes.get(str(process_id))
        if item and item[0] == str(ws.root):
            return item
    return None


@tool("process_status", "获取 start_process 启动的服务状态、退出码及最近输出。",
      {"process_id": "由 start_process 返回的 process_id"})
def process_status(ws, cfg, process_id):
    entry = _get_entry(ws, process_id)
    if entry is None:
        return "错误: 未找到当前工作区对应的进程 ID"
    _, proc, output = entry
    exit_code = proc.poll()
    status = "running" if exit_code is None else f"exited({exit_code})"
    return f"process_id={process_id} status={status} pid={proc.pid}\n{_tail(output)}"


@tool("stop_process", "停止本工作区通过 start_process 启动的服务并回收进程资源。",
      {"process_id": "由 start_process 返回的 process_id"})
def stop_process(ws, cfg, process_id):
    entry = _get_entry(ws, process_id)
    if entry is None:
        return "错误: 未找到当前工作区对应的进程 ID"
    _, proc, output = entry
    try:
        _stop(proc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"错误: 停止进程失败: {exc}"
    finally:
        if proc.poll() is not None:
            with _lock:
                _processes.pop(str(process_id), None)
            output.close()
    return f"已停止 process_id={process_id}, exit_code={proc.returncode}"


@tool("tcp_echo_test",
      "自动 TCP 联调：启动工作区服务端脚本，等待本机端口就绪，"
      "通过 socket 发送测试内容并核对回显；可同时测试生成的客户端脚本，最后自动停止服务端。"
      "返回明确的通过/失败结果。不要用 run_python 执行永不退出的服务器。",
      {"server_path": "工作区内服务端脚本，如 tcp_server.py",
       "client_path": "可选，工作区内客户端脚本，如 tcp_client.py，若填写则也执行并校验其输出",
       "message": "发送的测试消息，默认 hello",
       "port": "服务端监听端口，默认 12345",
       "host": "本地服务地址，默认 127.0.0.1"})
def tcp_echo_test(ws, cfg, server_path="tcp_server.py", client_path="",
                  message="hello", port=12345, host="127.0.0.1", _confirmed=False):
    if not getattr(cfg, "python_enabled", True):
        return "错误: Python 执行已关闭"
    if not _confirmed:
        raise ConfirmRequired(f"启动 {server_path} 并在 {host}:{port} 进行 TCP 联调")
    if host not in ("127.0.0.1", "localhost", "::1"):
        return "错误: 联调工具仅支持本机回环地址"
    try:
        server = _script(ws, server_path)
        client = _script(ws, client_path) if client_path else None
        port = int(port)
        if not 1 <= port <= 65535:
            return "错误: 端口范围非法"
        payload = str(message).encode("utf-8")
        if not payload or len(payload) > 2048:
            return "错误: 测试消息长度必须在 1~2048 字节"
        # 不占用现有服务器的端口，避免误测到其他进程。
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                return f"错误: {host}:{port} 已被占用，请使用空闲端口"
        proc, output = _spawn(ws, server)
    except (WorkspaceError, ValueError, OSError) as exc:
        return f"错误: TCP 联调初始化失败: {exc}"
    try:
        deadline = time.monotonic() + 6
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return f"错误: 服务端提前退出，退出码 {proc.returncode}; {_tail(output)}"
            try:
                with socket.create_connection((host, port), timeout=0.3):
                    ready = True
                    break
            except OSError:
                time.sleep(0.1)
        if not ready:
            return f"错误: 服务端 6 秒内未监听 {host}:{port}; {_tail(output)}"
        with socket.create_connection((host, port), timeout=3) as sock:
            sock.settimeout(3)
            sock.sendall(payload)
            received = bytearray()
            while len(received) < len(payload):
                part = sock.recv(len(payload) - len(received))
                if not part:
                    break
                received.extend(part)
        if bytes(received) != payload:
            return f"错误: TCP 回显不匹配，发送={payload!r}, 收到={bytes(received)!r}"
        result = f"socket 客户端发送 {message!r}，服务端原样返回；"
        if client:
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            try:
                run = subprocess.run(
                    _argv(client, ""), input=f"{message}\nquit\n",
                    cwd=str(ws.root), env=env, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=7)
            except subprocess.TimeoutExpired:
                return "错误: 生成的 TCP 客户端未在 7 秒内退出，请提供可自动测试的非交互模式"
            if run.returncode != 0 or str(message) not in run.stdout:
                return ("错误: socket 回环通过，但生成的客户端未通过校验；"
                        f"exit={run.returncode}; stdout={run.stdout[-500:]}; stderr={run.stderr[-300:]}")
            result += "生成的客户端也发送/接收到测试消息；"
        return f"TCP联调通过：{result}服务端已在测试结束后回收。"
    except (OSError, ValueError) as exc:
        return f"错误: TCP 联调失败: {exc}"
    finally:
        try:
            _stop(proc)
        finally:
            output.close()
