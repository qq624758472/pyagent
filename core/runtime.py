"""Shared paths and script execution for source and single-file builds."""
import os
import runpy
import subprocess
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
DATA_ROOT = Path(sys.executable).resolve().parent if FROZEN else RESOURCE_ROOT
BACKGROUND_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def initialize_stdio():
    """Restore redirected handles for windowed workers; GUI uses null streams.

    The windowed Python runtime sets sys.std* to None, even when a parent
    explicitly redirects the Windows handles to pipes or files.
    """
    for name, number, mode in (("stdin", -10, "r"), ("stdout", -11, "w"),
                               ("stderr", -12, "w")):
        if getattr(sys, name) is not None:
            continue
        stream = None
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetStdHandle.argtypes = [wintypes.DWORD]
            kernel.GetStdHandle.restype = wintypes.HANDLE
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
                wintypes.BOOL, wintypes.DWORD]
            kernel.DuplicateHandle.restype = wintypes.BOOL
            handle = kernel.GetStdHandle(number & 0xFFFFFFFF)
            if handle and handle != ctypes.c_void_p(-1).value:
                duplicate = wintypes.HANDLE()
                process = kernel.GetCurrentProcess()
                if kernel.DuplicateHandle(process, handle, process,
                                          ctypes.byref(duplicate), 0, False, 2):
                    flags = os.O_RDONLY if mode == "r" else os.O_WRONLY
                    fd = msvcrt.open_osfhandle(duplicate.value, flags | os.O_BINARY)
                    stream = os.fdopen(fd, mode, encoding="utf-8", errors="replace",
                                       buffering=1)
        if stream is None:
            stream = open(os.devnull, mode, encoding="utf-8")
        setattr(sys, name, stream)


def python_command(path):
    """Reuse the bundled interpreter, without starting another GUI."""
    return [sys.executable, "--run-python" if FROZEN else "-u", str(path)]


def run_python_worker():
    """Internal child-process entry; called before importing the GUI/agent."""
    if len(sys.argv) < 3:
        raise SystemExit("Usage: --run-python SCRIPT [ARGS ...]")
    target = Path(sys.argv[2]).resolve()
    sys.argv = [str(target)] + sys.argv[3:]
    sys.path.insert(0, str(target.parent))
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", write_through=True)
    # Both GUI and worker entry points initialize redirected streams first.
    os.environ.setdefault("PYTHONUTF8", "1")
    runpy.run_path(str(target), run_name="__main__")
