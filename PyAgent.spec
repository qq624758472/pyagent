# Build: python -m PyInstaller --noconfirm --clean PyAgent.spec
from pathlib import Path

root = Path(SPECPATH)
a = Analysis(
    [str(root / "main.py")],
    pathex=[str(root)],
    binaries=[],
    # Explicit allowlist: never bundle local config, sessions, logs or workspaces.
    datas=[(str(root / "icons" / "app.png"), "icons"),
           (str(root / "icons" / "zhixing.svg"), "icons"),
           (str(root / "skills"), "skills"),
           (str(root / "LICENSE"), ".")],
    hiddenimports=["PyQt5.QtSvg", "csv", "sqlite3", "statistics", "decimal",
                   "fractions", "http.server", "socketserver", "unittest",
                   "asyncio", "xml.etree.ElementTree", "zipfile", "tarfile",
                   "concurrent.futures", "ctypes"],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "numpy", "pandas", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="Zhixing", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
    icon=str(root / "icons" / "zhixing.png"),
)
