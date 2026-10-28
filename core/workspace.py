"""工作区：Agent 文件操作的沙箱根目录，带路径越界保护（跨平台）"""
import shutil
from pathlib import Path

MAX_READ_BYTES = 200_000


class WorkspaceError(Exception):
    pass


class Workspace:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"工作区目录不存在: {self.root}")

    def __repr__(self):
        return f"Workspace({self.root})"

    # ---- 路径安全：所有操作先经过这里 ----
    def resolve(self, rel: str = "", must_exist: bool = False) -> Path:
        # 模型常给 Windows 风格反斜杠（E:\ai_work\…）；归一成 / 保证
        # 跨平台一致（Windows 下 / 同样合法，Linux 下 \ 是普通字符会被
        # 当成文件名的一部分 → write 与 run 同一文件不同写法就对不上）
        rel = (rel or "").strip().replace("\\", "/")
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root):
            raise WorkspaceError(f"路径越界: '{rel}' 不在工作区内，已拒绝")
        if must_exist and not target.exists():
            raise WorkspaceError(f"路径不存在: {rel}")
        return target

    # ---- 文件操作（均返回字符串观察结果）----
    def list_files(self, path=".", recursive=False) -> str:
        base = self.resolve(path, must_exist=True)
        items = base.rglob("*") if recursive else base.iterdir()
        lines = []
        for p in sorted(items)[:500]:
            rel = p.relative_to(self.root)
            if p.is_dir():
                lines.append(f"[目录] {rel}")
            else:
                lines.append(f"[文件 {p.stat().st_size}B] {rel}")
        return "\n".join(lines) or "(空目录)"

    def read_file(self, path) -> str:
        p = self.resolve(path, must_exist=True)
        if p.is_dir():
            raise WorkspaceError(f"'{path}' 是目录，不是文件")
        if p.stat().st_size > MAX_READ_BYTES:
            raise WorkspaceError(f"文件过大(>{MAX_READ_BYTES // 1000}KB)，拒绝读取")
        text = p.read_text(encoding="utf-8", errors="replace")
        if "\x00" in text[:1024]:
            raise WorkspaceError("疑似二进制文件，无法作为文本读取")
        return text or "(空文件)"

    def write_file(self, path, content="") -> str:
        p = self.resolve(path)
        if p.is_dir():
            raise WorkspaceError(f"'{path}' 是已存在的目录")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(content), encoding="utf-8")
        return f"已写入 {path} ({len(str(content))} 字符)"

    def append_file(self, path, content="") -> str:
        p = self.resolve(path)
        if p.is_dir():
            raise WorkspaceError(f"'{path}' 是已存在的目录")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(str(content))
        return f"已追加 {len(str(content))} 字符到 {path}"

    def create_dir(self, path) -> str:
        p = self.resolve(path)
        p.mkdir(parents=True, exist_ok=True)
        return f"已创建目录 {path}"

    def delete_file(self, path, recursive=False) -> str:
        p = self.resolve(path, must_exist=True)
        if p == self.root:
            raise WorkspaceError("不允许删除工作区根目录")
        if p.is_dir():
            if recursive:
                shutil.rmtree(p)
            else:
                try:
                    p.rmdir()
                except OSError:
                    raise WorkspaceError(f"目录 '{path}' 非空，需 recursive=true")
        else:
            p.unlink()
        return f"已删除 {path}"

    def search_files(self, pattern) -> str:
        hits = [str(p.relative_to(self.root)) for p in self.root.rglob(pattern)]
        return "\n".join(hits[:200]) or f"未找到匹配 '{pattern}' 的文件"
