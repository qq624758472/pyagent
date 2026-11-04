"""工作区：Agent 文件操作的沙箱根目录，带路径越界保护（跨平台）"""
import re
import os
import shutil
import tempfile
from pathlib import Path

MAX_READ_BYTES = 200_000

# 内容搜索视为文本的扩展名（与 tools.py 的 _KB_TEXT_EXT 口径一致）
_TEXT_EXT = {".txt", ".md", ".py", ".json", ".c", ".h", ".cpp", ".hpp", ".log",
             ".csv", ".yaml", ".yml", ".xml", ".ini", ".cfg", ".rst", ".toml",
             ".java", ".js", ".ts", ".sh", ".bat", ".sql"}


class WorkspaceError(Exception):
    pass


def _as_line(v):
    """行号参数归一：模型可能给 int / "950" / "950-980" / None"""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v > 0 else None
    s = str(v).strip()
    if not s:
        return None
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m and int(m.group(1)) > 0 else None


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

    def read_file(self, path, line_start=None, line_end=None) -> str:
        p = self.resolve(path, must_exist=True)
        if p.is_dir():
            raise WorkspaceError(f"'{path}' 是目录，不是文件")
        if p.suffix.lower() == ".docx":
            from core.word_documents import DocumentError, inspect_docx
            try:
                return "[DOCX 结构校验通过，以下为正文文本]\n" + inspect_docx(p)["text"]
            except DocumentError as exc:
                raise WorkspaceError(str(exc)) from exc
        if p.stat().st_size > MAX_READ_BYTES:
            raise WorkspaceError(f"文件过大(>{MAX_READ_BYTES // 1000}KB)，拒绝读取")
        text = p.read_text(encoding="utf-8", errors="replace")
        if "\x00" in text[:1024]:
            raise WorkspaceError("疑似二进制文件，无法作为文本读取")
        if not text:
            return "(空文件)"
        # 行区间读取（V31.2）：模型常要"看第 900 行附近"，不必整篇拉进来
        start, end = _as_line(line_start), _as_line(line_end)
        if start or end:
            lines = text.splitlines()
            total = len(lines)
            s = max(1, start or 1)
            e = min(total, end or total)
            if s > total:
                return f"(文件共 {total} 行，起始行 {s} 超出范围)"
            body = "\n".join(lines[s - 1:e])
            head = f"(文件共 {total} 行，显示第 {s}-{e} 行)\n"
            return head + body
        return text

    def write_file(self, path, content="") -> str:
        p = self.resolve(path)
        if p.is_dir():
            raise WorkspaceError(f"'{path}' 是已存在的目录")
        if p.suffix.lower() == ".docx":
            from core.word_documents import DocumentError, build_docx, inspect_docx
            try:
                data = build_docx(content)
                info = inspect_docx(data)
            except DocumentError as exc:
                raise WorkspaceError(str(exc)) from exc
            p.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=p.parent, suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                os.replace(temporary, p)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return (f"已生成真正的 Word 文档 {path} ({len(data)} 字节)，"
                    f"DOCX ZIP/XML 结构校验通过，正文 {info['characters']} 字符。"
                    "结构有效不代表内容完整，请继续核对用户要求；尚未进行 Word 排版验证。")
        self._check_text_extension(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(content), encoding="utf-8")
        return f"已写入 {path} ({len(str(content))} 字符)"

    def append_file(self, path, content="") -> str:
        p = self.resolve(path)
        if p.suffix.lower() == ".docx":
            raise WorkspaceError("不能向 DOCX 文件追加纯文本，否则会损坏 Word 包。请先 read_file 读取正文，再 write_file 写入完整内容以重新生成 DOCX。")
        self._check_text_extension(p)
        if p.is_dir():
            raise WorkspaceError(f"'{path}' 是已存在的目录")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(str(content))
        return f"已追加 {len(str(content))} 字符到 {path}"

    @staticmethod
    def _check_text_extension(path):
        if path.suffix.lower() in {".doc", ".docm", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf"}:
            raise WorkspaceError("该扩展名需要真实的二进制文档格式，不能直接保存纯文本；Word 文档请使用 .docx，其他格式需通过对应生成程序创建。")

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

    def search_files(self, pattern, content=False, path=None, regex=False) -> str:
        """搜索工作区文件。

        默认按文件名 glob 匹配（rglob）；pattern 不含 glob 通配符
        （* ? [）时多半是模型想找代码符号/文本片段——文件名未命中后
        自动降级为内容搜索，避免模型拿着普通关键字反复空手重试（V31）。
        content=True 显式强制内容搜索。path 可限定搜索的子目录。"""
        pat = str(pattern or "").strip()
        if not pat:
            return "错误: pattern 不能为空"
        base = self.resolve(path or ".", must_exist=True)
        hits = [] if content or regex else [str(p.relative_to(self.root))
                for p in (base.rglob(pat) if base.is_dir() else [base])
                if p.is_file() and (base.is_dir() or p.match(pat))]
        if hits:
            return "\n".join(hits[:200])
        has_glob = any(c in pat for c in "*?[")
        if content or regex or not has_glob:
            # 对模型常见的明确正则转义兼容；普通关键词仍按字面匹配。
            use_regex = regex or bool(re.search(r'\\[sbdw]|\(\?[=:!]', pat))
            return self._search_content(pat, base, regex=use_regex)
        return (f"未找到文件名匹配 '{pat}' 的文件（本工具只按**文件名** glob 匹配。"
                f"若要在文件**内容**中搜索关键字，请加参数 content=true，如 "
                f'search_files({{"pattern": "{pat}", "content": true}})）')

    def _search_content(self, keyword, base=None, regex=False) -> str:
        """在文本文件内容中逐行搜索关键字，输出 文件:行号: 内容。
        二进制/超大文件跳过；命中上限 100 行防刷屏。"""
        base = base or self.root
        try:
            matcher = re.compile(keyword) if regex else None
        except re.error as exc:
            return f'错误: 无效正则表达式: {exc}。请修正或设置 regex=false 使用普通关键词。'
        out, scanned = [], 0
        for p in (sorted(base.rglob("*")) if base.is_dir() else [base]):
            if not p.is_file() or p.suffix.lower() not in _TEXT_EXT:
                continue
            try:
                p = self.resolve(str(p), must_exist=True)
                if p.stat().st_size > MAX_READ_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            scanned += 1
            for i, line in enumerate(text.splitlines(), 1):
                if (matcher.search(line) if matcher else keyword in line):
                    rel = p.relative_to(self.root)
                    out.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(out) >= 100:
                        break
            if len(out) >= 100:
                break
        if out:
            head = (f"[内容搜索] 关键词「{keyword}」命中 {len(out)} 处"
                    f"（扫描 {scanned} 个文本文件）：")
            tail = "\n...(命中过多已截断)" if len(out) >= 100 else ""
            return head + "\n" + "\n".join(out) + tail
        return (f"文件名和文件内容均未找到「{keyword}」"
                f"（已扫描 {scanned} 个文本文件；二进制/超大文件未扫描）。"
                "可尝试更换关键词或确认该符号确实存在。")
