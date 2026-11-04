"""Word 文件生成与结构验收，仅依赖标准库，源码版和打包版行为一致。"""
import io
import re
import zipfile
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
MAIN_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
ET.register_namespace("w", W)


class DocumentError(ValueError):
    pass


def _w(tag, parent=None, **attrs):
    node = ET.Element(f"{{{W}}}{tag}", {f"{{{W}}}{k}": str(v) for k, v in attrs.items()})
    if parent is not None:
        parent.append(node)
    return node


def _xml(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def inspect_docx(path_or_bytes):
    """验收 ZIP 完整性、包类型/入口、全部 XML 和 Word 正文，返回可读文本。"""
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    try:
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if len(entries) > 2048 or sum(i.file_size for i in entries) > 20_000_000:
                raise DocumentError("Word 文件解压后过大，拒绝解析")
            if len({i.filename for i in entries}) != len(entries):
                raise DocumentError("Word 包含重复条目")
            roots = {}
            for info in entries:
                data = archive.read(info)  # 同时验证 CRC，不向文件系统解压。
                if info.filename.endswith((".xml", ".rels")):
                    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
                        raise DocumentError("Word XML 不允许 DTD/实体声明")
                    roots[info.filename] = ET.fromstring(data)
            types = roots.get("[Content_Types].xml")
            rels = roots.get("_rels/.rels")
            document = roots.get("word/document.xml")
            if types is None or rels is None or document is None:
                raise DocumentError("缺少 Word OOXML 必需部件")
            if not any(n.get("PartName") == "/word/document.xml" and n.get("ContentType") == MAIN_TYPE
                       for n in types.findall(f"{{{CT}}}Override")):
                raise DocumentError("Word 正文内容类型不正确")
            if not any(n.get("Type") == OFFICE + "officeDocument"
                       and n.get("Target", "").lstrip("/") == "word/document.xml"
                       and n.get("TargetMode") != "External"
                       for n in rels.findall(f"{{{R}}}Relationship")):
                raise DocumentError("Word 包入口关系不正确")
            if document.tag != f"{{{W}}}document" or document.find(f"{{{W}}}body") is None:
                raise DocumentError("缺少 Word 文档正文")
            paragraphs = ["".join(t.text or "" for t in p.iter(f"{{{W}}}t"))
                          for p in document.iter(f"{{{W}}}p")]
            text = "\n".join(paragraphs)
            if not text.strip():
                raise DocumentError("Word 文档正文为空")
            return {"text": text, "characters": len(text.strip()), "paragraphs": len(paragraphs)}
    except DocumentError:
        raise
    except (OSError, zipfile.BadZipFile, ET.ParseError, KeyError, RuntimeError, ValueError) as exc:
        raise DocumentError(f"不是有效的 DOCX 文件：{exc}；不能把纯文本直接改名为 .docx") from exc


def _paragraph(parent, text, style=None, bullet=False, code=False):
    paragraph = _w("p", parent)
    props = _w("pPr", paragraph)
    if style:
        _w("pStyle", props, val=style)
    if bullet:
        numbering = _w("numPr", props)
        _w("ilvl", numbering, val=0)
        _w("numId", numbering, val=1)
    # 保留明确的粗体/行内代码，其他文本按原文写入并由 XML 库转义。
    for part in re.split(r"(\*\*.+?\*\*|`[^`]+`)", text):
        if not part:
            continue
        run = _w("r", paragraph)
        if code or part.startswith(("**", "`")):
            rprops = _w("rPr", run)
            if part.startswith("**") and part.endswith("**"):
                _w("b", rprops)
                part = part[2:-2]
            elif part.startswith("`") and part.endswith("`"):
                part = part[1:-1]
            if code or "`" in text:
                _w("rFonts", rprops, ascii="Consolas", hAnsi="Consolas", eastAsia="宋体")
        t = _w("t", run)
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = part
    return paragraph


def build_docx(content):
    """将模型文本/基本 Markdown 确定性转换成真正的 Word 包，保存前先自检。"""
    content = str(content)
    if not content.strip():
        raise DocumentError("Word 内容为空，未创建文件")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", content):
        raise DocumentError("正文含 XML 不支持的控制字符，未覆盖原文件")
    doc = _w("document")
    body = _w("body", doc)
    code = False
    title_used = False
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.strip().startswith("```"):
            code = not code
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line) if not code else None
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line) if not code else None
        if heading:
            level, text = heading.groups()
            style = "Title" if len(level) == 1 and not title_used else f"Heading{min(len(level), 3)}"
            title_used = True
            _paragraph(body, text, style)
        elif (not code and "|" in line and i < len(lines)
              and re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", lines[i])):
            table = _w("tbl", body)
            props = _w("tblPr", table)
            _w("tblW", props, w=9360, type="dxa")
            borders = _w("tblBorders", props)
            for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
                _w(side, borders, val="single", sz=4, color="D9D9D9")
            rows = [line]
            i += 1  # 分隔行不作为内容。
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(lines[i])
                i += 1
            cols = max(len(row.strip().strip("|").split("|")) for row in rows)
            grid = _w("tblGrid", table)
            for _ in range(cols):
                _w("gridCol", grid, w=9360 // cols)
            for row_index, row in enumerate(rows):
                tr = _w("tr", table)
                cells = row.strip().strip("|").split("|")
                for cell in cells + [""] * (cols - len(cells)):
                    tc = _w("tc", tr)
                    tcprops = _w("tcPr", tc)
                    _w("tcW", tcprops, w=9360 // cols, type="dxa")
                    _w("vAlign", tcprops, val="center")
                    if row_index == 0:
                        _w("shd", tcprops, fill="E7E6E6")
                    _paragraph(tc, cell.strip())
        else:
            _paragraph(body, bullet.group(1) if bullet else line, bullet=bool(bullet), code=code)
    section = _w("sectPr", body)
    _w("pgSz", section, w=12240, h=15840)
    _w("pgMar", section, top=1440, right=1440, bottom=1440, left=1440, header=720, footer=720, gutter=0)
    styles = _w("styles")
    for name, size in (("Normal", 22), ("Title", 40), ("Heading1", 32), ("Heading2", 28), ("Heading3", 24)):
        style = _w("style", styles, type="paragraph", styleId=name)
        if name == "Normal":
            style.set(f"{{{W}}}default", "1")
        _w("name", style, val=name)
        if name != "Normal":
            _w("basedOn", style, val="Normal")
        props = _w("pPr", style)
        _w("spacing", props, after=120, line=280, lineRule="auto")
        if name != "Normal":
            _w("keepNext", props)
        runs = _w("rPr", style)
        _w("rFonts", runs, ascii="Calibri", hAnsi="Calibri", eastAsia="宋体")
        _w("sz", runs, val=size)
        _w("color", runs, val="000000")
        if name.startswith("Heading"):
            _w("b", runs)
    numbering = _w("numbering")
    abstract = _w("abstractNum", numbering, abstractNumId=0)
    level = _w("lvl", abstract, ilvl=0)
    _w("start", level, val=1)
    _w("numFmt", level, val="bullet")
    _w("lvlText", level, val="•")
    indent = _w("pPr", level)
    _w("ind", indent, left=360, hanging=180)
    num = _w("num", numbering, numId=1)
    _w("abstractNumId", num, val=0)
    types = ET.Element(f"{{{CT}}}Types")
    ET.SubElement(types, f"{{{CT}}}Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
    ET.SubElement(types, f"{{{CT}}}Default", Extension="xml", ContentType="application/xml")
    for part, mime in (("document", MAIN_TYPE), ("styles", "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"),
                       ("numbering", "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml")):
        ET.SubElement(types, f"{{{CT}}}Override", PartName=f"/word/{part}.xml", ContentType=mime)
    root_rels = ET.Element(f"{{{R}}}Relationships")
    ET.SubElement(root_rels, f"{{{R}}}Relationship", Id="rId1", Type=OFFICE + "officeDocument", Target="word/document.xml")
    doc_rels = ET.Element(f"{{{R}}}Relationships")
    for index, part in enumerate(("styles", "numbering"), 1):
        ET.SubElement(doc_rels, f"{{{R}}}Relationship", Id=f"rId{index}", Type=OFFICE + part, Target=part + ".xml")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, root in (("[Content_Types].xml", types), ("_rels/.rels", root_rels),
                           ("word/document.xml", doc), ("word/styles.xml", styles),
                           ("word/numbering.xml", numbering), ("word/_rels/document.xml.rels", doc_rels)):
            archive.writestr(name, _xml(root))
    data = output.getvalue()
    inspect_docx(data)
    return data
