"""내보내기 — TXT / MD / DOCX / XLSX / JSON / 검색 가능한 PDF."""
from __future__ import annotations

import io
import json
from pathlib import Path

from . import refine


def effective_lines(r: dict) -> list[dict]:
    """편집·재가공(마스킹 등)한 텍스트를 줄 데이터에 반영. 줄 수가 같으면 좌표 유지, 다르면 좌표 없이."""
    from .engines import lines_to_text
    lines = r.get("lines") or []
    text = r.get("text") or ""
    if text == lines_to_text(lines):
        return lines
    new = [t for t in text.splitlines() if t.strip()]
    if len(new) == len(lines):
        return [{**l, "text": t} for l, t in zip(lines, new)]
    out, block = [], 0
    for t in text.splitlines():
        if not t.strip():
            block += 1
            continue
        out.append({"text": t, "conf": None, "box": None, "block": block})
    return out


def _page_text(p: dict) -> str:
    r = p.get("result") or {}
    return r.get("text", "")


def to_txt(doc: dict, pages: list[dict], sep: bool = True) -> bytes:
    parts = []
    for p in pages:
        if sep and len(pages) > 1:
            parts.append(f"===== {p['index'] + 1} 페이지 =====")
        parts.append(_page_text(p))
        parts.append("")
    return "\n".join(parts).strip().encode("utf-8-sig")  # 메모장 한글 깨짐 방지 BOM


def to_md(doc: dict, pages: list[dict]) -> bytes:
    out = [f"# {doc['name']}", ""]
    for p in pages:
        if len(pages) > 1:
            out += [f"## {p['index'] + 1} 페이지", ""]
        out += [_page_text(p), ""]
    return "\n".join(out).encode("utf-8")


def to_json(doc: dict, pages: list[dict]) -> bytes:
    data = {
        "document": doc["name"],
        "pages": [{"page": p["index"] + 1, "width": p.get("width"), "height": p.get("height"), **(p.get("result") or {}),
                   "lines": effective_lines(p.get("result") or {})} for p in pages],
    }
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def to_docx(doc: dict, pages: list[dict]) -> bytes:
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.oxml.ns import qn
    from docx.shared import Pt

    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Malgun Gothic"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    d.add_heading(doc["name"], level=1)
    for i, p in enumerate(pages):
        if len(pages) > 1:
            d.add_heading(f"{p['index'] + 1} 페이지", level=2)
        for para in _page_text(p).split("\n"):
            d.add_paragraph(para)
        if i < len(pages) - 1:
            d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def to_xlsx(doc: dict, pages: list[dict], tpl: dict | None = None, model: str | None = None) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    head_fill = PatternFill("solid", fgColor="1F2A44")
    head_font = Font(bold=True, color="FFFFFF")
    low_fill = PatternFill("solid", fgColor="FFE7CC")

    ws = wb.active
    ws.title = "줄별 인식결과"
    ws.append(["페이지", "줄", "텍스트", "신뢰도", "엔진", "x", "y", "w", "h"])
    for p in pages:
        r = p.get("result") or {}
        for j, l in enumerate(effective_lines(r)):
            b = l.get("box") or [None] * 4
            ws.append([p["index"] + 1, j + 1, l["text"], l.get("conf"), r.get("engine"),
                       *[round(v, 4) if v is not None else None for v in b]])
            if l.get("conf") is not None and l["conf"] < 70:
                for c in ws[ws.max_row]:
                    c.fill = low_fill
    ws.column_dimensions["C"].width = 80

    ws2 = wb.create_sheet("공문서 항목")
    ws2.append(["항목", "값"])
    full = "\n".join(_page_text(p) for p in pages)
    for f in refine.extract_fields(full):
        ws2.append([f["label"], f["value"]])
    ws2.column_dimensions["A"].width = 16
    ws2.column_dimensions["B"].width = 80

    if tpl:
        from . import extract
        wsx = wb.create_sheet("추출 항목", 0)
        wsx.append(["항목", "값"])
        for v in extract.values_for(tpl, full, model):
            wsx.append([v["name"], v["value"][:32000]])
            wsx.cell(wsx.max_row, 2).alignment = Alignment(wrap_text=True, vertical="top")
        wsx.column_dimensions["A"].width = 18
        wsx.column_dimensions["B"].width = 90
        wb.active = 0

    ws3 = wb.create_sheet("페이지 전체")
    ws3.append(["페이지", "텍스트", "평균 신뢰도", "엔진"])
    for p in pages:
        r = p.get("result") or {}
        ws3.append([p["index"] + 1, r.get("text", ""), r.get("mean_conf"), r.get("engine")])
        ws3.cell(ws3.max_row, 2).alignment = Alignment(wrap_text=True, vertical="top")
    ws3.column_dimensions["B"].width = 100

    for sh in wb.worksheets:
        for c in sh[1]:
            c.fill, c.font = head_fill, head_font
        sh.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def to_pdf(doc: dict, pages: list[dict], doc_dir: Path) -> bytes:
    """원본 이미지 위에 보이지 않는 텍스트 층을 얹은 '검색 가능한 PDF'. 편집한 텍스트가 반영된다."""
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    font = "HYGothic-Medium"
    try:
        pdfmetrics.getFont(font)
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont(font))

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for p in pages:
        r = p.get("result") or {}
        info = r.get("pre") or {}
        use_proc = bool(info.get("skew") or info.get("rotate")) and (doc_dir / p["proc"]).exists()
        img_path = doc_dir / (p["proc"] if use_proc else p["file"])
        im = Image.open(img_path)
        im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        base_dpi = p.get("dpi") or 300
        dpi = base_dpi if not use_proc else base_dpi * (info.get("scale") or 1)
        W, H = im.width * 72 / dpi, im.height * 72 / dpi
        c.setPageSize((W, H))
        jb = io.BytesIO()
        im.save(jb, "JPEG", quality=82)
        jb.seek(0)
        c.drawImage(ImageReader(jb), 0, 0, W, H)

        lines = effective_lines(r)
        txt = c.beginText()
        txt.setTextRenderMode(3)  # 보이지 않는 텍스트
        if lines and all(l.get("box") for l in lines):
            for l in lines:
                x, y, w, h = l["box"]
                size = max(4.0, h * H * 0.85)
                tw = pdfmetrics.stringWidth(l["text"], font, size) or 1
                txt.setFont(font, size)
                txt.setHorizScale(max(10, min(400, (w * W) / tw * 100)))
                txt.setTextOrigin(x * W, H - (y + h) * H + h * H * 0.15)
                txt.textOut(l["text"])
        else:  # 좌표 없는 엔진(AI): 위에서부터 차례로 배치
            size, y = 10, H - 24
            for line in (r.get("text") or "").splitlines():
                txt.setFont(font, size)
                txt.setTextOrigin(24, y)
                txt.textOut(line)
                y -= size * 1.4
                if y < 12:
                    break
        c.drawText(txt)
        c.showPage()
    c.save()
    return buf.getvalue()


FORMATS = {
    "txt": ("text/plain", ".txt"),
    "md": ("text/markdown", ".md"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
    "json": ("application/json", ".json"),
    "pdf": ("application/pdf", ".pdf"),
}


def export(fmt: str, doc: dict, pages: list[dict], doc_dir: Path, tpl: dict | None = None, model: str | None = None) -> bytes:
    if fmt == "txt":
        return to_txt(doc, pages)
    if fmt == "md":
        return to_md(doc, pages)
    if fmt == "docx":
        return to_docx(doc, pages)
    if fmt == "xlsx":
        return to_xlsx(doc, pages, tpl, model)
    if fmt == "json":
        return to_json(doc, pages)
    if fmt == "pdf":
        return to_pdf(doc, pages, doc_dir)
    raise ValueError(f"지원하지 않는 형식: {fmt}")
