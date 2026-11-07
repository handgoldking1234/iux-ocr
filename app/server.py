"""IUX OCR 로컬 서버 — 127.0.0.1 에서만 동작. 문서는 PC 밖으로 나가지 않는다."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import quote

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from . import engines, exporters, extract, preprocess, refine

ROOT = Path(__file__).resolve().parent.parent
if getattr(sys, "frozen", False):  # 설치형(exe): 프로그램 폴더는 쓰기 불가 → 사용자 폴더에 저장
    _default_ws = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "IUX OCR" / "workspace"
else:
    _default_ws = ROOT / "workspace"
WS = Path(os.environ.get("IUX_OCR_WORKSPACE", _default_ws))
DOCS = WS / "docs"
DOCS.mkdir(parents=True, exist_ok=True)
STATIC = ROOT / "app" / "static"
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
PDF_DPI = 300

import tempfile as _tf  # noqa: E402
_tf.tempdir = engines._ascii_tmp_root() or _tf.tempdir  # 한글 사용자 폴더 대응 (OpenCV·Tesseract)

app = FastAPI(title="IUX OCR")
_tok = WS / "helper_token.txt"
if not _tok.exists():
    _tok.write_text(uuid.uuid4().hex, "utf-8")
HELPER_TOKEN = _tok.read_text("utf-8").strip()
CURRENT: dict = {"doc_id": None}


@app.middleware("http")
async def helper_cors(request, call_next):
    """시스템 입력 도우미(다른 사이트 위에서 실행)만 /api/helper 를 부를 수 있게 CORS 허용. 토큰으로 보호."""
    is_helper = request.url.path.startswith("/api/helper")
    if is_helper and request.method == "OPTIONS":
        resp = Response(status_code=204)
    else:
        resp = await call_next(request)
    if is_helper:
        resp.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp
TEMPLATES = extract.TemplateStore(WS / "templates.json")
_lock = threading.Lock()


# ================================================================ 저장소

def _meta_path(doc_id: str) -> Path:
    return DOCS / doc_id / "meta.json"


def load_doc(doc_id: str) -> dict:
    p = _meta_path(doc_id)
    if not re.fullmatch(r"[a-f0-9]{12}", doc_id) or not p.exists():
        raise HTTPException(404, "문서를 찾을 수 없습니다.")
    return json.loads(p.read_text("utf-8"))


def save_doc(doc: dict) -> None:
    p = _meta_path(doc["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
    tmp.replace(p)


def _thumb(src: Path, dst: Path) -> None:
    img = preprocess.load_bgr(str(src))
    h, w = img.shape[:2]
    s = 220 / w
    t = cv2.resize(img, (220, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 78])
    buf.tofile(str(dst))


def ingest_file(src: Path, name: str) -> dict:
    """이미지/PDF 한 개를 문서로 등록."""
    return ingest_files([(src, name)], name)


def ingest_files(items: list[tuple[Path, str]], doc_name: str) -> dict:
    """여러 이미지/PDF를 한 문서(여러 쪽)로 등록 (페이지별 PNG + 썸네일). 폴더 단위 기록물용."""
    doc_id = uuid.uuid4().hex[:12]
    d = DOCS / doc_id
    (d / "pages").mkdir(parents=True)
    (d / "thumbs").mkdir()
    pages: list[dict] = []

    def new_page(img, dpi: int, source: str) -> dict:
        i = len(pages)
        fn = f"pages/p{i + 1:04d}.png"
        img.save(d / fn)
        pinfo = {"index": i, "file": fn, "proc": f"pages/p{i + 1:04d}_proc.png", "thumb": f"thumbs/p{i + 1:04d}.jpg",
                 "width": img.width, "height": img.height, "dpi": dpi, "result": None, "source": source}
        _thumb(d / fn, d / pinfo["thumb"])
        pages.append(pinfo)
        return pinfo

    try:
        for src, name in items:
            ext = src.suffix.lower()
            if ext == ".pdf":
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(str(src))
                for i in range(len(pdf)):
                    page = pdf[i]
                    scale = PDF_DPI / 72
                    pw, ph = page.get_size()
                    if pw * ph * scale * scale > 60_000_000:  # 너무 큰 페이지는 해상도 낮춤
                        scale = (60_000_000 / (pw * ph)) ** 0.5
                    pinfo = new_page(page.render(scale=scale).to_pil(), round(scale * 72), name)
                    try:  # 이미 텍스트가 있는 PDF는 바로 추출
                        text = page.get_textpage().get_text_range().replace("\r\n", "\n").replace("\r", "\n").strip()
                    except Exception:  # noqa: BLE001
                        text = ""
                    if len(text) >= 20:
                        lines = [{"text": l, "conf": None, "box": None, "block": 0} for l in text.splitlines()]
                        pinfo["pdf_text"] = True
                        pinfo["result"] = {"engine": "pdf-text", "lines": lines, "text": text, "mean_conf": None, "time_ms": 0}
                pdf.close()
            elif ext in IMG_EXT:
                from PIL import Image, ImageSequence
                im = Image.open(src)
                frames = [f.copy() for f in ImageSequence.Iterator(im)] if ext in (".tif", ".tiff") else [im]
                dpi_info = im.info.get("dpi")
                for fr in frames:
                    if fr.mode not in ("RGB", "L"):
                        fr = fr.convert("RGB")
                    dpi = round(float(dpi_info[0])) if dpi_info and float(dpi_info[0]) >= 72 else 300
                    new_page(fr, dpi, name)
            else:
                raise HTTPException(400, f"지원하지 않는 파일 형식입니다: {name}")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise

    doc = {"id": doc_id, "name": doc_name, "created": time.time(), "pages": pages}
    if len(items) > 1:
        doc["files"] = [n for _, n in items]
    save_doc(doc)
    return doc


def ocr_page(doc: dict, idx: int, engine: str, lang: str, psm: int, model: str | None, pre: dict | None, fix_spacing: bool = True) -> dict:
    d = DOCS / doc["id"]
    p = doc["pages"][idx]
    color = preprocess.load_bgr(str(d / p["file"]))
    gray, info = preprocess.preprocess(color, pre)
    preprocess.save_png(str(d / p["proc"]), gray)
    res = engines.run_engine(engine, gray, color, lang, psm, model, fix_spacing)
    res["pre"] = info
    res["lang"] = lang
    res["at"] = time.time()
    p["result"] = res
    p.pop("pdf_text", None)
    return p


# ================================================================ API 모델

class OcrReq(BaseModel):
    page: int
    engine: str = "tesseract"
    lang: str = "kor+eng"
    psm: int = 4
    model: str | None = None
    pre: dict | None = None
    fix_spacing: bool = True


class FieldsUpdate(BaseModel):
    template_id: str | None = None
    values: list[dict] = []
    checked: bool = False
    edited: bool = True


class AutofillReq(BaseModel):
    template_id: str
    model: str | None = None


class PageUpdate(BaseModel):
    text: str
    lines: list[dict] | None = None


class RefineReq(BaseModel):
    mode: str
    text: str
    ai: bool = False
    model: str | None = None
    custom: str | None = None
    opts: dict = {}


class ExportReq(BaseModel):
    format: str
    pages: list[int] | None = None
    template_id: str | None = None
    model: str | None = None


class ExtractReq(BaseModel):
    text: str
    template_id: str | None = None
    fields: list[dict] | None = None
    model: str | None = None


class BatchReq(BaseModel):
    input_dir: str
    output_dir: str
    engine: str = "tesseract"
    lang: str = "kor+eng"
    psm: int = 4
    model: str | None = None
    pre: dict | None = None
    fix_spacing: bool = True
    formats: list[str] = ["txt"]
    recursive: bool = False
    clean: bool = False
    mask: bool = False
    skip_existing: bool = True
    template_id: str | None = None
    llm: str | None = None


# ================================================================ 라우트

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text("utf-8")


@app.get("/helper.js")
def helper_js():
    return Response((STATIC / "helper.js").read_text("utf-8"), media_type="application/javascript")


@app.get("/helper-demo", response_class=HTMLResponse)
def helper_demo():
    return (STATIC / "helper-demo.html").read_text("utf-8")


@app.get("/helper-demo/form", response_class=HTMLResponse)
def helper_demo_form():
    return (STATIC / "helper-demo-form.html").read_text("utf-8")


@app.get("/api/engines")
def api_engines():
    return engines.engines_status()


@app.get("/api/docs")
def api_docs():
    out = []
    for m in DOCS.glob("*/meta.json"):
        try:
            d = json.loads(m.read_text("utf-8"))
            done = sum(1 for p in d["pages"] if p.get("result"))
            f = d.get("fields") or {}
            out.append({"id": d["id"], "name": d["name"], "created": d["created"], "pages": len(d["pages"]), "done": done,
                        "checked": bool(f.get("checked")), "filled": sum(1 for v in f.get("values", []) if v.get("value"))})
        except Exception:  # noqa: BLE001
            continue
    return sorted(out, key=lambda x: -x["created"])


def _decode_filename(name: str) -> str:
    """일부 프로그램이 보내는 '=?utf-8?B?...?=' 형식 파일명 복원."""
    if "=?" in name:
        try:
            from email.header import decode_header, make_header
            return str(make_header(decode_header(name)))
        except Exception:  # noqa: BLE001
            pass
    return name


def _sniff_ext(path: Path) -> str | None:
    head = path.read_bytes()[:16]
    if head.startswith(b"%PDF"):
        return ".pdf"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    try:
        from PIL import Image
        fmt = (Image.open(path).format or "").lower()
        return {"jpeg": ".jpg", "png": ".png", "bmp": ".bmp", "webp": ".webp", "gif": ".gif", "tiff": ".tif"}.get(fmt)
    except Exception:  # noqa: BLE001
        return None


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...), merge: str = Form(""), doc_name: str = Form("")):
    """merge=1 이면 받은 파일 전체를 한 문서(여러 쪽)로, 아니면 파일마다 문서."""
    tmp = WS / "tmp"
    tmp.mkdir(exist_ok=True)
    docs = []
    staged: list[tuple[Path, str]] = []
    for f in files:
        name = _decode_filename(Path(f.filename or "file").name)
        t = tmp / f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
        with open(t, "wb") as fo:
            shutil.copyfileobj(f.file, fo)
        if t.suffix not in IMG_EXT and t.suffix != ".pdf":  # 확장자가 없거나 깨진 경우 내용으로 판별
            ext = _sniff_ext(t)
            if ext:
                t2 = t.with_suffix(ext)
                t.rename(t2)
                t = t2
                name = Path(name).stem + ext
        staged.append((t, name))
    try:
        if merge in ("1", "true") and staged:
            docs.append(ingest_files(staged, doc_name or staged[0][1]))
        else:
            for t, name in staged:
                docs.append(ingest_file(t, name))
    finally:
        for t, _ in staged:
            t.unlink(missing_ok=True)
    return docs


@app.get("/api/docs/{doc_id}")
def api_doc(doc_id: str):
    return load_doc(doc_id)


@app.delete("/api/docs/{doc_id}")
def api_doc_delete(doc_id: str):
    load_doc(doc_id)
    shutil.rmtree(DOCS / doc_id, ignore_errors=True)
    return {"ok": True}


@app.get("/api/docs/{doc_id}/file/{kind}/{idx}")
def api_file(doc_id: str, kind: str, idx: int):
    doc = load_doc(doc_id)
    if not 0 <= idx < len(doc["pages"]):
        raise HTTPException(404)
    p = doc["pages"][idx]
    key = {"image": "file", "proc": "proc", "thumb": "thumb"}.get(kind)
    if not key:
        raise HTTPException(404)
    path = DOCS / doc_id / p[key]
    if not path.exists():
        path = DOCS / doc_id / p["file"]
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@app.post("/api/docs/{doc_id}/ocr")
def api_ocr(doc_id: str, req: OcrReq):
    with _lock:
        doc = load_doc(doc_id)
    if not 0 <= req.page < len(doc["pages"]):
        raise HTTPException(400, "페이지 번호 오류")
    try:
        page = ocr_page(doc, req.page, req.engine, req.lang, req.psm, req.model, req.pre, req.fix_spacing)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"OCR 실패: {e}") from e
    with _lock:
        fresh = load_doc(doc_id)
        fresh["pages"][req.page] = page
        save_doc(fresh)
    return page


@app.put("/api/docs/{doc_id}/pages/{idx}")
def api_page_update(doc_id: str, idx: int, upd: PageUpdate):
    with _lock:
        doc = load_doc(doc_id)
        p = doc["pages"][idx]
        r = p.get("result") or {"engine": "manual", "lines": [], "mean_conf": None}
        r["text"] = upd.text
        if upd.lines is not None:
            r["lines"] = upd.lines
        r["edited"] = True
        p["result"] = r
        save_doc(doc)
    return p


@app.post("/api/refine")
def api_refine(req: RefineReq):
    try:
        if req.ai:
            if not req.model:
                raise ValueError("AI 모델을 선택하세요.")
            return {"text": refine.ai_refine(req.text, req.mode, req.model, req.custom)}
        return refine.run_rule(req.mode, req.text, req.opts or {})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"재가공 실패: {e}") from e


@app.put("/api/docs/{doc_id}/fields")
def api_fields_save(doc_id: str, upd: FieldsUpdate):
    with _lock:
        doc = load_doc(doc_id)
        doc["fields"] = {**upd.model_dump(), "at": time.time()}
        save_doc(doc)
    return doc["fields"]


@app.post("/api/docs/{doc_id}/autofill")
def api_autofill(doc_id: str, req: AutofillReq):
    doc = load_doc(doc_id)
    t = TEMPLATES.get(req.template_id)
    if not t:
        raise HTTPException(400, "템플릿을 찾을 수 없습니다.")
    text = "\n".join(p["result"]["text"] for p in doc["pages"] if p.get("result"))
    if not text.strip():
        raise HTTPException(400, "인식된 텍스트가 없습니다.")
    vals = extract.run_template(text, t["fields"], req.model)
    with _lock:
        doc = load_doc(doc_id)
        doc["fields"] = {"template_id": t["id"], "values": [{"name": v["name"], "value": v["value"], "note": v.get("note", "")} for v in vals],
                         "checked": False, "edited": False, "at": time.time()}
        save_doc(doc)
    return doc["fields"]


# ---------------------------------------------------------------- 시스템 입력 도우미 (북마클릿)

def _helper_auth(t: str | None) -> None:
    if t != HELPER_TOKEN:
        raise HTTPException(403, "토큰이 맞지 않습니다. IUX OCR에서 도우미를 다시 설치하세요.")


def _doc_order() -> list[str]:
    return [d["id"] for d in api_docs()]


def _helper_payload() -> dict:
    order = _doc_order()
    did = CURRENT["doc_id"] if CURRENT["doc_id"] in order else (order[0] if order else None)
    if not did:
        return {"doc": None, "fields": [], "index": 0, "total": 0}
    doc = load_doc(did)
    f = doc.get("fields") or {}
    t = TEMPLATES.get(f.get("template_id"))
    return {"doc": {"id": did, "name": doc["name"]}, "template": t["name"] if t else "",
            "fields": f.get("values", []), "checked": f.get("checked", False),
            "index": order.index(did) + 1, "total": len(order)}


@app.get("/api/helper/token")
def api_helper_token(request: Request):
    # 같은 PC의 IUX OCR 화면에서만 (Origin 없음 또는 127.0.0.1)
    origin = request.headers.get("origin", "")
    if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
        raise HTTPException(403)
    return {"token": HELPER_TOKEN}


@app.post("/api/helper/select")
def api_helper_select(body: dict, t: str | None = None):
    _helper_auth(t)
    CURRENT["doc_id"] = body.get("doc_id")
    return {"ok": True}


@app.get("/api/helper/current")
def api_helper_current(t: str | None = None):
    _helper_auth(t)
    return _helper_payload()


@app.post("/api/helper/move")
def api_helper_move(body: dict, t: str | None = None):
    _helper_auth(t)
    order = _doc_order()
    if order:
        i = order.index(CURRENT["doc_id"]) if CURRENT["doc_id"] in order else 0
        if body.get("check") and CURRENT["doc_id"] in order:
            with _lock:
                d = load_doc(CURRENT["doc_id"])
                if d.get("fields"):
                    d["fields"]["checked"] = True
                    save_doc(d)
        i = max(0, min(len(order) - 1, i + int(body.get("dir", 1))))
        CURRENT["doc_id"] = order[i]
    return _helper_payload()


@app.get("/api/templates")
def api_templates():
    return {"templates": TEMPLATES.all(), "presets": {k: v[0] for k, v in extract.PRESETS.items()},
            "gongmun": extract.GONGMUN_KEYS, "formats": extract.FORMATS}


@app.put("/api/templates")
def api_templates_save(items: list[dict]):
    return {"templates": TEMPLATES.save_all(items)}


@app.post("/api/extract")
def api_extract(req: ExtractReq):
    fields = req.fields
    if fields is None:
        t = TEMPLATES.get(req.template_id)
        if not t:
            raise HTTPException(400, "템플릿을 찾을 수 없습니다.")
        fields = t["fields"]
    try:
        return {"values": extract.run_template(req.text, fields, req.model)}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"추출 실패: {e}") from e


@app.post("/api/docs/{doc_id}/export")
def api_export(doc_id: str, req: ExportReq):
    doc = load_doc(doc_id)
    if req.format not in exporters.FORMATS:
        raise HTTPException(400, "지원하지 않는 형식")
    pages = [p for p in doc["pages"] if req.pages is None or p["index"] in req.pages]
    pages = [p for p in pages if p.get("result")]
    if not pages:
        raise HTTPException(400, "인식된 페이지가 없습니다. 먼저 OCR을 실행하세요.")
    try:
        tpl = TEMPLATES.get(req.template_id)
        data = exporters.export(req.format, doc, pages, DOCS / doc_id, tpl, req.model)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"내보내기 실패: {e}") from e
    mime, ext = exporters.FORMATS[req.format]
    fname = Path(doc["name"]).stem + ext
    return Response(data, media_type=mime,
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"})


# ---------------------------------------------------------------- 폴더 선택 (윈도우/맥 로컬 대화상자)

@app.post("/api/pick-folder")
def api_pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory()
        root.destroy()
        return {"path": path or ""}
    except Exception as e:  # noqa: BLE001
        return {"path": "", "error": f"폴더 선택 창을 열 수 없습니다. 경로를 직접 입력하세요. ({e})"}


# ================================================================ 폴더 일괄 처리

BATCH: dict = {"running": False, "stop": False, "total": 0, "done": 0, "current": "", "log": [], "errors": 0,
               "started": None, "finished": None, "output_dir": ""}


def _batch_log(msg: str) -> None:
    BATCH["log"].append(time.strftime("%H:%M:%S ") + msg)
    BATCH["log"] = BATCH["log"][-300:]


def _batch_worker(req: BatchReq) -> None:
    inp, out = Path(req.input_dir), Path(req.output_dir)
    try:
        it = inp.rglob("*") if req.recursive else inp.glob("*")
        files = sorted(f for f in it if f.is_file() and (f.suffix.lower() in IMG_EXT or f.suffix.lower() == ".pdf"))
        BATCH["total"] = len(files)
        _batch_log(f"대상 파일 {len(files)}개")
        report = []
        tpl = TEMPLATES.get(req.template_id)
        ext_rows: list[list] = []
        if tpl:
            _batch_log(f"항목 추출 템플릿: {tpl['name']}")
        for f in files:
            if BATCH["stop"]:
                _batch_log("사용자가 중지했습니다.")
                break
            rel = f.relative_to(inp)
            BATCH["current"] = str(rel)
            target_dir = out / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            first = target_dir / (f.stem + exporters.FORMATS[req.formats[0]][1])
            if req.skip_existing and first.exists():
                _batch_log(f"건너뜀(이미 있음) {rel}")
                BATCH["done"] += 1
                continue
            t0 = time.time()
            try:
                doc = ingest_file(f, f.name)
                for i, p in enumerate(doc["pages"]):
                    if BATCH["stop"]:
                        break
                    if not p.get("pdf_text"):
                        ocr_page(doc, i, req.engine, req.lang, req.psm, req.model, req.pre, req.fix_spacing)
                    r = p["result"]
                    if req.clean:
                        r["text"] = refine.clean_text(r["text"])
                    if req.mask:
                        r["text"], _ = refine.mask_pii(r["text"])
                save_doc(doc)
                if tpl:
                    full = "\n".join(p["result"]["text"] for p in doc["pages"])
                    vals = extract.run_template(full, tpl["fields"], req.llm)
                    ext_rows.append([str(rel), len(doc["pages"])] + [v["value"] for v in vals])
                for fmt in req.formats:
                    data = exporters.export(fmt, doc, doc["pages"], DOCS / doc["id"])
                    (target_dir / (f.stem + exporters.FORMATS[fmt][1])).write_bytes(data)
                confs = [p["result"].get("mean_conf") for p in doc["pages"] if p["result"].get("mean_conf") is not None]
                low = sum(1 for p in doc["pages"] for l in p["result"]["lines"] if l.get("conf") is not None and l["conf"] < 70)
                report.append([str(rel), len(doc["pages"]), round(sum(confs) / len(confs), 1) if confs else "", low,
                               round(time.time() - t0, 1), "완료"])
                _batch_log(f"완료 {rel} ({len(doc['pages'])}p, {time.time() - t0:.1f}s)")
                shutil.rmtree(DOCS / doc["id"], ignore_errors=True)  # 일괄 처리는 작업공간에 남기지 않음
            except Exception as e:  # noqa: BLE001
                BATCH["errors"] += 1
                report.append([str(rel), "", "", "", round(time.time() - t0, 1), f"오류: {e}"])
                _batch_log(f"오류 {rel}: {e}")
            BATCH["done"] += 1
        if tpl and ext_rows:
            from openpyxl import Workbook
            from openpyxl.styles import Alignment, Font, PatternFill
            wb = Workbook()
            ws = wb.active
            ws.title = "추출 항목"
            ws.append(["파일", "페이지수"] + [f["name"] for f in tpl["fields"]])
            for r in ext_rows:
                ws.append([str(v)[:32000] if isinstance(v, str) else v for v in r])
            for c in ws[1]:
                c.fill, c.font = PatternFill("solid", fgColor="1F2A44"), Font(bold=True, color="FFFFFF")
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(vertical="top", wrap_text=True)
            ws.freeze_panes = "C2"
            xp = out / f"항목추출_{tpl['name']}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx".replace("/", "_")
            wb.save(xp)
            _batch_log(f"항목 추출 결과 저장: {xp.name}")
        # 처리 보고서 (CSV, 엑셀에서 바로 열림)
        import csv
        rp = out / f"OCR_처리보고서_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(rp, "w", newline="", encoding="utf-8-sig") as fo:
            w = csv.writer(fo)
            w.writerow(["파일", "페이지수", "평균신뢰도", "저신뢰줄수(<70)", "소요(초)", "상태"])
            w.writerows(report)
        _batch_log(f"보고서 저장: {rp.name}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _batch_log(f"중단: {e}")
    finally:
        BATCH["running"] = False
        BATCH["current"] = ""
        BATCH["finished"] = time.time()


@app.post("/api/batch/start")
def api_batch_start(req: BatchReq):
    if BATCH["running"]:
        raise HTTPException(409, "이미 일괄 처리가 진행 중입니다.")
    if not Path(req.input_dir).is_dir():
        raise HTTPException(400, "입력 폴더가 존재하지 않습니다.")
    if not req.output_dir:
        raise HTTPException(400, "출력 폴더를 지정하세요.")
    bad = [f for f in req.formats if f not in exporters.FORMATS]
    if bad or not req.formats:
        raise HTTPException(400, "내보낼 형식을 선택하세요.")
    Path(req.output_dir).mkdir(parents=True, exist_ok=True)
    BATCH.update({"running": True, "stop": False, "total": 0, "done": 0, "current": "", "log": [], "errors": 0,
                  "started": time.time(), "finished": None, "output_dir": req.output_dir})
    threading.Thread(target=_batch_worker, args=(req,), daemon=True).start()
    return {"ok": True}


@app.get("/api/batch/status")
def api_batch_status():
    return BATCH


@app.post("/api/batch/stop")
def api_batch_stop():
    BATCH["stop"] = True
    return {"ok": True}


# ================================================================ 실행

def main() -> None:
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("IUX_OCR_PORT", 8765)))
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    print(f"\n  IUX OCR 실행 중 → {url}\n  (이 창을 닫으면 프로그램이 종료됩니다)\n")
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
