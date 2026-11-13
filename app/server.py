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
import numpy as np
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


def ingest_into(holder: dict, items: list[tuple[Path, str]]) -> None:
    """holder(id, pages)에 쪽을 만들어 붙이기만 함 (저장은 호출한 쪽에서)."""
    tmpdoc = ingest_files(items, "", {"id": holder["id"], "pages": holder["pages"], "_nosave": True})
    holder["pages"] = tmpdoc["pages"]


def ingest_file(src: Path, name: str) -> dict:
    """이미지/PDF 한 개를 문서로 등록."""
    return ingest_files([(src, name)], name)


def ingest_files(items: list[tuple[Path, str]], doc_name: str, existing: dict | None = None) -> dict:
    """여러 이미지/PDF를 한 문서(여러 쪽)로 등록 (페이지별 PNG + 썸네일). 폴더 단위 기록물용.
    existing 을 주면 그 문서 뒤에 쪽을 이어 붙인다."""
    if existing:
        doc_id = existing["id"]
        d = DOCS / doc_id
        pages: list[dict] = existing["pages"]
    else:
        doc_id = uuid.uuid4().hex[:12]
        d = DOCS / doc_id
        (d / "pages").mkdir(parents=True)
        (d / "thumbs").mkdir()
        pages = []
    start_len = len(pages)

    def new_page(img, dpi: int, source: str) -> dict:
        i = len(pages)
        key = uuid.uuid4().hex[:10]  # 쪽 순서가 바뀌어도 겹치지 않는 파일 이름
        fn = f"pages/{key}.png"
        img.save(d / fn)
        pinfo = {"index": i, "key": key, "file": fn, "proc": f"pages/{key}_proc.png", "thumb": f"thumbs/{key}.jpg",
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
                    try:  # 휴대폰·스캐너 사진의 회전 정보(EXIF) 반영
                        from PIL import ImageOps
                        fr = ImageOps.exif_transpose(fr)
                    except Exception:  # noqa: BLE001
                        pass
                    if fr.mode not in ("RGB", "L"):
                        fr = fr.convert("RGB")
                    dpi = round(float(dpi_info[0])) if dpi_info and float(dpi_info[0]) >= 72 else 300
                    new_page(fr, dpi, name)
            else:
                raise HTTPException(400, f"지원하지 않는 파일 형식입니다: {name}")
    except Exception:
        if existing:
            del pages[start_len:]
        else:
            shutil.rmtree(d, ignore_errors=True)
        raise

    if existing and existing.get("_nosave"):
        return existing
    if existing:
        doc = existing
        doc.setdefault("files", [])
        doc["files"] += [n for _, n in items]
    else:
        doc = {"id": doc_id, "name": doc_name, "created": time.time(), "pages": pages}
        if len(items) > 1:
            doc["files"] = [n for _, n in items]
    save_doc(doc)
    return doc


def rotate_page(doc: dict, idx: int, deg: int) -> dict:
    """페이지 이미지를 시계 방향으로 deg(90/180/270)만큼 돌려 저장. 인식 결과는 지운다."""
    import cv2 as _cv
    d = DOCS / doc["id"]
    p = doc["pages"][idx]
    code = {90: _cv.ROTATE_90_CLOCKWISE, 180: _cv.ROTATE_180, 270: _cv.ROTATE_90_COUNTERCLOCKWISE}.get(deg % 360)
    if code is None:
        return p
    img = _cv.rotate(preprocess.load_bgr(str(d / p["file"])), code)
    preprocess.save_png(str(d / p["file"]), img)
    p["height"], p["width"] = img.shape[:2]
    p["rev"] = p.get("rev", 0) + 1
    _thumb(d / p["file"], d / p["thumb"])
    (d / p["proc"]).unlink(missing_ok=True)
    p["result"] = None
    p.pop("pdf_text", None)
    return p


def ocr_page(doc: dict, idx: int, engine: str, lang: str, psm: int, model: str | None, pre: dict | None, fix_spacing: bool = True) -> dict:
    d = DOCS / doc["id"]
    p = doc["pages"][idx]
    color = preprocess.load_bgr(str(d / p["file"]))
    gray, info = preprocess.preprocess(color, pre)
    if info.get("rotate"):  # 돌아간 페이지를 찾았으면 원본도 바로 세워서 저장 (화면·썸네일도 똑바로)
        rotate_page(doc, idx, info["rotate"])
        color = preprocess.load_bgr(str(d / p["file"]))
        info["rotate_fixed"] = info.pop("rotate")
    if engine == "iux" and info.get("skew"):  # 기울어진 문서: 원본 색 이미지도 같은 각도로 바로 세워서 읽는다 (표 선·글줄이 수평이 되게)
        color = preprocess._rotate(color, info["skew"])
    # IUX 엔진은 원본을 그대로 읽으므로, 화면 박스가 어긋나지 않게 전처리 보기에도 원본(흑백)을 둔다
    preprocess.save_png(str(d / p["proc"]), cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if engine == "iux" and color.ndim == 3 else gray)
    res = engines.run_engine(engine, gray, color, lang, psm, model, fix_spacing)
    if (pre or {}).get("second", True):
        res = second_pass(res, gray, engine, lang, model)
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


class RegionReq(BaseModel):
    page: int
    box: list[float]  # [x0, y0, x1, y1] 0~1
    view: str = "image"
    engine: str = "tesseract"
    lang: str = "kor+eng"
    model: str | None = None


class RotateReq(BaseModel):
    pages: list[int]
    deg: int = 90


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
    tables: list[dict] | None = None


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


class ExportAllReq(BaseModel):
    format: str
    doc_ids: list[str] | None = None
    template_id: str | None = None
    model: str | None = None
    name: str | None = None


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
            pend = sum(1 for p in d["pages"] if p.get("pending"))
            f = d.get("fields") or {}
            out.append({"id": d["id"], "name": d["name"], "created": d["created"], "pages": len(d["pages"]), "done": done,
                        "checked": bool(f.get("checked")), "filled": sum(1 for v in f.get("values", []) if v.get("value")), "pending": pend,
                        "folder": d.get("folder") or ""})
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
async def api_upload(files: list[UploadFile] = File(...), merge: str = Form(""), doc_name: str = Form(""), append_to: str = Form(""), folder: str = Form("")):
    """merge=1 이면 받은 파일 전체를 한 문서(여러 쪽)로, append_to=문서ID 면 그 문서 뒤에 쪽 추가, 아니면 파일마다 문서."""
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
        if append_to and staged:
            with _lock:
                docs.append(ingest_files(staged, "", load_doc(append_to)))
        elif merge in ("1", "true") and staged:
            d_ = ingest_files(staged, doc_name or staged[0][1])
            if folder:  # 폴더로 넣은 문서: 탐색기 보기에서 폴더 구조로 묶음
                d_["folder"] = folder
                save_doc(d_)
            docs.append(d_)
        else:
            for t, name in staged:
                docs.append(ingest_file(t, name))
    finally:
        for t, _ in staged:
            t.unlink(missing_ok=True)
    return docs


class LazyDocReq(BaseModel):
    name: str
    files: list[str]  # 폴더 안 상대 경로 (정렬된 순서)
    folder: str = ""  # 폴더 상대 경로 (탐색기 보기용)


@app.post("/api/docs/lazy")
def api_docs_lazy(req: LazyDocReq):
    """검수 프로그램처럼 파일 목록만 먼저 등록. 이미지는 변환할 때 한 장씩 불러온다."""
    doc_id = uuid.uuid4().hex[:12]
    d = DOCS / doc_id
    (d / "pages").mkdir(parents=True)
    (d / "thumbs").mkdir()
    pages = [{"index": i, "key": uuid.uuid4().hex[:10], "pending": True, "path": f, "source": Path(f).name, "result": None}
             for i, f in enumerate(req.files)]
    doc = {"id": doc_id, "name": req.name, "created": time.time(), "pages": pages, "lazy": True, "folder": req.folder}
    save_doc(doc)
    return doc


@app.post("/api/docs/{doc_id}/fill")
async def api_fill(doc_id: str, key: str = Form(...), files: list[UploadFile] = File(...)):
    """대기 중인 쪽(key)에 실제 파일을 채움. PDF·여러 쪽 TIFF면 그 자리에 여러 쪽으로 펼침."""
    f = files[0]
    tmp = WS / "tmp"
    tmp.mkdir(exist_ok=True)
    name = _decode_filename(Path(f.filename or "file").name)
    t = tmp / f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
    with open(t, "wb") as fo:
        shutil.copyfileobj(f.file, fo)
    if t.suffix not in IMG_EXT and t.suffix != ".pdf":
        ext = _sniff_ext(t)
        if ext:
            t2 = t.with_suffix(ext)
            t.rename(t2)
            t = t2
    try:
        with _lock:
            doc = load_doc(doc_id)
            pos = next((i for i, p in enumerate(doc["pages"]) if p.get("key") == key), None)
            if pos is None:
                raise HTTPException(404, "대기 중인 쪽을 찾을 수 없습니다.")
            holder = doc["pages"][pos]
            temp = {"id": doc_id, "pages": []}
            ingest_into(temp, [(t, name)])
            new_pages = temp["pages"]
            for q in new_pages:
                q["path"] = holder.get("path")
            doc["pages"][pos:pos + 1] = new_pages
            for i, q in enumerate(doc["pages"]):
                q["index"] = i
            save_doc(doc)
    finally:
        t.unlink(missing_ok=True)
    return {"doc": doc, "keys": [q["key"] for q in new_pages]}


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
    if not key or p.get("pending"):
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
    if doc["pages"][req.page].get("pending"):
        raise HTTPException(400, "아직 불러오지 않은 파일입니다.")
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


@app.post("/api/docs/{doc_id}/rotate")
def api_rotate(doc_id: str, req: RotateReq):
    with _lock:
        doc = load_doc(doc_id)
        for i in req.pages:
            if 0 <= i < len(doc["pages"]):
                rotate_page(doc, i, req.deg)
        save_doc(doc)
    return doc


class AutoOrientReq(BaseModel):
    pages: list[int]


@app.post("/api/docs/{doc_id}/auto-orient")
def api_auto_orient(doc_id: str, req: AutoOrientReq):
    """인식 없이 방향만 감지해 돌아간 쪽을 바로 세움."""
    fixed = []
    for i in req.pages:
        with _lock:
            doc = load_doc(doc_id)
        if not 0 <= i < len(doc["pages"]):
            continue
        p = doc["pages"][i]
        img = preprocess.load_bgr(str(DOCS / doc_id / p["file"]))
        h, w = img.shape[:2]
        s_ = min(1.0, 1800 / max(h, w))
        small = cv2.resize(img, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA) if s_ < 1 else img
        deg = preprocess.detect_orientation(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        if deg:
            with _lock:
                doc = load_doc(doc_id)
                rotate_page(doc, i, deg)
                save_doc(doc)
            fixed.append({"page": i, "deg": deg})
    return {"fixed": fixed, "doc": load_doc(doc_id)}


@app.post("/api/docs/{doc_id}/ocr-region")
def api_ocr_region(doc_id: str, req: RegionReq):
    """화면에서 드래그한 영역만 잘라 인식 (선택 영역 복사용)."""
    doc = load_doc(doc_id)
    if not 0 <= req.page < len(doc["pages"]):
        raise HTTPException(400, "페이지 번호 오류")
    p = doc["pages"][req.page]
    path = DOCS / doc_id / (p["proc"] if req.view == "proc" and (DOCS / doc_id / p["proc"]).exists() else p["file"])
    img = preprocess.load_bgr(str(path))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [max(0.0, min(1.0, v)) for v in (req.box + [0, 0, 1, 1])[:4]]
    # 선택한 영역만 정확히 자르고(이웃 줄 침범 X) 바깥에 흰 여백을 둔다
    X0, Y0 = int(x0 * w), int(y0 * h)
    X1, Y1 = min(w, int(round(x1 * w))), min(h, int(round(y1 * h)))
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        raise HTTPException(400, "선택 영역이 너무 작습니다.")
    crop = img[Y0:Y1, X0:X1].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    _trim_edge_fragments(gray, crop)
    if gray.shape[0] < 120:  # 작은 영역은 키워서 인식률 올림
        s_ = min(3.0, 120 / max(1, gray.shape[0]))
        gray = cv2.resize(gray, None, fx=s_, fy=s_, interpolation=cv2.INTER_CUBIC)
        crop = cv2.resize(crop, None, fx=s_, fy=s_, interpolation=cv2.INTER_CUBIC)
    m = 16
    gray = cv2.copyMakeBorder(gray, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    crop = cv2.copyMakeBorder(crop, m, m, m, m, cv2.BORDER_CONSTANT, value=(255, 255, 255) if crop.ndim == 3 else 255)
    try:
        res = engines.run_engine(req.engine, gray, crop, req.lang, 6, req.model)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"영역 인식 실패: {e}") from e
    if res.get("lines"):
        text = "\n".join(refine.drop_stray_lines([(l.get("text", ""), l.get("conf")) for l in res["lines"]]))
    else:
        text = "\n".join(refine.drop_stray_lines([(t, None) for t in (res.get("text") or "").splitlines()]))
    return {"text": text, "mean_conf": res.get("mean_conf")}


SECOND_CONF = 80      # 이 신뢰도 미만인 줄만 2차 인식
SECOND_MAX = 60       # 한 페이지에서 다시 읽을 최대 줄 수
SECOND_H = 64         # 다시 읽을 때 줄 높이(px) — 작은·얇은 글씨는 키워서 읽음


_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def _count(t: str | None, rx) -> int:
    return len(rx.findall(t or ""))


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t or "")


# 2차 인식 변형: (가로 늘림 배율, 획 굵게) — 길쭉·얇은 글씨는 가로로 늘리고 획을 굵히면 크게 좋아짐
SECOND_VARIANTS = [(1.0, False), (1.4, False), (1.8, False), (1.6, True)]


def second_pass(res: dict, gray, engine: str, lang: str, model: str | None) -> dict:
    """2차 인식: 신뢰도 낮은 줄만 잘라 키우고(작은 글씨), 가로로 늘리고(길쭉한 글씨), 획을 굵혀(얇은 글씨)
    여러 버전을 한 장에 모아 한 번에 다시 읽은 뒤, 가장 또렷하게 읽힌 버전이 확실히 나을 때만 바꾼다."""
    if engine != "tesseract" or not res.get("lines"):
        return res
    lines = res["lines"]
    H, W = gray.shape[:2]
    cand = [i for i, l in enumerate(lines)
            if l.get("box") and (l.get("conf") is None or l["conf"] < SECOND_CONF) and len(_norm(l.get("text"))) >= 1]
    cand = sorted(cand, key=lambda i: lines[i].get("conf") or 0)[:SECOND_MAX]
    if not cand:
        return res
    tiles, slots, y, gap = [], [], 0, 28
    kernel = np.ones((2, 2), np.uint8)
    for i in cand:
        x, yy, w, h = lines[i]["box"]
        padx, pady = int(h * H * 0.5), int(h * H * 0.25)
        X0, Y0 = max(0, int(x * W) - padx), max(0, int(yy * H) - pady)
        X1, Y1 = min(W, int(round((x + w) * W)) + padx), min(H, int(round((yy + h) * H)) + pady)
        if X1 - X0 < 6 or Y1 - Y0 < 6:
            continue
        g0 = gray[Y0:Y1, X0:X1].copy()
        _trim_edge_fragments(g0, g0)
        s_ = max(1.0, min(4.0, SECOND_H / max(1, g0.shape[0])))
        if s_ > 1.0:
            g0 = cv2.resize(g0, None, fx=s_, fy=s_, interpolation=cv2.INTER_CUBIC)
        for v, (sx, thick) in enumerate(SECOND_VARIANTS):
            g = cv2.resize(g0, None, fx=sx, fy=1, interpolation=cv2.INTER_CUBIC) if sx != 1 else g0
            if thick:
                g = cv2.erode(g, kernel)
            tiles.append(g)
            slots.append((i, v, y + gap, y + gap + g.shape[0]))
            y += gap + g.shape[0]
    if not tiles:
        return res
    TW = max(t.shape[1] for t in tiles) + 32
    sheet = np.full((y + gap, TW), 255, np.uint8)
    for g, (_, _, a, b) in zip(tiles, slots):
        sheet[a:b, 16:16 + g.shape[1]] = g
    try:
        r2 = engines.run_engine("tesseract", sheet, cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR), lang, 6, model)
    except Exception:
        return res
    SH = sheet.shape[0]
    got: dict[tuple, list] = {}
    for l in r2.get("lines") or []:
        if not l.get("box") or not _norm(l.get("text")):
            continue
        cy = (l["box"][1] + l["box"][3] / 2) * SH
        for i, v, a, b in slots:
            if a - gap / 2 <= cy <= b + gap / 2:
                got.setdefault((i, v), []).append(l)
                break
    best: dict[int, tuple] = {}
    for (i, v), ls in got.items():
        ls.sort(key=lambda l: l["box"][0])
        new = " ".join(refine.drop_stray_lines([(l["text"], l.get("conf")) for l in ls])).strip()
        if "  " not in (lines[i].get("text") or ""):
            new = re.sub(r" {2,}", " ", new)
        confs = [(l["conf"], len(_norm(l["text"]))) for l in ls if l.get("conf") is not None]
        n = sum(k for _, k in confs)
        if not new or not n:
            continue
        nc = sum(c * k for c, k in confs) / n
        if i not in best or nc > best[i][1]:
            best[i] = (new, nc, v)
    improved = 0
    for i, (new, nc, v) in best.items():
        old = lines[i]
        oc = old.get("conf") or 0
        on, nn = len(_norm(old.get("text"))), len(_norm(new))
        if nc < 60 or nc < oc + 5:
            continue
        if on and not (on * 0.6 <= nn <= on * 1.6 + 2):
            continue
        # 신뢰도만 높은 엉터리(영문 대문자 조합 등) 방지: 한글이 줄거나 영문이 늘면 버림
        ok_, nk = _count(old.get("text"), _HANGUL), _count(new, _HANGUL)
        ol, nl = _count(old.get("text"), _LATIN), _count(new, _LATIN)
        if nk < ok_ - 1 or nl > ol + 2:
            continue
        old["text_1st"], old["conf_1st"] = old.get("text"), old.get("conf")
        old["text"], old["conf"], old["second"] = new, round(nc, 1), True
        improved += 1
    if improved:
        txt, pos = res.get("text") or "", 0
        for l in lines:
            if l.get("second") and l.get("text_1st"):
                k = txt.find(l["text_1st"], pos)
                if k >= 0:
                    txt = txt[:k] + l["text"] + txt[k + len(l["text_1st"]):]
                    pos = k + len(l["text"])
        res["text"] = txt
        cs = [(l.get("conf"), len(_norm(l.get("text")))) for l in lines if l.get("conf") is not None]
        tot = sum(n for _, n in cs)
        if tot:
            res["mean_conf"] = round(sum(c * n for c, n in cs) / tot, 1)
    res["second"] = {"tried": len(cand), "improved": improved}
    return res


def _trim_edge_fragments(gray, crop) -> None:
    """선택 위/아래 경계에 걸쳐 잘린 옆 줄 글자 조각을 흰색으로 지운다 (제자리 수정)."""
    h = gray.shape[0]
    if h < 12:
        return
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows = ink.sum(axis=1) > max(1, gray.shape[1] * 0.002)
    bands, start = [], None
    for i, r in enumerate(rows):
        if r and start is None:
            start = i
        elif not r and start is not None:
            bands.append((start, i)); start = None
    if start is not None:
        bands.append((start, h))
    if len(bands) < 2:
        return
    tallest = max(b - a for a, b in bands)
    for a, b in (bands[0], bands[-1]):
        if (a == 0 or b == h) and (b - a) < tallest * 0.6:
            gray[a:b] = 255
            crop[a:b] = 255


@app.put("/api/docs/{doc_id}/pages/{idx}")
def api_page_update(doc_id: str, idx: int, upd: PageUpdate):
    with _lock:
        doc = load_doc(doc_id)
        p = doc["pages"][idx]
        r = p.get("result") or {"engine": "manual", "lines": [], "mean_conf": None}
        r["text"] = upd.text
        if upd.lines is not None:
            r["lines"] = upd.lines
        if upd.tables is not None:
            r["tables"] = upd.tables
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
    vals = extract.values_for(t, text, req.model)
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


@app.post("/api/export-all")
def api_export_all(req: ExportAllReq):
    """여러 문서(폴더 전체)의 인식 텍스트를 하나의 파일로 합쳐 내보내기."""
    if req.format not in exporters.FORMATS:
        raise HTTPException(400, "지원하지 않는 형식")
    ids = req.doc_ids or [d["id"] for d in reversed(api_docs())]
    pages, skipped = [], 0
    for did in ids:
        d = load_doc(did)
        multi = len(d["pages"]) > 1
        for p in d["pages"]:
            if not p.get("result"):
                skipped += 1
                continue
            q = dict(p)
            q["label"] = f"{d['name']} · {p['index'] + 1}쪽" if multi else d["name"]
            q["doc_name"] = d["name"]
            q["_dir"] = str(DOCS / did)
            pages.append(q)
    if not pages:
        raise HTTPException(400, "인식된 페이지가 없습니다. 먼저 '모두 인식'을 실행하세요.")
    for i, q in enumerate(pages):
        q["index"] = i
    name = req.name or f"OCR_전체_{len(ids)}건_{time.strftime('%Y%m%d')}"
    doc = {"id": "all", "name": name, "pages": pages}
    try:
        tpl = TEMPLATES.get(req.template_id)
        data = exporters.export(req.format, doc, pages, DOCS, tpl, req.model)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"내보내기 실패: {e}") from e
    mime, ext = exporters.FORMATS[req.format]
    return Response(data, media_type=mime, headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name + ext)}",
        "X-Skipped-Pages": str(skipped), "Access-Control-Expose-Headers": "X-Skipped-Pages"})


@app.get("/api/templates")
def api_templates():
    return {"templates": [extract.AUTO_TEMPLATE] + TEMPLATES.all(), "presets": {k: v[0] for k, v in extract.PRESETS.items()},
            "gongmun": extract.GONGMUN_KEYS, "formats": extract.FORMATS}


@app.put("/api/templates")
def api_templates_save(items: list[dict]):
    return {"templates": TEMPLATES.save_all(items)}


@app.post("/api/extract")
def api_extract(req: ExtractReq):
    fields = req.fields
    t = None
    if fields is None:
        t = TEMPLATES.get(req.template_id)
        if not t:
            raise HTTPException(400, "템플릿을 찾을 수 없습니다.")
    try:
        if t is not None:
            return {"values": extract.values_for(t, req.text, req.model)}
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


# ---------------------------------------------------------------- 작업 파일 (.iuxocr): 인식 상태 그대로 저장·다시 열기

PROJ_FMT = "iuxocr"
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class ProjectExportReq(BaseModel):
    doc_ids: list[str]
    name: str | None = None


def _data_url(path: Path) -> str | None:
    if not path.exists():
        return None
    import base64
    return f"data:{_MIME.get(path.suffix.lower(), 'image/png')};base64," + base64.b64encode(path.read_bytes()).decode()


@app.post("/api/project/export")
def api_project_export(req: ProjectExportReq):
    """문서(원본 이미지 + 인식 결과·고친 글자·표·항목)를 작업 파일 하나로 묶는다. 다른 PC·웹에서 그대로 이어서 작업."""
    out_docs, skipped = [], 0
    for did in req.doc_ids:
        d = load_doc(did)
        dd = DOCS / did
        pages, files = [], {}
        for p in d["pages"]:
            if p.get("pending"):
                skipped += 1  # 아직 불러오지 않은 쪽(폴더 원본 경로만 있음)은 담을 수 없음
                continue
            key = p.get("key") or uuid.uuid4().hex[:10]
            for kind, rel in (("image", p.get("file")), ("proc", p.get("proc")), ("thumb", p.get("thumb"))):
                u = _data_url(dd / rel) if rel else None
                if u:
                    files[f"{kind}/{key}"] = u
            q = {k: v for k, v in p.items() if k not in ("file", "proc", "thumb", "path")}
            q["key"] = key
            pages.append(q)
        for i, q in enumerate(pages):
            q["index"] = i
        out_docs.append({"name": d["name"], "created": d.get("created"), "fields": d.get("fields"), "pages": pages, "files": files})
    if not out_docs:
        raise HTTPException(400, "저장할 문서가 없습니다.")
    data = json.dumps({"format": PROJ_FMT, "version": 1, "app": "IUX OCR", "saved": time.time(), "docs": out_docs}, ensure_ascii=False).encode()
    name = req.name or (out_docs[0]["name"] if len(out_docs) == 1 else f"IUX작업_{len(out_docs)}건_{time.strftime('%Y%m%d')}")
    fname = Path(name).stem + ".iuxocr"
    return Response(data, media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}", "x-skipped-pages": str(skipped)})


@app.post("/api/project/import")
async def api_project_import(files: list[UploadFile] = File(...)):
    """작업 파일(.iuxocr)을 열어 문서 목록에 새 문서로 추가한다 (기존 문서는 건드리지 않음)."""
    import base64
    made = []
    for f in files:
        try:
            pj = json.loads((await f.read()).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"작업 파일을 읽을 수 없습니다: {f.filename}") from e
        if pj.get("format") != PROJ_FMT:
            raise HTTPException(400, f"IUX OCR 작업 파일이 아닙니다: {f.filename}")
        for src in pj.get("docs", []):
            did = uuid.uuid4().hex[:12]
            dd = DOCS / did
            (dd / "pages").mkdir(parents=True)
            (dd / "thumbs").mkdir()
            pages = []
            for p in src.get("pages", []):
                key = re.sub(r"[^A-Za-z0-9_-]", "", str(p.get("key") or "")) or uuid.uuid4().hex[:10]
                q = dict(p)
                q["key"] = key
                for kind, folder, suffix in (("image", "pages", ""), ("proc", "pages", "_proc"), ("thumb", "thumbs", "")):
                    u = (src.get("files") or {}).get(f"{kind}/{p.get('key')}")
                    ext = ".png"
                    if u and u.startswith("data:"):
                        head, b64 = u.split(",", 1)
                        ext = {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(head[5:].split(";")[0], ".png")
                        (dd / folder / f"{key}{suffix}{ext}").write_bytes(base64.b64decode(b64))
                    q[{"image": "file", "proc": "proc", "thumb": "thumb"}[kind]] = f"{folder}/{key}{suffix}{ext}"
                if not (dd / q["file"]).exists():
                    continue
                if not (dd / q["thumb"]).exists():
                    _thumb(dd / q["file"], dd / q["thumb"])
                q.pop("pending", None)
                pages.append(q)
            if not pages:
                shutil.rmtree(dd, ignore_errors=True)
                continue
            for i, q in enumerate(pages):
                q["index"] = i
            doc = {"id": did, "name": src.get("name") or f.filename, "created": time.time(), "pages": pages}
            if src.get("fields"):
                doc["fields"] = src["fields"]
            save_doc(doc)
            made.append(doc)
    if not made:
        raise HTTPException(400, "작업 파일에 문서가 없습니다.")
    return made


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
                    vals = extract.values_for(tpl, full, req.llm)
                    ext_rows.append((str(rel), len(doc["pages"]), [(v["name"], v["value"]) for v in vals]))
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
            cols: list[str] = []  # 자동 항목은 파일마다 항목이 달라 전체 합집합을 열로
            for _, _, kv in ext_rows:
                for n, _v in kv:
                    if n not in cols:
                        cols.append(n)
            ws.append(["파일", "페이지수"] + cols)
            for rel_, npg, kv in ext_rows:
                d_ = dict(kv)
                ws.append([rel_, npg] + [str(d_.get(c, ""))[:32000] for c in cols])
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
