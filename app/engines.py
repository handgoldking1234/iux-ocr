"""OCR 엔진 어댑터 — Tesseract / PaddleOCR / Ollama(비전 모델).

모든 엔진은 같은 형태의 결과를 돌려준다:
{
  "engine": "tesseract", "lines": [{"text", "conf"(0~100|None), "box"([x,y,w,h] 0~1 정규화|None), "block"}],
  "text": "...", "mean_conf": float|None
}
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

# ---------------------------------------------------------------- Tesseract

_tess_cmd: str | None = None
_tess_dir: str | None = None


def _find_tesseract() -> str | None:
    cands = [
        os.environ.get("TESSERACT_CMD"),
        str(Path(sys.executable).parent / "bin" / "tesseract" / "tesseract.exe"),  # 설치형(exe) 동봉 위치
        str(ROOT / "bin" / "tesseract" / "tesseract.exe"),   # 포터블 동봉 위치
        str(ROOT / "bin" / "tesseract" / "tesseract"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def tesseract_info() -> dict:
    global _tess_cmd, _tess_dir
    cmd = _find_tesseract()
    if not cmd:
        return {"available": False, "reason": "Tesseract가 설치되어 있지 않습니다."}
    try:
        import pytesseract
    except ImportError:
        return {"available": False, "reason": "pytesseract 패키지가 없습니다."}
    pytesseract.pytesseract.tesseract_cmd = cmd
    _tess_cmd = cmd
    # 프로그램 폴더의 tessdata(예: kor 고정밀 모델)를 우선 사용
    local = ROOT / "tessdata"
    if not (local / "kor.traineddata").exists():
        exe_td = Path(sys.executable).parent / "tessdata"
        if (exe_td / "kor.traineddata").exists():
            local = exe_td
    _tess_dir = str(local) if (local / "kor.traineddata").exists() else None
    if _tess_dir is None:
        bundled = Path(cmd).parent / "tessdata"
        if (bundled / "kor.traineddata").exists():
            _tess_dir = str(bundled)
    # 윈도우 Tesseract는 경로에 한글이 있으면 언어 데이터를 못 읽는다
    # (예: C:\Users\홍길동\AppData\Local\Programs\IUX OCR) → 영문 경로로 복사해 사용
    if _tess_dir and os.name == "nt" and not _tess_dir.isascii():
        _tess_dir = _mirror_ascii(_tess_dir) or _tess_dir
    try:
        langs = pytesseract.get_languages(config=_tess_cfg(""))
        ver = str(pytesseract.get_tesseract_version())
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"Tesseract 실행 실패: {e}"}
    has_kor = "kor" in langs
    return {
        "available": True, "version": ver, "path": cmd, "langs": langs, "has_kor": has_kor,
        "reason": None if has_kor else "한국어(kor) 언어 데이터가 없습니다. 설치 시 Korean을 체크하세요.",
    }


def _mirror_ascii(src: str) -> str | None:
    """언어 데이터(*.traineddata)를 한글 없는 고정 경로로 복사하고 그 경로를 돌려준다."""
    cands = [os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "IUXOCR", "tessdata"),
             os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), "IUXOCR", "tessdata"),
             r"C:\IUXOCR\tessdata"]
    files = [f for f in os.listdir(src) if f.endswith(".traineddata")]
    for dst in cands:
        if not dst.isascii():
            continue
        try:
            os.makedirs(dst, exist_ok=True)
            for f in files:
                a, b = os.path.join(src, f), os.path.join(dst, f)
                if not os.path.exists(b) or os.path.getsize(a) != os.path.getsize(b):
                    shutil.copy2(a, b)
            for sub in ("configs", "tessconfigs"):  # psm 등 설정 파일
                if os.path.isdir(os.path.join(src, sub)) and not os.path.isdir(os.path.join(dst, sub)):
                    shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub))
            return dst
        except OSError:
            continue
    return None


def _tess_cfg(extra: str) -> str:
    cfg = extra
    if _tess_dir:
        cfg = f'--tessdata-dir "{_tess_dir}" ' + cfg
    return cfg


def run_tesseract(gray: np.ndarray, lang: str = "kor+eng", psm: int = 4, fix_spacing: bool = True) -> dict:
    import pytesseract
    if _tess_cmd is None:
        tesseract_info()
    h, w = gray.shape[:2]
    cfg = _tess_cfg(f"--oem 1 --psm {psm} -c preserve_interword_spaces=1")
    try:  # 한 번의 인식으로 TSV(좌표)와 TXT(띄어쓰기 정확)를 함께 받는다
        txt_out, tsv_out = _tess_txt_tsv(gray, lang, psm)
        d = _parse_tsv(tsv_out)
    except Exception as e:  # noqa: BLE001
        print("[IUX OCR] Tesseract 직접 실행 실패, pytesseract로 재시도:", e)
        txt_out = None
        try:
            import tempfile
            tempfile.tempdir = _ascii_tmp_root() or tempfile.tempdir
            d = pytesseract.image_to_data(gray, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(f"{e} / {e2}") from e2
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []
    for i, word in enumerate(d["text"]):
        word = (word or "").strip()
        if not word:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"words": [], "confs": [], "x1": 1e9, "y1": 1e9, "x2": 0, "y2": 0, "block": d["block_num"][i] * 100 + d["par_num"][i]}
            order.append(key)
        x, y, bw, bh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        g["words"].append(word)
        c = float(d["conf"][i])
        if c >= 0:
            g["confs"].append(c)
        g["x1"], g["y1"] = min(g["x1"], x), min(g["y1"], y)
        g["x2"], g["y2"] = max(g["x2"], x + bw), max(g["y2"], y + bh)
    lines = []
    for k in order:
        g = groups[k]
        conf = round(sum(g["confs"]) / len(g["confs"]), 1) if g["confs"] else None
        lines.append({
            "text": " ".join(g["words"]),
            "conf": conf,
            "box": [g["x1"] / w, g["y1"] / h, (g["x2"] - g["x1"]) / w, (g["y2"] - g["y1"]) / h],
            "block": g["block"],
        })
    # 한국어는 image_to_data의 '단어'가 음절 단위로 쪼개져 띄어쓰기가 망가진다.
    # Tesseract 자체 문자열 출력(image_to_string)의 띄어쓰기를 줄 단위로 맞춰 가져온다.
    aligned = False
    if "kor" in lang or "jpn" in lang or "chi" in lang:
        try:
            s = txt_out if txt_out is not None else pytesseract.image_to_string(gray, lang=lang, config=cfg)
            str_lines = [x.strip() for x in s.splitlines() if x.strip()]
            nos = lambda t: "".join(t.split())
            if len(str_lines) == len(lines):
                for l, t in zip(lines, str_lines):
                    if nos(t) == nos(l["text"]):
                        l["text"] = t
                aligned = True
        except Exception:  # noqa: BLE001
            pass
    if fix_spacing and "kor" in lang:
        from .refine import fix_syllable_spacing
        for l in lines:
            if " " in l["text"] and (not aligned or _looks_split(l["text"])):
                l["text"] = fix_syllable_spacing(l["text"])
    return _finish("tesseract", lines)


def _ascii_tmp_root() -> str | None:
    """윈도우 사용자 이름이 한글이면 임시 폴더 경로에 한글이 들어가
    OpenCV·Tesseract가 파일을 못 읽는다 → 영문 경로의 임시 폴더를 고른다."""
    import tempfile
    cands = [tempfile.gettempdir()]
    if os.name == "nt":
        cands += [os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "IUXOCR_tmp"),
                  os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), "IUXOCR", "tmp"),
                  r"C:\IUXOCR_tmp"]
    for c in cands:
        if not c or not c.isascii():
            continue
        try:
            os.makedirs(c, exist_ok=True)
            probe = os.path.join(c, f".w{os.getpid()}")
            with open(probe, "w") as f:
                f.write("1")
            os.remove(probe)
            return c
        except OSError:
            continue
    return None


def _tess_txt_tsv(gray: np.ndarray, lang: str, psm: int) -> tuple[str, str]:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(dir=_ascii_tmp_root()) as td:
        src = os.path.join(td, "in.png")
        ok, buf = cv2.imencode(".png", gray)
        if not ok:
            raise RuntimeError("이미지 변환 실패")
        buf.tofile(src)  # 한글 경로에서도 안전
        base = os.path.join(td, "out")
        cmd = [_tess_cmd, src, base, "-l", lang, "--oem", "1", "--psm", str(psm), "-c", "preserve_interword_spaces=1"]
        if _tess_dir:
            cmd[3:3] = ["--tessdata-dir", _tess_dir]
        cmd += ["txt", "tsv"]
        flags = 0x08000000 if os.name == "nt" else 0  # 윈도우에서 콘솔 창 숨김
        pr = subprocess.run(cmd, capture_output=True, creationflags=flags, timeout=600)
        if pr.returncode != 0 or not os.path.exists(base + ".tsv"):
            err = (pr.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError("Tesseract 실행 오류: " + (err[-1] if err else f"코드 {pr.returncode}"))
        with open(base + ".txt", encoding="utf-8") as f:
            txt = f.read()
        with open(base + ".tsv", encoding="utf-8") as f:
            tsv = f.read()
    return txt, tsv


def _parse_tsv(tsv: str) -> dict:
    rows = tsv.splitlines()
    head = rows[0].split("\t")
    d: dict = {k: [] for k in head}
    ints = {"level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height"}
    for r in rows[1:]:
        parts = r.split("\t")
        if len(parts) < len(head):
            parts += [""] * (len(head) - len(parts))
        for k, v in zip(head, parts):
            d[k].append(int(v) if k in ints and v.lstrip("-").isdigit() else (float(v) if k == "conf" and v else v))
    return d


def _looks_split(t: str) -> bool:
    """한 글자 한글 토큰이 절반 이상이면 음절이 쪼개진 줄로 판단."""
    toks = t.split()
    singles = sum(1 for x in toks if len(x) == 1 and "가" <= x <= "힣")
    return len(toks) >= 4 and singles / len(toks) >= 0.5


# ---------------------------------------------------------------- PaddleOCR

_paddle = None


def paddle_info() -> dict:
    if importlib.util.find_spec("paddleocr") is None:
        return {"available": False, "reason": "PaddleOCR 미설치 (install\\install_paddleocr.bat 실행)"}
    return {"available": True, "loaded": _paddle is not None}


def _get_paddle():
    global _paddle
    if _paddle is None:
        from paddleocr import PaddleOCR
        try:  # PaddleOCR 3.x
            _paddle = PaddleOCR(
                lang="korean",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:  # 2.x
            _paddle = PaddleOCR(lang="korean", use_angle_cls=False, show_log=False)
    return _paddle


def run_paddle(gray: np.ndarray) -> dict:
    ocr = _get_paddle()
    h, w = gray.shape[:2]
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    lines = []
    if hasattr(ocr, "predict"):
        for r in ocr.predict(img):
            data = r
            try:
                data = r.json.get("res", r.json)
            except Exception:  # noqa: BLE001
                pass
            texts = data.get("rec_texts", []) if isinstance(data, dict) else r["rec_texts"]
            scores = data.get("rec_scores", []) if isinstance(data, dict) else r["rec_scores"]
            polys = data.get("rec_polys", data.get("dt_polys", [])) if isinstance(data, dict) else r["rec_polys"]
            for t, s, p in zip(texts, scores, polys):
                lines.append(_poly_line(t, s, p, w, h))
    else:
        res = ocr.ocr(img, cls=False)
        for page in res or []:
            for p, (t, s) in page or []:
                lines.append(_poly_line(t, s, p, w, h))
    lines = _reading_order(lines)
    return _finish("paddle", lines)


def _poly_line(text, score, poly, w, h) -> dict:
    pts = np.array(poly, dtype=float).reshape(-1, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return {"text": str(text), "conf": round(float(score) * 100, 1),
            "box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h], "block": 0}


def _reading_order(lines: list[dict]) -> list[dict]:
    """위→아래, 같은 줄이면 왼→오른. 큰 세로 간격이면 문단 구분."""
    if not lines:
        return lines
    lines = sorted(lines, key=lambda l: (l["box"][1] + l["box"][3] / 2))
    rows: list[list[dict]] = []
    for ln in lines:
        cy = ln["box"][1] + ln["box"][3] / 2
        if rows:
            last = rows[-1]
            lcy = sum(x["box"][1] + x["box"][3] / 2 for x in last) / len(last)
            lh = sum(x["box"][3] for x in last) / len(last)
            if abs(cy - lcy) < lh * 0.5:
                last.append(ln)
                continue
        rows.append([ln])
    out, block, prev_bottom, prev_h = [], 0, None, None
    for row in rows:
        row.sort(key=lambda l: l["box"][0])
        top = min(l["box"][1] for l in row)
        hgt = sum(l["box"][3] for l in row) / len(row)
        if prev_bottom is not None and top - prev_bottom > (prev_h or hgt) * 1.2:
            block += 1
        merged = {
            "text": "  ".join(l["text"] for l in row) if len(row) > 1 else row[0]["text"],
            "conf": round(min(l["conf"] for l in row), 1),
            "box": [min(l["box"][0] for l in row), top,
                    max(l["box"][0] + l["box"][2] for l in row) - min(l["box"][0] for l in row),
                    max(l["box"][1] + l["box"][3] for l in row) - top],
            "block": block,
        }
        out.append(merged)
        prev_bottom, prev_h = merged["box"][1] + merged["box"][3], hgt
    return out


# ---------------------------------------------------------------- Ollama

VISION_HINTS = ("vl", "vision", "llava", "minicpm-v", "moondream", "gemma3", "llama4", "mistral-small3", "bakllava", "glm-4v", "ocr")


def ollama_info() -> dict:
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=1.5) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:  # noqa: BLE001
        return {"available": False, "reason": "Ollama가 실행 중이 아닙니다 (ollama.com 설치 후 실행)", "models": [], "vision": []}
    vision = [m for m in models if any(k in m.lower() for k in VISION_HINTS)]
    return {"available": bool(models), "models": models, "vision": vision,
            "reason": None if models else "설치된 모델이 없습니다 (예: ollama pull qwen2.5vl:7b)"}


def ollama_chat(model: str, prompt: str, image: np.ndarray | None = None, system: str | None = None,
                fmt: str | None = None, timeout: int = 600) -> str:
    msg: dict = {"role": "user", "content": prompt}
    if image is not None:
        h, w = image.shape[:2]
        s = min(1.0, 1800 / max(h, w))
        if s < 1:
            image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", image)
        msg["images"] = [base64.b64encode(buf.tobytes()).decode()]
    messages = ([{"role": "system", "content": system}] if system else []) + [msg]
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192}}
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["message"]["content"]


VLM_PROMPT = (
    "이 이미지는 스캔한 문서입니다. 이미지에 보이는 모든 글자를 위에서 아래, 왼쪽에서 오른쪽 순서로 "
    "빠짐없이 그대로 옮겨 적으세요.\n"
    "규칙: 원문에 없는 내용을 추가하거나 요약·설명하지 말 것. 줄바꿈은 원문 줄 단위를 따를 것. "
    "표는 칸을 ' | '로 구분할 것. 읽을 수 없는 글자는 □로 표시할 것. 결과 텍스트만 출력할 것."
)


def run_ollama_vlm(img: np.ndarray, model: str) -> dict:
    text = ollama_chat(model, VLM_PROMPT, image=img)
    text = text.strip().strip("`").strip()
    lines, block = [], 0
    for raw in text.splitlines():
        if not raw.strip():
            block += 1
            continue
        lines.append({"text": raw.rstrip(), "conf": None, "box": None, "block": block})
    return _finish(f"ollama:{model}", lines)


# ---------------------------------------------------------------- 공통

def _finish(engine: str, lines: list[dict]) -> dict:
    confs = [l["conf"] for l in lines if l.get("conf") is not None]
    return {
        "engine": engine,
        "lines": lines,
        "text": lines_to_text(lines),
        "mean_conf": round(sum(confs) / len(confs), 1) if confs else None,
    }


def lines_to_text(lines: list[dict]) -> str:
    out, prev = [], None
    for l in lines:
        if prev is not None and l.get("block") != prev:
            out.append("")
        out.append(l["text"])
        prev = l.get("block")
    return "\n".join(out)


def run_engine(engine: str, gray: np.ndarray, color: np.ndarray, lang: str, psm: int, model: str | None, fix_spacing: bool = True) -> dict:
    t = time.time()
    if engine == "tesseract":
        res = run_tesseract(gray, lang=lang, psm=psm, fix_spacing=fix_spacing)
    elif engine == "iux":
        res = run_iux(gray, color, lang)
    elif engine == "paddle":
        res = run_paddle(gray)
    elif engine == "ollama":
        if not model:
            raise ValueError("Ollama 비전 모델을 선택하세요.")
        res = run_ollama_vlm(color, model)
    else:
        raise ValueError(f"알 수 없는 엔진: {engine}")
    res["time_ms"] = int((time.time() - t) * 1000)
    return res


def run_iux(gray: np.ndarray, color: np.ndarray | None = None, lang: str = "kor+eng") -> dict:
    """IUX 엔진: 내장 딥러닝 모델(ONNX, CPU)로 글줄을 찾아 읽는다. 손글씨·흐린 글씨에 강함.
    원본 색상 이미지를 그대로 읽는 쪽이 정확해서(모눈·줄 공책 등) 원본을 우선 쓴다."""
    from . import iux_engine
    img = color if color is not None and color.size else gray
    # 색 도장 지우기·흐린 글씨 진하게는 늘 켬, 한자는 언어에 한자(chi_tra)를 고른 경우에만
    return _finish("iux", iux_engine.read_lines(img, {"hanja": "chi_tra" in (lang or "")}))


def iux_info() -> dict:
    try:
        from . import iux_engine
        return iux_engine.info()
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e)}


def engines_status() -> dict:
    return {
        "tesseract": tesseract_info(),
        "iux": iux_info(),
        "paddle": paddle_info(),
        "ollama": ollama_info(),
        "python": sys.version.split()[0],
    }
