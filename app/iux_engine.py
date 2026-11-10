"""IUX 엔진 — 인터넷·GPU 없이 CPU로 도는 딥러닝 OCR (ONNX Runtime).

글줄 찾기(PP-OCRv5 det) + 한글 읽기(korean PP-OCRv5 rec, 한글 11,172자 전부 지원) 모델을
ONNX로 직접 돌린다. paddle 설치가 필요 없고 설치 파일에 모델을 넣어 완전 오프라인으로 동작한다.
모델: PaddleOCR (Apache-2.0).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

DET_NAME = "iux_det.onnx"
REC_NAME = "iux_rec_ko.onnx"   # 인쇄체 위주 (PP-OCRv5 한국어)
HW_NAME = "iux_rec_hw.onnx"    # 손글씨 추가 학습 모델 (korean-handwriting-rec, AI Hub 손글씨 386만 단어)
DICT_NAME = "iux_rec_ko_dict.txt"

_lock = threading.Lock()
_sess: dict = {}
_chars: list[str] | None = None


def model_dir() -> Path | None:
    cands = []
    if os.environ.get("IUX_MODELS"):
        cands.append(Path(os.environ["IUX_MODELS"]))
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).parent
        cands += [exe / "models", exe / "_internal" / "models"]
        if hasattr(sys, "_MEIPASS"):
            cands.append(Path(sys._MEIPASS) / "models")
    cands.append(Path(__file__).resolve().parent.parent / "models")
    for c in cands:
        if (c / DET_NAME).exists() and (c / REC_NAME).exists() and (c / DICT_NAME).exists():
            return c
    return None


def info() -> dict:
    try:
        import onnxruntime as ort  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"onnxruntime 없음: {e}"}
    d = model_dir()
    if not d:
        return {"available": False, "reason": "IUX 엔진 모델 파일(models 폴더)이 없습니다"}
    return {"available": True, "dir": str(d)}


def _load():
    global _chars
    with _lock:
        if _sess:
            return
        import onnxruntime as ort
        d = model_dir()
        if not d:
            raise RuntimeError("IUX 엔진 모델 파일이 없습니다")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 2)))
        so.log_severity_level = 3
        # 한글 경로에서도 열리도록 파일 내용을 읽어서 넘긴다
        _sess["det"] = ort.InferenceSession((d / DET_NAME).read_bytes(), so, providers=["CPUExecutionProvider"])
        _sess["rec"] = ort.InferenceSession((d / REC_NAME).read_bytes(), so, providers=["CPUExecutionProvider"])
        if (d / HW_NAME).exists():
            _sess["hw"] = ort.InferenceSession((d / HW_NAME).read_bytes(), so, providers=["CPUExecutionProvider"])
        chars = (d / DICT_NAME).read_text(encoding="utf-8").split("\n")
        if chars and chars[-1] == "":
            chars.pop()
        _chars = ["", *chars, " "]  # 0 = CTC blank, 끝 = 공백


# ---------------- 글줄 찾기 (DB) ----------------
def _det(bgr: np.ndarray, side: int = 1536, thresh: float = 0.3, box_thresh: float = 0.5, unclip: float = 1.7):
    h, w = bgr.shape[:2]
    s = min(1.0, side / max(h, w))
    if max(h, w) < 960:
        s = min(2.0, 960 / max(h, w))  # 작은 이미지는 키움
    nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
    img = cv2.resize(bgr, (nw, nh))
    x = (img.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)[None]
    s_ = _sess["det"]
    prob = s_.run(None, {s_.get_inputs()[0].name: x})[0][0, 0]
    mask = (prob > thresh).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    sx, sy = w / nw, h / nh
    for c in cnts[:2000]:
        if cv2.contourArea(c) < 12:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < 3:
            continue
        m = np.zeros_like(mask)
        cv2.fillPoly(m, [c], 1)
        score = float(cv2.mean(prob, m)[0])
        if score < box_thresh:
            continue
        d = rw * rh * unclip / (2 * (rw + rh) + 1e-6)  # 줄어든 영역을 원래 글자 크기로 되돌림
        pts = cv2.boxPoints(((cx, cy), (rw + 2 * d, rh + 2 * d), ang))
        pts[:, 0] = np.clip(pts[:, 0] * sx, 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1] * sy, 0, h - 1)
        boxes.append((pts.astype(np.float32), score))
    return boxes


def _order_pts(p: np.ndarray) -> np.ndarray:
    p = p[np.argsort(p[:, 0])]
    l, r = p[:2], p[2:]
    l = l[np.argsort(l[:, 1])]
    r = r[np.argsort(r[:, 1])]
    return np.array([l[0], r[0], r[1], l[1]], np.float32)


def _crop(bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = _order_pts(pts)
    wd = int(max(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[3] - p[2])))
    ht = int(max(np.linalg.norm(p[0] - p[3]), np.linalg.norm(p[1] - p[2])))
    if wd < 2 or ht < 2:
        return np.zeros((0, 0, 3), np.uint8)
    M = cv2.getPerspectiveTransform(p, np.array([[0, 0], [wd, 0], [wd, ht], [0, ht]], np.float32))
    out = cv2.warpPerspective(bgr, M, (wd, ht), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if ht >= wd * 1.5:  # 세로로 선 글줄
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out


# ---------------- 글자 읽기 (CTC) ----------------
def _rec(crops: list[np.ndarray], batch: int = 8, which: str = "rec") -> list[tuple]:
    """(글자, 평균 확신도, 글자별 확신도 목록) 을 돌려준다."""
    out: list[tuple] = [("", 0.0, [])] * len(crops)
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(1, crops[i].shape[0]))
    s_ = _sess[which]
    name = s_.get_inputs()[0].name
    H = 48
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        ratio = max(crops[i].shape[1] / max(1, crops[i].shape[0]) for i in idx)
        W = int(min(3200, max(160, np.ceil(H * ratio / 32) * 32)))
        x = np.zeros((len(idx), 3, H, W), np.float32)
        for k, i in enumerate(idx):
            c = crops[i]
            w = min(W, max(8, int(round(H * c.shape[1] / max(1, c.shape[0])))))
            r = cv2.resize(c, (w, H), interpolation=cv2.INTER_CUBIC if c.shape[0] < H else cv2.INTER_AREA)
            x[k, :, :, :w] = ((r.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
        prob = s_.run(None, {name: x})[0]  # (N, T, C)
        ids, ps = prob.argmax(2), prob.max(2)
        T = prob.shape[1]
        for k, i in enumerate(idx):
            c = crops[i]
            w = min(W, max(8, int(round(H * c.shape[1] / max(1, c.shape[0])))))
            txt, cs, xs, prev = [], [], [], 0
            for t, (ci, p) in enumerate(zip(ids[k], ps[k])):
                if ci != 0 and ci != prev and ci < len(_chars):
                    txt.append(_chars[ci])
                    cs.append(float(p))
                    xs.append((t + 0.5) / T * W / w)  # 글줄 안에서의 가로 위치(0~1)
                prev = ci
            txt, pr = _respace(c, txt, xs, cs)
            while txt and txt[0] == " ":
                txt, pr = txt[1:], pr[1:]
            while txt and txt[-1] == " ":
                txt, pr = txt[:-1], pr[:-1]
            out[i] = ("".join(txt), float(np.mean(cs)) if cs else 0.0, pr)
    return out


def _respace(crop: np.ndarray, chars: list[str], xs: list[float], probs: list[float]) -> tuple[list[str], list[float]]:
    """모델이 띄어쓰기를 자주 빠뜨려서, 글줄 이미지의 실제 빈칸(넓은 틈)을 보고 공백을 되살린다."""
    if len(chars) < 2:
        return chars, probs
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = g.shape[:2]
    _, ink = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink[ink.sum(1) > w * 0.6] = 0  # 공책 줄·밑줄처럼 가로로 길게 이어진 선은 글자가 아님
    rows = np.where(ink.sum(1) > 0)[0]
    th = (rows[-1] - rows[0] + 1) if len(rows) else h  # 실제 글자 높이
    col = ink.sum(0) > max(1, th * 0.06)  # 모눈·얼룩 같은 옅은 잡티는 빈칸으로 봄
    gaps, st = [], None
    for x_ in range(w):
        if not col[x_] and st is None:
            st = x_
        elif col[x_] and st is not None:
            gaps.append((st, x_))
            st = None
    first, last = (np.argmax(col), w - np.argmax(col[::-1])) if col.any() else (0, w)
    inner = [(a, b) for a, b in gaps if a > first and b < last]
    lim = th * 0.28
    # 글자가 하나하나 떨어진 글씨(인쇄체·또박또박 쓴 손글씨)는 글자 사이에도 틈이 있다.
    # 틈 너비를 크기순으로 놓고 '글자 사이 틈'과 '낱말 사이 틈'이 갈리는 지점(가장 크게 벌어지는 곳)을 찾아 기준으로 삼는다.
    n_ch = sum(1 for c in chars if c != " ")
    if len(inner) >= max(3, n_ch * 0.5):
        ws = sorted(b - a for a, b in inner)
        best, cut = 1.0, None
        for k in range(max(1, len(ws) // 3), len(ws)):
            if ws[k] >= th * 0.18 and ws[k - 1] > 0 and ws[k] / ws[k - 1] > best:
                best, cut = ws[k] / ws[k - 1], (ws[k] + ws[k - 1]) / 2
        lim = cut if cut is not None and best >= 1.45 else max(lim, ws[len(ws) // 2] * 1.7)
    gaps = [(a, b) for a, b in inner if (b - a) >= lim]
    if not gaps:
        return chars, probs
    out, pr = [chars[0]], [probs[0]]
    for j in range(1, len(chars)):
        if chars[j] != " " and out[-1] != " ":
            x0, x1 = xs[j - 1] * w, xs[j] * w
            if any(x0 < (a + b) / 2 < x1 for a, b in gaps):
                out.append(" ")
                pr.append(1.0)
        out.append(chars[j])
        pr.append(probs[j])
    return out, pr


MODE = os.environ.get("IUX_REC_MODE", "best")  # best: 두 모델 중 더 확신하는 쪽 / hw / rec


def _word_spans(crop: np.ndarray) -> list[tuple[int, int]]:
    """글줄 이미지를 낱말 사이 빈칸에서 나눈 가로 구간들. 빈칸을 못 찾으면 통째로 하나."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = g.shape[:2]
    _, ink = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink[ink.sum(1) > w * 0.6] = 0
    rows = np.where(ink.sum(1) > 0)[0]
    th = (rows[-1] - rows[0] + 1) if len(rows) else h
    col = ink.sum(0) > max(1, th * 0.06)
    if not col.any():
        return [(0, w)]
    first, last = int(np.argmax(col)), int(w - np.argmax(col[::-1]))
    gaps, st = [], None
    for x_ in range(first, last):
        if not col[x_] and st is None:
            st = x_
        elif col[x_] and st is not None:
            gaps.append((st, x_))
            st = None
    if not gaps:
        return [(0, w)]
    ws = sorted(b - a for a, b in gaps)
    lim = th * 0.35
    if len(ws) >= 3:  # 글자 사이 틈과 낱말 사이 틈이 갈리는 지점
        best, cut = 1.0, None
        for k in range(max(1, len(ws) // 3), len(ws)):
            if ws[k] >= th * 0.2 and ws[k - 1] > 0 and ws[k] / ws[k - 1] > best:
                best, cut = ws[k] / ws[k - 1], (ws[k] + ws[k - 1]) / 2
        if cut is not None and best >= 1.45:
            lim = max(th * 0.2, cut)
    cuts = [(a + b) // 2 for a, b in gaps if b - a >= lim]
    if not cuts:
        return [(0, w)]
    pad = max(2, int(th * 0.1))
    edges = [first] + cuts + [last]
    return [(max(0, edges[i] - pad), min(w, edges[i + 1] + pad)) for i in range(len(edges) - 1) if edges[i + 1] - edges[i] > 2]


def _pick(x: tuple, y: tuple, mar: float = 0.1) -> tuple:
    """인쇄체 결과 x와 손글씨 결과 y 중 하나. 손글씨 모델은 인쇄체에서도 확신이 높게 나오는 편이라
    뚜렷이(mar 이상) 더 확신할 때만 손글씨 결과를 쓴다."""
    return x if (x[0] and x[1] + mar >= y[1]) or not y[0] else y


def _rec_best(crops: list[np.ndarray]) -> list[tuple[str, float]]:
    """줄마다 인쇄체·손글씨 두 모델로 읽어 더 확신하는 결과를 고른다.
    긴 줄은 낱말 단위로도 잘라 읽고(모델이 짧은 낱말 이미지로 학습돼 긴 줄에서 무너지는 경우가 있음),
    낱말마다 두 모델 중 나은 쪽을 골라 이어 붙인 결과가 더 확신이 높으면 그쪽을 쓴다."""
    if "hw" not in _sess or MODE == "rec":
        return _rec(crops)
    if MODE == "hw":
        return _rec(crops, which="hw")
    mar = float(os.environ.get("IUX_MAR", "0.1"))
    line = [_pick(x, y, mar) for x, y in zip(_rec(crops), _rec(crops, which="hw"))]
    if os.environ.get("IUX_WORDS", "1") != "1":
        return line
    pieces, owner = [], []
    for i, c in enumerate(crops):
        if c.shape[1] <= c.shape[0] * 7 or line[i][1] >= float(os.environ.get("IUX_WLOW", "0.85")):
            continue  # 짧은 줄이나 이미 확신이 높은 줄은 그대로
        spans = _word_spans(c)
        if len(spans) < 2:
            continue
        for a, b in spans:
            pieces.append(c[:, a:b])
            owner.append(i)
    if not pieces:
        return line
    wa, wb = _rec(pieces), _rec(pieces, which="hw")
    words: dict[int, list] = {}
    for i, x, y in zip(owner, wa, wb):
        r = _pick(x, y, mar)
        if r[0]:
            words.setdefault(i, []).append(r)
    out = list(line)
    for i, ws in words.items():
        n = sum(len(r[0]) for r in ws)
        cf = sum(r[1] * len(r[0]) for r in ws) / n if n else 0.0
        if cf > line[i][1] + 0.02:
            pr: list[float] = []
            for k, r in enumerate(ws):
                pr += ([1.0] if k else []) + list(r[2])
            out[i] = (" ".join(r[0] for r in ws), cf, pr)
    return out


import re as _re

_MEAN = _re.compile(r"[가-힣A-Za-z0-9\u3400-\u9fff]")
_HAN = _re.compile(r"[가-힣]")


def _rect(pts: np.ndarray):
    """글줄 상자의 중심·긴 변·짧은 변·기울기(도, 긴 변 기준 -90~90)."""
    p = _order_pts(pts)
    wv, hv = p[1] - p[0], p[3] - p[0]
    wl, hl = float(np.linalg.norm(wv)), float(np.linalg.norm(hv))
    c = p.mean(0)
    if hl > wl * 1.5:  # 세로로 선 상자
        return c, hl, wl, None
    return c, wl, hl, float(np.degrees(np.arctan2(wv[1], wv[0])))


def decolor(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """빨강·주황·노랑(직인·무궁화 바탕·형광펜)을 밝기로 바꿔 흰색에 가깝게 한다.
    그 위에 겹친 검은 글씨는 밝기가 낮아 그대로 남는다. (결과, 따뜻한 색 비율)"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    warm = ((h_ <= 35) | (h_ >= 150)) & (s_ >= 60)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = np.where(warm, v_, g).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), float(warm.mean())


def enhance_faint(bgr: np.ndarray) -> np.ndarray | None:
    """흐린 글씨(옅은 타자기·복사본)면 대비를 늘려 진하게 한 이미지를, 아니면 None."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) if max(g.shape) > 1200 else g
    p1, p50 = np.percentile(small, [0.5, 50])
    if p1 < 110 or p50 - p1 < 25:
        return None
    x = np.clip((g.astype(np.float32) - p1) / max(1.0, (p50 - p1)) * 255.0, 0, 255)
    x = (255.0 * (x / 255.0) ** 1.6).astype(np.uint8)  # 중간 회색을 더 어둡게
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


CHECK_P = float(os.environ.get("IUX_CHECK_P", "0.9"))

_HANJA = _re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _tess_hanja(img: np.ndarray, psms: tuple = (7,)) -> tuple[str, float]:
    """한자 데이터(chi_tra)로 한 덩어리를 읽는다. (글자, 평균 확신도 0~100). 실패하면 ("", 0)."""
    from . import engines
    try:
        if engines._tess_cmd is None:
            engines.tesseract_info()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        s = max(1.0, 64 / max(1, g.shape[0]))
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        g = cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        best = ("", 0.0)
        for psm in psms:
            _, tsv = engines._tess_txt_tsv(g, "chi_tra", psm)
            d = engines._parse_tsv(tsv)
            ws = [(t.strip(), c) for t, c in zip(d.get("text", []), d.get("conf", [])) if t and t.strip() and isinstance(c, float) and c >= 0]
            if ws:
                c = sum(c for _, c in ws) / len(ws)
                if c > best[1]:
                    best = ("".join(t for t, _ in ws), c)
        return best
    except Exception:
        return ("", 0.0)


def _is_hanja(t: str, cf: float) -> bool:
    core = _re.sub(r"\s+", "", t)
    if not core or len(_HANJA.findall(core)) != len(core):
        return False
    return (len(core) >= 2 and cf >= 70) or cf >= 88


def has_hanja_data() -> bool:
    from . import engines
    try:
        info = engines.tesseract_info()
        return "chi_tra" in (info.get("langs") or [])
    except Exception:
        return False


def _hanja_line(crop: np.ndarray, cur: tuple) -> tuple | None:
    """한 글줄에서 IUX가 못 읽는(확신 낮은) 낱말을 한자로 다시 읽어 본다. 한자를 찾았으면 새 결과, 아니면 None."""
    if cur[0] and cur[1] >= 0.985:
        return None
    if crop.shape[1] <= crop.shape[0] * 1.6:  # 한두 글자짜리 덩어리
        if cur[1] >= 0.6 and cur[0]:
            return None
        t, cf = _tess_hanja(crop, (7, 8, 10))
        return (t, cf / 100, [1.0] * len(t)) if _is_hanja(t, cf) else None
    spans = _word_spans(crop)
    if len(spans) < 2:
        if cur[1] >= 0.6 and cur[0]:
            return None
        t, cf = _tess_hanja(crop, (7, 8))
        return (t, cf / 100, [1.0] * len(t)) if _is_hanja(t, cf) else None
    # 줄 전체가 한자일 수도 있다 (한자 위주 문서): IUX가 거의 못 읽은 줄은 통째로 먼저 시도
    if not cur[0] or cur[1] < 0.6:
        t, cf = _tess_hanja(crop, (7,))
        if _is_hanja(t, cf):
            return (t, cf / 100, [1.0] * len(t))
    pieces = [crop[:, a:b] for a, b in spans]
    rr = _rec_best(pieces)
    low = [i for i, r in enumerate(rr) if (not r[0] or r[1] < 0.7) and not _re.search(r"[0-9*#@%()\-]", r[0] or "")]
    if not low:
        return None
    hz: dict[int, tuple[int, str, float]] = {}  # 시작 조각 → (끝 조각, 한자, 확신도)
    used: set[int] = set()
    for i in low:
        if i in used:
            continue
        t, cf = _tess_hanja(pieces[i])
        core = _re.sub(r"\s+", "", t)
        if not core or len(_HANJA.findall(core)) != len(core) or cf < 55:
            continue  # 한자 후보가 아님
        best = (i, i, t, cf)
        # 이웃 조각(낱글자로 떨어진 한자 이름 등)으로 넓혀 가며 여전히 한자로 읽히면 묶는다
        sure_ko = lambda k: rr[k][1] >= 0.97 and len(rr[k][0]) >= 2 and not _re.search(r"[0-9]", rr[k][0])
        for step in (-1, 1):
            while True:
                a, b = best[0], best[1]
                k = a - 1 if step < 0 else b + 1
                if k < 0 or k >= len(spans) or k in used or sure_ko(k):
                    break
                a2, b2 = (k, b) if step < 0 else (a, k)
                t2, cf2 = _tess_hanja(crop[:, spans[a2][0]:spans[b2][1]])
                c2 = _re.sub(r"\s+", "", t2)
                if not (c2 and len(_HANJA.findall(c2)) == len(c2) and cf2 >= 55 and len(c2) > len(_re.sub(r"\s+", "", best[2]))):
                    break
                best = (a2, b2, t2, cf2)
        # 최종 판정: 두 글자 이상이면 확신 70 이상, 한 글자면 IUX가 거의 못 읽은 경우에만
        if best and not (_is_hanja(best[2], best[3]) and (len(_re.sub(r"\s+", "", best[2])) >= 2 or rr[i][1] < 0.4)):
            best = None
        if best:
            a, b, t, cf = best
            hz[a] = (b, t, cf)
            used.update(range(a, b + 1))
    if not hz:
        return None
    # 한자가 아닌 구간은 이어 붙여 한 번에 다시 읽어 줄의 품질을 유지
    segs, k = [], 0
    while k < len(spans):
        if k in hz:
            b, t, cf = hz[k]
            segs.append(("hz", t, cf / 100))
            k = b + 1
        else:
            j = k
            while j + 1 < len(spans) and (j + 1) not in hz:
                j += 1
            segs.append(("ko", spans[k][0], spans[j][1]))
            k = j + 1
    ko = [crop[:, x[1]:x[2]] for x in segs if x[0] == "ko"]
    kr = iter(_rec_best(ko) if ko else [])
    parts, pr, cfs = [], [], []
    for x in segs:
        r = (x[1], x[2], [1.0] * len(x[1])) if x[0] == "hz" else next(kr)
        if not r[0]:
            continue
        if parts:
            pr.append(1.0)
        parts.append(r[0])
        pr += list(r[2])
        cfs.append((r[1], len(r[0])))
    n = sum(k for _, k in cfs)
    return (" ".join(parts), sum(c * k for c, k in cfs) / n if n else 0.0, pr)


def read_lines(bgr: np.ndarray, opts: dict | None = None) -> list[dict]:
    """이미지 전체에서 글줄을 찾아 읽고, 읽는 순서대로 정리해 돌려준다.
    비스듬히 찍은 사진(기울어진 글줄)과 세로쓰기(오른쪽 줄부터)도 순서를 맞춘다.
    opts: decolor(색 도장 지우기, 기본 켬), faint(흐린 글씨 진하게, 기본 켬)"""
    _load()
    o = {"decolor": True, "faint": True, "hanja": False, **(opts or {})}
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    srcs = [bgr]
    if o["faint"]:
        f = enhance_faint(bgr)
        if f is not None:
            srcs = [f, bgr]
    if o["decolor"]:
        d, ratio = decolor(srcs[0])
        if ratio > 0.003:
            srcs = [d] + srcs  # 도장 지운 이미지로 글줄을 찾고, 줄마다 원본과 비교해 나은 쪽을 쓴다
    boxes = _det(srcs[0])
    keep, per = [], [[] for _ in srcs]
    for pts, sc in boxes:
        cs_ = [_crop(s_, pts) for s_ in srcs]
        c = cs_[0]
        if c.size and c.shape[0] >= 6 and c.shape[1] >= 6:
            keep.append(pts)
            for k, cc in enumerate(cs_):
                per[k].append(cc)
    res = _rec_best(per[0]) if keep else []
    for k in range(1, len(srcs)):
        if not keep:
            break
        alt = _rec_best(per[k])
        res = [b if (b[0] and b[1] > a[1] + 0.01) or not a[0] else a for a, b in zip(res, alt)]
    if o["hanja"] and keep and has_hanja_data():
        res = [(_hanja_line(c, r) or r) for c, r in zip(per[0], res)]
    items = []
    for pts, (t, cf, pr) in zip(keep, res):
        if not t or not _MEAN.search(t):
            continue  # 도장·얼룩·기호만 읽힌 것
        core = _re.sub(r"\s+", "", t)
        if (len(core) <= 2 and cf < 0.6) or (not _HAN.search(core) and not _HANJA.search(core) and cf < 0.55):
            continue  # 짧거나 한글 없는 저신뢰 조각
        c, L, S, ang = _rect(pts)
        chk = [k for k, (ch, p) in enumerate(zip(t, pr)) if ch.isdigit() and p < CHECK_P]
        items.append({"pts": pts, "text": t, "conf": cf, "c": c, "L": L, "S": S, "ang": ang, "chk": chk})
    if not items:
        return []

    def out(group: list[dict], block: int) -> dict:
        P = np.concatenate([g["pts"] for g in group])
        x1, y1, x2, y2 = P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
        chk, off = [], 0
        for g in group:  # 확인이 필요한 숫자 위치 (합친 줄 기준)
            chk += [off + k for k in g["chk"]]
            off += len(g["text"]) + 2
        d = {"text": "  ".join(g["text"] for g in group), "conf": round(min(g["conf"] for g in group) * 100, 1), "block": block,
             "box": [round(float(x1 / w), 5), round(float(y1 / h), 5), round(float((x2 - x1) / w), 5), round(float((y2 - y1) / h), 5)]}
        if chk:
            d["check"] = chk
        return d

    vert = [it for it in items if it["ang"] is None]
    if len(vert) >= max(2, len(items) * 0.6):  # 세로쓰기: 오른쪽 줄부터, 줄마다 한 줄
        items.sort(key=lambda it: -it["c"][0])
        return [out([it], 0) for it in items]

    angs = [it["ang"] for it in items if it["ang"] is not None and it["L"] > it["S"] * 2]
    A = float(np.median(angs)) if angs else 0.0
    if abs(A) < 1.5:
        A = 0.0
    ca, sa = np.cos(np.radians(-A)), np.sin(np.radians(-A))
    for it in items:  # 글줄 방향에 맞춰 돌린 좌표에서 줄을 묶는다
        it["u"] = it["c"][0] * ca - it["c"][1] * sa
        it["v"] = it["c"][0] * sa + it["c"][1] * ca
    items.sort(key=lambda it: it["v"])
    rows: list[list[dict]] = []
    for it in items:
        if rows:
            last = rows[-1]
            lv = sum(x["v"] for x in last) / len(last)
            lh = sum(x["S"] for x in last) / len(last)
            if abs(it["v"] - lv) < min(lh, it["S"]) * 0.55:
                last.append(it)
                continue
        rows.append([it])
    lines, block, prev = [], 0, None
    for row in rows:
        row.sort(key=lambda it: it["u"])
        v = sum(x["v"] for x in row) / len(row)
        hh = sum(x["S"] for x in row) / len(row)
        if prev is not None and v - prev[0] > (prev[1] + hh) * 1.25:
            block += 1
        prev = (v, hh)
        lines.append(out(row, block))
    return lines


def read_crop(bgr: np.ndarray) -> tuple[str, float]:
    """한 줄짜리 이미지를 바로 읽는다."""
    _load()
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    r = _rec_best([bgr])[0]
    return r[0], r[1] * 100
