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
REC_NAME = "iux_rec_ko.onnx"
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
def _rec(crops: list[np.ndarray], batch: int = 8) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(1, crops[i].shape[0]))
    s_ = _sess["rec"]
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
            txt = _respace(c, txt, xs)
            out[i] = ("".join(txt).strip(), float(np.mean(cs)) if cs else 0.0)
    return out


def _respace(crop: np.ndarray, chars: list[str], xs: list[float]) -> list[str]:
    """모델이 띄어쓰기를 자주 빠뜨려서, 글줄 이미지의 실제 빈칸(넓은 틈)을 보고 공백을 되살린다."""
    if len(chars) < 2:
        return chars
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = g.shape[:2]
    _, ink = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows = np.where(ink.sum(1) > 0)[0]
    th = (rows[-1] - rows[0] + 1) if len(rows) else h  # 실제 글자 높이
    col = ink.sum(0) > 0
    gaps, st = [], None
    for x_ in range(w):
        if not col[x_] and st is None:
            st = x_
        elif col[x_] and st is not None:
            gaps.append((st, x_))
            st = None
    first, last = (np.argmax(col), w - np.argmax(col[::-1])) if col.any() else (0, w)
    gaps = [(a, b) for a, b in gaps if a > first and b < last and (b - a) >= th * 0.32]
    if not gaps:
        return chars
    out = [chars[0]]
    for j in range(1, len(chars)):
        if chars[j] != " " and out[-1] != " ":
            x0, x1 = xs[j - 1] * w, xs[j] * w
            if any(x0 < (a + b) / 2 < x1 for a, b in gaps):
                out.append(" ")
        out.append(chars[j])
    return out


def read_lines(bgr: np.ndarray) -> list[dict]:
    """글줄 상자를 주면 그대로 읽는 대신, 이미지 전체에서 글줄을 찾아 읽는다."""
    _load()
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    boxes = _det(bgr)
    crops, keep = [], []
    for pts, sc in boxes:
        c = _crop(bgr, pts)
        if c.size and c.shape[0] >= 6 and c.shape[1] >= 6:
            crops.append(c)
            keep.append(pts)
    res = _rec(crops) if crops else []
    lines = []
    for pts, (t, cf) in zip(keep, res):
        if not t:
            continue
        x1, y1, x2, y2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        lines.append({"text": t, "conf": round(cf * 100, 1),
                      "box": [round(float(x1 / w), 5), round(float(y1 / h), 5), round(float((x2 - x1) / w), 5), round(float((y2 - y1) / h), 5)]})
    return lines


def read_crop(bgr: np.ndarray) -> tuple[str, float]:
    """한 줄짜리 이미지를 바로 읽는다."""
    _load()
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    t, c = _rec([bgr])[0]
    return t, c * 100
