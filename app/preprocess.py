"""이미지 전처리 — 일반 PC(CPU)용. OpenCV 기반."""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_OPTS = {
    "deskew": True,      # 기울기 자동 보정
    "orient": True,      # 90/180/270도 돌아간 페이지 자동 감지 (Tesseract OSD)
    "contrast": True,    # 대비 향상 (CLAHE)
    "denoise": False,    # 노이즈 제거 (느림)
    "binarize": False,   # 흑백 이진화
    "upscale": True,     # 저해상도 확대
}


def load_bgr(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)  # 한글 경로 대응
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 열 수 없습니다: {path}")
    return img


def save_png(path: str, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("이미지 저장 실패")
    buf.tofile(path)


def _rotate(img: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    val = border if img.ndim == 2 else (border, border, border)
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=val)


def estimate_skew(gray: np.ndarray) -> float:
    """투영 프로파일 방식으로 기울기(도) 추정. 범위 ±10도."""
    h, w = gray.shape
    scale = 900 / max(w, 1)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if th.mean() < 1:  # 거의 빈 페이지
        return 0.0

    def score(a: float) -> float:
        r = _rotate(th, a, border=0)
        prof = r.sum(axis=1, dtype=np.float64)
        return float(np.var(prof))

    best, best_s = 0.0, score(0.0)
    for a in np.arange(-10, 10.01, 0.5):
        s = score(float(a))
        if s > best_s:
            best, best_s = float(a), s
    for a in np.arange(best - 0.5, best + 0.51, 0.1):
        s = score(float(a))
        if s > best_s:
            best, best_s = float(a), s
    return round(best, 2)


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _text_score(gray: np.ndarray) -> tuple[int, int] | None:
    """빠른 인식으로 (또렷하게 읽힌 글자 수, 전체 글자 수)를 센다 — 방향 판단용."""
    import re
    from . import engines
    try:
        if engines._tess_cmd is None:
            engines.tesseract_info()
        h, w = gray.shape[:2]
        s = min(1.0, 1400 / max(h, w))
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        _, tsv = engines._tess_txt_tsv(small, "kor+eng", 6)  # psm 3은 옆으로 누운 글줄도 알아서 읽어 버려 판단이 흐려짐
        d = engines._parse_tsv(tsv)
    except Exception:
        return None
    good = tot = 0
    for t, c in zip(d.get("text", []), d.get("conf", [])):
        n = len(re.sub(r"[^가-힣A-Za-z0-9]", "", t or ""))
        if not n or not isinstance(c, float) or c < 0:
            continue
        tot += n
        if c >= 65:
            good += n
    return good, tot


def _osd(gray: np.ndarray) -> tuple[int, float]:
    try:
        import pytesseract
        rot, conf = 0, 0.0
        for line in pytesseract.image_to_osd(gray, config="--psm 0").splitlines():
            if line.startswith("Rotate:"):
                rot = int(line.split(":")[1].strip())
            elif line.startswith("Orientation confidence:"):
                conf = float(line.split(":")[1].strip())
        return rot, conf
    except Exception:
        return 0, 0.0


def detect_orientation(gray: np.ndarray) -> int:
    """돌아간 페이지(90/180/270) 감지.
    Tesseract OSD는 표·도장 많은 한글 공문서에서 자주 틀리므로 힌트로만 쓰고,
    실제로 읽어 보아 또렷하게 읽히지 않으면 다른 방향들로 돌려 읽어 가장 잘 읽히는 방향을 고른다."""
    rot, conf = _osd(gray)
    if rot == 0 and conf >= 2.5:
        return 0
    base = _text_score(gray)
    if base is None:
        return rot if rot in _ROT and conf >= 6 else 0
    bg, bt = base
    if bg >= 10 and bg >= bt * 0.6:  # 지금 방향으로 잘 읽힘
        return 0
    cands = ([rot] if rot in _ROT else []) + [r for r in (90, 270, 180) if r != rot]
    best, best_g = 0, bg
    for r in cands:
        sc = _text_score(cv2.rotate(gray, _ROT[r]))
        if not sc:
            continue
        g, t = sc
        if g >= max(15, bg * 1.5) and g >= t * 0.5 and g > best_g:
            best, best_g = r, g
            if g >= t * 0.75:  # 확실하면 나머지 방향은 건너뜀
                break
    return best


def preprocess(img: np.ndarray, opts: dict | None = None) -> tuple[np.ndarray, dict]:
    o = {**DEFAULT_OPTS, **(opts or {})}
    info: dict = {"skew": 0.0, "rotate": 0, "scale": 1.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    if o.get("orient"):
        rot = detect_orientation(gray)
        if rot:
            code = _ROT.get(rot)
            if code is not None:
                gray = cv2.rotate(gray, code)
                info["rotate"] = rot

    if o.get("upscale"):
        h, w = gray.shape
        if w < 1600:
            s = min(2.5, 2000 / w)
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            info["scale"] = round(s, 3)

    if o.get("deskew"):
        ang = estimate_skew(gray)
        if abs(ang) >= 0.15:
            gray = _rotate(gray, ang)
            info["skew"] = ang

    if o.get("denoise"):
        gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)

    if o.get("contrast"):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if o.get("binarize"):
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)

    return gray, info
