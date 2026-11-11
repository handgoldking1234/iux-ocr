"""표 칸(셀) 찾기와 칸별 글자 정리.

1) 가로선·세로선만 뽑아 선 그림을 만든다 (가로/세로로 길게 이어진 검은 점만 남김).
2) 선으로 둘러싸인 빈 영역 하나하나가 칸이다. 칸 경계를 모아 행·열 번호와 병합(rowspan/colspan)을 정한다.
3) 맞닿은 칸끼리 묶어 표 하나로 만든다.
웹 버전(web/iux.js findTables)도 같은 규칙을 쓴다.
"""
from __future__ import annotations

import cv2
import numpy as np


def _line_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    _, b1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blk = max(15, (min(w, h) // 25) | 1)
    b2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk, 8)  # 흐린 선·고르지 않은 조명
    if (b1 > 0).mean() > 0.25:
        b1[:] = 0  # 그림자 진 사진: 전체 문턱값은 버림
    b = cv2.bitwise_or(b1, b2)
    kh = max(20, w // 50)   # 가로선: 이 길이 이상 이어진 검은 점 (글자 가로획보다 길게)
    kv = int(max(20, min(h, w * 1.6) // 40))   # 세로선: 글자 세로획보다 길게
    hor = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)))
    ver = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)))
    # 끊긴 선 잇기 (같은 방향 선끼리만)
    hor = cv2.morphologyEx(hor, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, kh // 3), 1)))
    ver = cv2.morphologyEx(ver, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, kv // 3))))
    m = cv2.bitwise_or(hor, ver)
    d = _dil(w, h)  # 모서리 이어 붙이기
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (d * 2 + 1, d * 2 + 1)))


def _dil(w: int, h: int) -> int:
    return max(2, round(min(w, h) / 400))


def _cluster(vals: list[float], tol: float) -> list[float]:
    vals = sorted(vals)
    groups: list[list[float]] = []
    for v in vals:
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def find_tables(img: np.ndarray, scale_to: int = 2200) -> list[dict]:
    """표 목록. 각 표: box(px), rows, cols, cells[{r,c,rs,cs,box(px)}]"""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    s = min(scale_to / max(H, W), 2400 / W, 3.0)  # 작은 이미지는 키우고 큰 이미지는 줄여 같은 크기 기준으로 본다
    g = cv2.resize(gray, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC) if abs(s - 1) > 0.05 else gray
    if g is gray:
        s = 1.0
    h, w = g.shape
    m = _line_mask(g)
    n, lab, st, _ = cv2.connectedComponentsWithStats((m == 0).astype(np.uint8), connectivity=4)
    minw, minh = max(10, w * 0.008), max(10, h * 0.01)
    cand = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if bw < minw or bh < minh or x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue  # 너무 작거나 종이 바깥(가장자리에 닿음)
        if bw > w * 0.97 and bh > h * 0.97:
            continue
        sub = lab[y:y + bh, x:x + bw] == i
        rows = np.where(sub.any(1))[0]
        filled = 0
        for r in rows:  # 줄마다 왼쪽~오른쪽 끝까지 채운 넓이 (칸 안의 도장·가림 상자는 구멍으로 보지 않음)
            xs = np.flatnonzero(sub[r])
            filled += xs[-1] - xs[0] + 1
        if filled < bw * bh * 0.8 or area < bw * bh * 0.35:
            continue  # 네모가 아닌 영역
        cand.append([x, y, x + bw, y + bh])
    # 다른 칸을 둘 이상 품은 큰 영역(표 바깥 테두리 사이 공간 등)은 칸이 아님
    ins = lambda b, a: b is not a and b[0] >= a[0] - 2 and b[1] >= a[1] - 2 and b[2] <= a[2] + 2 and b[3] <= a[3] + 2
    big = [a for a in cand if sum(1 for b in cand if ins(b, a)) < 2]
    # 칸 안에 또 잡힌 작은 네모(가림 상자·도장 테두리 안쪽)는 버림
    cells = [a for a in big if not any(ins(a, b) for b in big)]
    if len(cells) < 4:
        return []
    # 맞닿은 칸끼리 묶기
    gap = _dil(w, h) * 2 + 2 + max(4, round(min(w, h) / 150))  # 선 두께만큼 떨어진 칸은 이웃
    par = list(range(len(cells)))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i, a in enumerate(cells):
        for j in range(i + 1, len(cells)):
            b = cells[j]
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            if (ox > 0 and -gap <= oy <= gap * 0.5) or (oy > 0 and -gap <= ox <= gap * 0.5) or (ox > 0 and oy > 0):
                par[f(i)] = f(j)
    groups: dict[int, list] = {}
    for i, c in enumerate(cells):
        groups.setdefault(f(i), []).append(c)
    tables = []
    for cs in groups.values():
        if len(cs) < 4:
            continue
        tol = _dil(w, h) * 2 + max(4, round(min(w, h) / 300))  # 같은 선으로 볼 거리
        xs = _cluster([c[0] for c in cs] + [c[2] for c in cs], tol)
        ys = _cluster([c[1] for c in cs] + [c[3] for c in cs], tol)
        near = lambda arr, v: int(np.argmin([abs(a - v) for a in arr]))
        out = []
        for c in cs:
            c0, c1 = near(xs, c[0]), near(xs, c[2])
            r0, r1 = near(ys, c[1]), near(ys, c[3])
            if c1 <= c0 or r1 <= r0:
                continue
            out.append({"r": r0, "c": c0, "rs": r1 - r0, "cs": c1 - c0, "box": [c[0] / s, c[1] / s, c[2] / s, c[3] / s]})
        if len(out) < 4 or len(xs) < 2 or len(ys) < 2:
            continue
        # 실제로 쓰이는 행·열만 남겨 번호를 다시 매김
        ur = sorted({o["r"] for o in out} | {o["r"] + o["rs"] for o in out})
        uc = sorted({o["c"] for o in out} | {o["c"] + o["cs"] for o in out})
        for o in out:
            r0, r1 = ur.index(o["r"]), ur.index(o["r"] + o["rs"])
            c0, c1 = uc.index(o["c"]), uc.index(o["c"] + o["cs"])
            o.update(r=r0, rs=r1 - r0, c=c0, cs=c1 - c0)
        if not _is_grid(out, len(ur) - 1, len(uc) - 1):
            continue
        out.sort(key=lambda o: (o["r"], o["c"]))
        bx = [min(o["box"][0] for o in out), min(o["box"][1] for o in out), max(o["box"][2] for o in out), max(o["box"][3] for o in out)]
        tables.append({"box": bx, "rows": len(ur) - 1, "cols": len(uc) - 1, "cells": out})
    tables.sort(key=lambda t: (t["box"][1], t["box"][0]))
    return tables


def _is_grid(cells: list[dict], rows: int, cols: int) -> bool:
    """진짜 표인지: 세로 경계가 여러 행에 걸쳐 같은 자리에 있어야 한다.
    (형광펜 띠·공책 줄·손글씨 세로획으로 생긴 가짜 칸을 거른다)"""
    if cols < 2:
        return False  # 한 열짜리는 공책 줄 같은 것 → 줄 단위로 읽는 편이 낫다
    if rows == 1:
        return cols >= 2
    inner = [k for k in range(1, cols) if any(c["c"] == k for c in cells)]
    if not inner:
        return False
    def rows_at(k):  # 경계 k 에 맞닿은 칸들의 시작 행
        return {c["r"] for c in cells if c["c"] == k or c["c"] + c["cs"] == k}
    shared = sum(1 for k in inner if len(rows_at(k)) >= 2)
    return shared >= max(1, len(inner) * 0.5)


def assign(tables: list[dict], items: list[dict]) -> list[dict]:
    """글 조각(items: box=[x1,y1,x2,y2] px, text, conf …)을 칸에 넣는다. 칸에 들어간 조각은 it['cell']=(표,칸) 로 표시.
    한 조각이 두 칸 이상에 걸치면 가장 많이 겹친 칸으로."""
    for it in items:
        x1, y1, x2, y2 = it["box"]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        best, bv, inside = None, 0.0, 0.0
        for ti, t in enumerate(tables):
            tb = t["box"]
            if x2 < tb[0] or x1 > tb[2] or y2 < tb[1] or y1 > tb[3]:
                continue
            inside = max(inside, max(0, min(x2, tb[2]) - max(x1, tb[0])) * max(0, min(y2, tb[3]) - max(y1, tb[1])))
            for ci, c in enumerate(t["cells"]):
                b = c["box"]
                ov = max(0, min(x2, b[2]) - max(x1, b[0])) * max(0, min(y2, b[3]) - max(y1, b[1]))
                if ov > bv:
                    best, bv = (ti, ci), ov
        if best and (bv >= area * 0.4 or inside >= area * 0.7):  # 잘게 나뉜 칸에 걸친 글은 가장 많이 겹친 칸으로
            it["cell"] = best
    return items


def cell_text(pieces: list[dict]) -> list[list[dict]]:
    """칸 안 조각들을 위→아래 줄, 줄 안은 왼→오 순서로 묶는다."""
    ps = sorted(pieces, key=lambda p: (p["box"][1] + p["box"][3]) / 2)
    rows: list[list[dict]] = []
    for p in ps:
        cy, hh = (p["box"][1] + p["box"][3]) / 2, p["box"][3] - p["box"][1]
        if rows:
            last = rows[-1]
            ly = sum((q["box"][1] + q["box"][3]) / 2 for q in last) / len(last)
            lh = sum(q["box"][3] - q["box"][1] for q in last) / len(last)
            if abs(cy - ly) < min(lh, hh) * 0.55:
                last.append(p)
                continue
        rows.append([p])
    for r in rows:
        r.sort(key=lambda p: p["box"][0])
    return rows


def _join(pieces_rows: list[list[dict]]) -> tuple[str, list[int], float]:
    """칸 글자 합치기 → (글, 확인할 숫자 위치, 최저 확신도). 한두 글자씩 세로로 쌓인 칸(선/결, 접/수)은 붙여 쓴다."""
    stacked = len(pieces_rows) >= 2 and all(len("".join(p["text"] for p in r).replace(" ", "")) <= 2 for r in pieces_rows)
    text, chk, off = "", [], 0
    for i, r in enumerate(pieces_rows):
        for j, p in enumerate(r):
            sep = "" if not text else ("" if stacked else ("\n" if j == 0 else " "))
            if stacked:
                t = p["text"].replace(" ", "")
                text += t
                off += len(t)
                continue
            text += sep
            off += len(sep)
            chk += [off + k for k in p.get("chk", [])]
            text += p["text"]
            off += len(p["text"])
    conf = min(p["conf"] for r in pieces_rows for p in r)
    return text, chk, conf


def prune_columns(t: dict) -> None:
    """면적 칸처럼 아주 좁은 칸들이 잘게 나뉜 열은, 글이 한쪽에만 있으면 옆 칸과 합친다."""
    W = (t["box"][2] - t["box"][0]) or 1
    while True:
        xs = {}
        for c in t["cells"]:
            xs.setdefault(c["c"], c["box"][0]); xs.setdefault(c["c"] + c["cs"], c["box"][2])
        done = False
        for k in range(1, t["cols"]):
            L = [c for c in t["cells"] if c["c"] + c["cs"] == k]
            R = [c for c in t["cells"] if c["c"] == k]
            if not L or len(L) != len(R):
                continue
            narrow = any(c["box"][2] - c["box"][0] < W * 0.03 for c in L + R)
            if not narrow:
                continue
            pairs, ok = [], True
            for a in L:
                b = next((b for b in R if b["r"] == a["r"] and b["rs"] == a["rs"]), None)
                if b is None or (a.get("text") and b.get("text")):
                    ok = False
                    break
                pairs.append((a, b))
            if not ok:
                continue
            for a, b in pairs:
                keep = b if b.get("text") and not a.get("text") else a
                for key in ("text", "check", "conf", "pieces"):
                    if key in keep:
                        a[key] = keep[key]
                    elif key in a and keep is not a:
                        a.pop(key)
                a["box"] = [min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]), max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]
                a["cs"] += b["cs"]
                t["cells"].remove(b)
            for c in t["cells"]:
                if c["c"] >= k:
                    c["c"] -= 1
                elif c["c"] + c["cs"] > k:
                    c["cs"] -= 1
            t["cols"] -= 1
            done = True
            break
        if not done:
            return


def split_boxes(tables: list[dict], boxes: list) -> list:
    """글줄 상자(pts, score)가 칸 세로 경계를 넘으면 칸마다 따로 자른다 (거의 수평인 상자만)."""
    out = []
    for pts, sc in boxes:
        x1, y1, x2, y2 = float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())
        hh = y2 - y1
        e = pts[1] - pts[0]
        an = abs(np.degrees(np.arctan2(e[1], e[0]))) % 180
        flat = min(an, 180 - an) < 4 if np.hypot(*e) > 1 else False
        segs = []
        if flat and x2 - x1 > hh * 1.5:
            for t in tables:
                for c in t["cells"]:
                    b = c["box"]
                    oy = min(y2, b[3]) - max(y1, b[1])
                    ox = min(x2, b[2]) - max(x1, b[0])
                    if oy >= hh * 0.5 and ox >= max(4, hh * 0.4):
                        segs.append((max(x1, b[0]), min(x2, b[2])))
        if len(segs) < 2:
            out.append((pts, sc))
            continue
        segs.sort()
        for a, b in segs:
            out.append((np.array([[a, y1], [b, y1], [b, y2], [a, y2]], np.float32), sc))
    return out
