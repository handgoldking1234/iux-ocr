"""본문 추출 + 사용자 정의 항목(템플릿) 추출."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from . import refine

# ================================================================ 본문 추출

_HEAD = re.compile(r"^\s*(?:제\s*목|수\s*신(?:\s*자)?|\(?\s*경\s*유\s*\)?|참\s*조)\b")
_TITLE = re.compile(r"^\s*제\s*목\s*[:：]?\s*\S")
_END = re.compile(
    r"^\s*(?:붙\s*임|기\s*안\s*자|검\s*토\s*자|결\s*재\s*권\s*자|협\s*조\s*자|시\s*행\s+\S|접\s*수\b|우\s*\d{5}|전\s*화\s*[\d(]|"
    r"수\s*신\s*자\s*[:：]|[가-힣·\s]{2,30}(?:장|청장|총장|원장|대표|위원장|이사장|교육감|군수|구청장|시장|도지사)\s*(?:\(?(?:직인|인)\)?)?\s*$)"
)


def extract_body(text: str) -> str:
    """공문서에서 머리(기관명·수신·제목)와 꼬리(붙임·발신명의·결재란·연락처)를 빼고 본문만 남긴다."""
    lines = refine.normalize(text).splitlines()
    if not lines:
        return ""
    start = 0
    t_idx = next((i for i, l in enumerate(lines) if _TITLE.match(l)), None)
    if t_idx is not None:
        start = t_idx + 1
    else:
        heads = [i for i, l in enumerate(lines[:15]) if _HEAD.match(l)]
        if heads:
            start = heads[-1] + 1
    end = len(lines)
    for i in range(start, len(lines)):
        l = lines[i]
        if not l.strip():
            continue
        if _END.match(l) and i > start:
            end = i
            break
        if re.search(r"끝\s*\.?\s*$", l):
            lines[i] = re.sub(r"\s*끝\s*\.?\s*$", "", l)
            end = i + 1
            break
    body = "\n".join(lines[start:end]).strip()
    return refine.clean_text(body) if body else ""


# ================================================================ 사용자 정의 항목

PRESETS: dict[str, tuple[str, str]] = {
    "date": ("날짜", refine.DATE + r"|\d{4}-\d{2}-\d{2}"),
    "period": ("기간(~)", r"(?:" + refine.DATE + r")\s*[~∼～\-]\s*(?:" + refine.DATE + r")"),
    "money": ("금액", r"(?:금\s*)?\d[\d,]*(?:\.\d+)?\s*(?:조|억|천만|백만|만|천)?\s*원"),
    "number": ("수량·숫자+단위", r"\d[\d,]*(?:\.\d+)?\s*(?:매|건|부|권|명|개소|개|대|회|차|%|㎡|평|km|kg)"),
    "phone": ("전화번호", r"(?<!\d)(?:01[016789]|0[2-6][1-5]?|070)\s*[-.)]?\s*\d{3,4}\s*[-.]\s*\d{4}(?!\d)"),
    "email": ("이메일", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "docno": ("문서번호", refine.DOCNO),
    "bizno": ("사업자등록번호", r"(?<!\d)\d{3}-\d{2}-\d{5}(?!\d)"),
    "url": ("홈페이지 주소", r"(?:https?://|www\.)[^\s/]+(?:/[^\s]*)?"),
}

GONGMUN_KEYS = {
    "title": "제목", "exec_no": "시행 문서번호", "exec_date": "시행일자", "recv_no": "접수 문서번호",
    "recv_date": "접수일자", "receiver": "수신", "via": "경유", "attach": "붙임", "sender": "발신명의",
    "drafter": "기안자", "reviewer": "검토자", "approver": "결재권자", "tel": "전화", "fax": "전송",
    "email": "이메일", "zip": "우편번호", "address": "주소", "disclosure": "공개구분",
}

# 다른 라벨 앞에서 값을 끊기 위한 흔한 라벨들
_STOP_LABELS = ["기안자", "검토자", "결재권자", "협조자", "시행", "접수", "전화", "전송", "공개", "담당자", "담당", "연락처", "주소"]


def _label_rx(label: str) -> str:
    """'사업기간' → '사\\s*업\\s*기\\s*간' (OCR 띄어쓰기 오류 허용)."""
    return r"\s*".join(re.escape(c) for c in label.replace(" ", ""))


def _by_label(lines: list[str], labels: list[str], stops: list[str], multi: bool) -> list[str]:
    alt = "|".join(_label_rx(l) for l in labels if l.strip())
    if not alt:
        return []
    rx = re.compile(r"(?:^|[\s(\[○□\-·|])(?:" + alt + r")\s*(?:[:：|]|은|는)?\s*([^|]*)")  # 표 행(칸 | 칸)은 칸 경계까지만
    stop_rx = re.compile(r"\s(?:" + "|".join(_label_rx(s) for s in stops) + r")\s*[:：]?\s") if stops else None
    out = []
    for i, l in enumerate(lines):
        m = rx.search(l)
        if not m:
            continue
        v = m.group(1).strip()
        if not v:  # 라벨만 있고 값은 다음 줄
            v = next((x.strip() for x in lines[i + 1:i + 3] if x.strip()), "")
        if stop_rx:
            sm = stop_rx.search(" " + v + " ")
            if sm:
                v = (" " + v)[:sm.start()].strip()
        v = v.strip(" :：,")
        if v:
            out.append(v)
            if not multi:
                break
    return out


FORMATS = {
    "": "그대로",
    "ymd-dash": "2026-11-06",
    "ymd": "20261106",
    "ymd-dot": "2026.11.06",
    "digits": "숫자만",
    "oneline": "한 줄로",
}


def _fmt_date(v: str, kind: str) -> str:
    def conv(m):
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        return {"ymd-dash": f"{y}-{mo:02d}-{d:02d}", "ymd": f"{y}{mo:02d}{d:02d}", "ymd-dot": f"{y}.{mo:02d}.{d:02d}"}[kind]
    return re.sub(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})\s*[.일]?", conv, v)


def apply_format(v: str, fmt: str | None) -> str:
    if not v or not fmt:
        return v
    if fmt in ("ymd-dash", "ymd", "ymd-dot"):
        return _fmt_date(v, fmt).strip()
    if fmt == "digits":
        return " / ".join(re.sub(r"[^\d.]", "", x) for x in v.split(" / "))
    if fmt == "oneline":
        return re.sub(r"\s*\n\s*", " ", v).strip()
    return v


def run_template(text: str, fields: list[dict], model: str | None = None) -> list[dict]:
    lines = refine.normalize(text).splitlines()
    joined = "\n".join(lines)
    labels_all = [l for f in fields if f.get("type") == "label" for l in re.split(r"[|,/]", f.get("label", "")) if l.strip()]
    gongmun = None
    results: list[dict] = []
    ai_fields: list[tuple[int, dict]] = []
    for f in fields:
        name, typ, multi = f.get("name", "").strip() or "항목", f.get("type", "label"), bool(f.get("multi"))
        val, note = "", ""
        try:
            if typ == "label":
                labs = [l.strip() for l in re.split(r"[|,/]", f.get("label") or name) if l.strip()]
                stops = [s for s in labels_all + _STOP_LABELS if s not in labs]
                vals = _by_label(lines, labs, stops, multi)
                val = " / ".join(vals)
            elif typ == "preset":
                pat = PRESETS.get(f.get("preset", ""), ("", ""))[1]
                found = [re.sub(r"\s+", " ", m.group(0)).strip() for m in re.finditer(pat, joined)] if pat else []
                found = list(dict.fromkeys(found))
                val = " / ".join(found if multi else found[:1])
            elif typ == "regex":
                rx = re.compile(f.get("pattern", ""), re.M)
                found = []
                for m in rx.finditer(joined):
                    found.append((m.group(1) if m.groups() else m.group(0)).strip())
                    if not multi:
                        break
                val = " / ".join(dict.fromkeys(found))
            elif typ == "gongmun":
                if gongmun is None:
                    gongmun = {x["key"]: x["value"] for x in refine.extract_fields(text)}
                val = gongmun.get(f.get("key", ""), "")
            elif typ == "body":
                val = extract_body(text)
            elif typ == "ai":
                ai_fields.append((len(results), f))
                note = "ai"
        except re.error as e:
            val, note = "", f"정규식 오류: {e}"
        results.append({"name": name, "value": val, "type": typ, "note": note})

    if ai_fields:
        if not model:
            for i, _ in ai_fields:
                results[i]["note"] = "AI 모델이 없어 건너뜀 (Ollama 필요)"
        else:
            from .engines import ollama_chat
            spec = "\n".join(f"- {f.get('name')}: {f.get('desc') or f.get('name')}" for _, f in ai_fields)
            prompt = (
                "다음 문서에서 아래 항목의 값을 찾아 JSON 객체로만 답하세요.\n"
                "- 키는 항목명을 그대로 사용\n- 값은 문서에 있는 표현을 그대로 발췌 (요약·추측 금지)\n"
                "- 여러 개면 ' / '로 연결, 없으면 빈 문자열\n\n"
                f"항목:\n{spec}\n\n<문서>\n{text[:12000]}\n</문서>"
            )
            try:
                data = json.loads(ollama_chat(model, prompt, system=refine.AI_SYSTEM, fmt="json"))
            except Exception as e:  # noqa: BLE001
                data, err = {}, str(e)
            else:
                err = ""
            for i, f in ai_fields:
                v = data.get(f.get("name"), "")
                if isinstance(v, (list, tuple)):
                    v = " / ".join(map(str, v))
                results[i]["value"] = str(v or "")
                results[i]["note"] = f"AI 오류: {err}" if err else "ai"
    for r, f in zip(results, fields):
        r["value"] = apply_format(r["value"], f.get("fmt"))
    return results


# ================================================================ 자동 항목 (web/rules.js 의 autoFields 와 같은 규칙)

STRONG = ["수신자", "수신", "경유", "참조", "받음", "발신명의", "발신", "제목", "문서번호", "시행일자", "시행", "접수일자", "접수",
          "보존기간", "공개여부", "공개구분", "기안자", "검토자", "결재권자", "협조자", "담당자", "붙임", "우편번호", "전화번호", "전화", "전송", "팩스",
          "홈페이지", "전자우편", "생산기관", "생산일자", "심사자", "처리과", "등록번호", "분류번호", "건명"]
_STRONG_ALT = "|".join(_label_rx(x) for x in sorted(STRONG, key=len, reverse=True))
_RX_STRONG = re.compile(r"(^|[\s/|·])[\[(（]?\s*(" + _STRONG_ALT + r")\s*[\])）]?(?=\s*[:：]|\s|$)")
_LIST_MARK = re.compile(r"^\s*(?:\d{1,2}\.|[가-하]\.|\d{1,2}\)|[가-하]\)|\(\d{1,2}\)|\([가-하]\)|[-·•○□■◦※])\s*")


def _is_syl(t: str) -> bool:
    return len(t) == 1 and "가" <= t <= "힣"


def _split_label(seg: str) -> tuple[str, str]:
    toks = seg.split()
    if not toks:
        return "", ""
    j = len(toks) - 1
    if _is_syl(toks[j]):
        while j > 0 and _is_syl(toks[j - 1]):
            j -= 1
    return "".join(toks[j:]), " ".join(toks[:j])


def _colon_pairs(line: str) -> list[tuple[str, str]]:
    L = _LIST_MARK.sub("", line, count=1)
    L = re.sub(r"^[\[(（]\s*|\s*[\])）](?=\s*[:：])", "", L)
    parts = re.split(r"(?<!\d)\s*[:：]\s*", L)
    if len(parts) < 2:
        return []
    if len(parts) == 2:
        lab = re.sub(r"\s+", "", parts[0])
        if not re.fullmatch(r"[가-힣]{2,8}|[A-Za-z]{2,15}", lab):
            return []
        return [(lab, parts[1])]
    pairs = []
    cur_label, _ = _split_label(parts[0])
    for k in range(1, len(parts)):
        if k < len(parts) - 1:
            nl, rest = _split_label(parts[k])
            pairs.append((cur_label, rest))
            cur_label = nl
        else:
            pairs.append((cur_label, parts[k]))
    return [(l, v) for l, v in pairs if re.fullmatch(r"[가-힣]{2,10}|[A-Za-z]{2,15}", l)]


def _clean_val(v: str) -> str:
    v = re.sub(r"^[\s:：]+|[\s/|:：]+$", "", v)
    v = re.sub(r"\s+([.,)\]])", r"\1", v)
    v = re.sub(r"([(\[])\s+", r"\1", v)
    from .engines import _looks_split
    if _looks_split(v):
        v = refine.fix_syllable_spacing(v)
    return v.strip()


def _is_label(c: str) -> bool:
    """표 칸 글이 이름표처럼 보이는지 (짧은 한글·영문, 숫자 없음)"""
    t = re.sub(r"[\s()（）]", "", c)
    return bool(re.fullmatch(r"[가-힣A-Za-z·/]{1,12}", t)) and not re.search(r"\d", c)


def auto_fields(text: str) -> list[dict]:
    """문서 속 '수신 ○○', '제 목 ○○', '참조 : ○○' 같은 이름표와 값을 찾아 항목으로 만든다."""
    lines = refine.normalize(text).split("\n")
    out: list[dict] = []
    seen: dict[str, int] = {}

    def add(name: str, value: str):
        name = re.sub(r"\s+", "", name)
        value = _clean_val(value)
        if not name or not value:
            return
        if name == "수신자":
            name = "수신"
        seen[name] = seen.get(name, 0) + 1
        out.append({"name": f"{name} {seen[name]}" if seen[name] > 1 else name, "value": value})

    head: list[str] | None = None
    for line in lines:
        if " | " in line:  # 표 행: 이름 칸 | 값 칸 짝, 또는 머리 행 아래 값 행
            cells = [c.strip() for c in line.split(" | ")]
            labs = [_is_label(c) for c in cells]
            if all(labs) and len(cells) >= 2:
                head = cells
                continue
            if head and len(cells) == len(head):
                for h_, v in zip(head, cells):
                    add(h_, v)
                continue
            head = None
            k = 0
            while k < len(cells) - 1:
                if labs[k] and not labs[k + 1]:
                    add(cells[k], cells[k + 1])
                    k += 2
                else:
                    k += 1
            continue
        head = None
        ms = list(_RX_STRONG.finditer(line))
        if ms and ms[0].start() == 0:
            segs = [(m.group(2), m.start() + len(m.group(1)), m.end()) for m in ms]
            for k, (label, _st, end) in enumerate(segs):
                v = line[end: segs[k + 1][1] if k + 1 < len(segs) else len(line)]
                v = re.sub(r"^\s*[\])）]?\s*[:：]?", "", v, count=1)
                if re.sub(r"\s", "", label) == "붙임":
                    v = re.sub(r"\s*끝\s*\.?\s*$", "", v)
                add(label, v)
            continue
        for l, v in _colon_pairs(line):
            add(l, v)
    names = {re.sub(r" \d+$", "", o["name"]) for o in out}
    gm = {f["key"]: f["value"] for f in refine.extract_fields(text)}
    for n, k in (("문서번호", "exec_no"), ("시행일자", "exec_date"), ("발신명의", "sender")):
        if gm.get(k) and n not in names:
            out.append({"name": n, "value": gm[k]})
    body = extract_body(text)
    if body:
        out.append({"name": "본문", "value": body})
    return out


AUTO_ID = "__auto__"
AUTO_TEMPLATE = {"id": AUTO_ID, "name": "자동 — 문서에서 항목 찾기", "fields": [], "auto": True}


def values_for(tpl: dict, text: str, model: str | None = None) -> list[dict]:
    """템플릿(또는 자동)으로 항목 값 목록을 만든다."""
    if tpl.get("auto"):
        return [{"name": v["name"], "value": v["value"], "type": "auto", "note": ""} for v in auto_fields(text)]
    return run_template(text, tpl["fields"], model)


# ================================================================ 템플릿 저장소

DEFAULT_TEMPLATES = [
    {
        "id": "records-meta",
        "name": "기록물 메타데이터 (예시)",
        "fields": [
            {"name": "제목", "type": "gongmun", "key": "title"},
            {"name": "생산기관", "type": "gongmun", "key": "sender"},
            {"name": "문서번호", "type": "gongmun", "key": "exec_no"},
            {"name": "생산일자", "type": "gongmun", "key": "exec_date", "fmt": "ymd-dash"},
            {"name": "수신", "type": "gongmun", "key": "receiver"},
            {"name": "사업기간", "type": "label", "label": "사업기간|사업 기간|기간"},
            {"name": "금액", "type": "preset", "preset": "money", "multi": True},
            {"name": "연락처", "type": "preset", "preset": "phone", "multi": True},
            {"name": "붙임", "type": "gongmun", "key": "attach"},
            {"name": "본문", "type": "body"},
        ],
    },
    {
        "id": "contract",
        "name": "계약서 (예시)",
        "fields": [
            {"name": "계약명", "type": "label", "label": "계약명|용역명|사업명|건명"},
            {"name": "계약금액", "type": "label", "label": "계약금액|총 계약금액|계약 금액"},
            {"name": "계약기간", "type": "preset", "preset": "period"},
            {"name": "발주처", "type": "label", "label": "발주처|발주기관|발주자"},
            {"name": "계약상대자", "type": "label", "label": "계약상대자|수급인|수급자"},
            {"name": "사업자등록번호", "type": "preset", "preset": "bizno", "multi": True},
        ],
    },
]


class TemplateStore:
    def __init__(self, path: Path):
        self.path = path

    def all(self) -> list[dict]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text("utf-8"))
            except Exception:  # noqa: BLE001
                pass
        return [dict(t) for t in DEFAULT_TEMPLATES]

    def save_all(self, items: list[dict]) -> list[dict]:
        clean = []
        for t in items:
            t = dict(t)
            if t.get("auto") or t.get("id") == AUTO_ID:
                continue
            t["id"] = t.get("id") or uuid.uuid4().hex[:8]
            t["name"] = (t.get("name") or "새 템플릿").strip()
            t["fields"] = [f for f in t.get("fields", []) if (f.get("name") or "").strip()]
            clean.append(t)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), "utf-8")
        return clean

    def get(self, tid: str | None) -> dict | None:
        if tid == AUTO_ID:
            return dict(AUTO_TEMPLATE)
        return next((t for t in self.all() if t["id"] == tid), None) if tid else None
