"""텍스트 재가공 — 규칙 기반(오프라인 즉시) + 로컬 AI(Ollama)."""
from __future__ import annotations

import json
import re
import unicodedata

# ================================================================ 규칙 기반

SENT_END = re.compile(r"[.!?。:;)\]」』\"'”’…]$|다\.?$|요\.?$|함\.?$|음\.?$|것\.?$|임\.?$|끝\.?$")
LIST_START = re.compile(
    r"^\s*(?:\d{1,2}\.|[가-하]\.|\d{1,2}\)|[가-하]\)|\(\d{1,2}\)|\([가-하]\)|[①-⑳]|[㉮-㉻]|[□■○●◦◎▶▷►•·\-–※*]|붙\s*임|수\s*신|제\s*목|시\s*행|접\s*수)"
)
PAGE_NO = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$|^\s*\d{1,4}\s*/\s*\d{1,4}\s*$")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.replace("　", " ").replace(" ", " ").replace("\t", " ")
    t = re.sub(r"[ ]{2,}", " ", t)
    t = "\n".join(l.strip() for l in t.splitlines())
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


_SYL = re.compile(r"^[가-힣]$")
_STANDALONE = set("및 등 각 약 총 또 더 그 저 수 것 때 안 못 잘 꼭 곧 늘 왜 좀 별 전 후 매 본 동 당 귀 위 아래 우".split())


_UNITS = {"년", "월", "일", "매", "부", "개", "명", "원", "건", "권", "회", "차", "시", "분", "호"}
_PARTICLE_SPLIT = {"을", "를", "의", "과", "와", "는"}


def _split_particles(word: str) -> str:
    """붙여 쓴 음절열을 조사 뒤에서 끊기: '종이기록물의보존과' → '종이기록물의 보존과'."""
    out, cur = [], ""
    for k, ch in enumerate(word):
        cur += ch
        rest = len(word) - k - 1
        if ch in _PARTICLE_SPLIT and len(cur) >= 3 and rest >= 2:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return " ".join(out)


def fix_syllable_spacing(text: str) -> str:
    """Tesseract 한국어의 '기 관 의' 같은 음절 사이 잘못된 띄어쓰기 보정.
    한 글자 한글 토큰이 2개 이상 연속되면 붙이고, 바로 앞/뒤 토큰과도 이어 붙인다."""
    out_lines = []
    for line in text.split("\n"):
        toks = line.split(" ")
        res: list[str] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            if _SYL.match(t) and t not in _STANDALONE:
                j = i
                run = []
                while j < len(toks) and _SYL.match(toks[j]):
                    run.append(toks[j])
                    j += 1
                if len(run) >= 2:
                    if run[0] in _UNITS and res and re.search(r"\d$", res[-1]):
                        res[-1] += run.pop(0)
                    if len(run) < 2:
                        res.extend(run)
                        i = j
                        continue
                    word = _split_particles("".join(run))
                    # 앞 토큰이 한글 단어로 끝나고 1음절 조사로 시작하지 않으면 그대로, 뒤 토큰은 조사/어미면 붙임
                    if j < len(toks) and re.fullmatch(r"(?:[가-힣]{1,2})?(?:을|를|이|가|은|는|의|에|에서|으로|로|과|와|도|만|하여|하고|합니다\.?|니다\.?|다\.?)[.,]?", toks[j]):
                        word += toks[j]
                        j += 1
                    res.append(word)
                    i = j
                    continue
            # 다음절 단어 + 1음절 조사/접미 (예: '기록물 의', '추진 하')
            if res and _SYL.match(t) and t in {"의", "을", "를", "은", "는", "이", "가", "에", "과", "와", "로", "도", "만", "적", "장", "청", "과"} \
                    and re.search(r"[가-힣]{2,}$", res[-1]):
                res[-1] += t
                i += 1
                continue
            if res and t in _UNITS and re.search(r"\d$", res[-1]):
                res[-1] += t
                i += 1
                continue
            res.append(t)
            i += 1
        out_lines.append(" ".join(res))
    return "\n".join(out_lines)


def clean_text(text: str, merge_lines: bool = True, drop_page_no: bool = True, fix_spacing: bool = True) -> str:
    t = normalize(text)
    lines = t.splitlines()
    if drop_page_no:
        lines = [l for l in lines if not PAGE_NO.match(l)]
    if fix_spacing:
        fixed = []
        for l in lines:
            l = re.sub(r"\s+([,.)\]}:;!?])", r"\1", l)          # 문장부호 앞 공백
            l = re.sub(r"([(\[{])\s+", r"\1", l)                 # 여는 괄호 뒤 공백
            l = re.sub(r"^([가-하]|\d{1,2})\s+([.)])", r"\1\2", l)  # '라 .' → '라.'
            l = re.sub(r"(\d)\s*\.\s*(\d)(?!\d*\s*\.)", r"\1.\2", l)  # 1 . 5 → 1.5
            l = re.sub(r"(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.", r"\1. \2. \3.", l)  # 날짜 표기 통일
            fixed.append(l)
        lines = fixed
    if not merge_lines:
        return "\n".join(lines).strip()
    out: list[str] = []
    for l in lines:
        if not l:
            if out and out[-1] != "":
                out.append("")
            continue
        if out and out[-1] and not SENT_END.search(out[-1]) and not LIST_START.match(l):
            prev = out[-1]
            if prev.endswith("-") and re.match(r"[a-zA-Z]", l):
                out[-1] = prev[:-1] + l
            else:
                out[-1] = prev + " " + l
        else:
            out.append(l)
    return "\n".join(out).strip()


# ---------------------------------------------------------------- 공문서 항목 추출

DATE = r"\d{4}\s*[.\-/년]\s*\d{1,2}\s*[.\-/월]\s*\d{1,2}\s*[.일]?"
DOCNO = r"[가-힣A-Za-z0-9·]{1,20}\s?-\s?\d{1,7}"

FIELD_RULES: list[tuple[str, str, str]] = [
    ("receiver", "수신", r"수\s*신(?:\s*자)?\s*[:：]?\s*(.+)"),
    ("via", "경유", r"^\(?\s*경\s*유\s*\)?\s*[:：]?\s*([^\s)].*)$"),
    ("title", "제목", r"제\s*목\s*[:：]?\s*(.+)"),
    ("attach", "붙임", r"붙\s*임\s*[:：]?\s*(.+)"),
    ("drafter", "기안자", r"기\s*안\s*자\s*[:：]?\s*([^\s].{0,20}?)(?=\s{2,}|\s*검\s*토|\s*결\s*재|$)"),
    ("reviewer", "검토자", r"검\s*토\s*자\s*[:：]?\s*([^\s].{0,20}?)(?=\s{2,}|\s*결\s*재|\s*협\s*조|$)"),
    ("approver", "결재권자", r"결\s*재\s*권\s*자\s*[:：]?\s*([^\s].{0,20}?)(?=\s{2,}|\s*협\s*조|$)"),
    ("tel", "전화", r"전\s*화\s*(?:번\s*호)?\s*[:：]?\s*([\d()\-\s]{7,20}\d)"),
    ("fax", "전송", r"(?:전\s*송|팩\s*스|FAX)\s*[:：]?\s*([\d()\-\s]{7,20}\d)"),
    ("email", "이메일", r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})"),
    ("zip", "우편번호", r"(?:우\s*)?(\d{5})\s+[가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)"),
    ("address", "주소", r"(?:주\s*소\s*[:：]?\s*|\d{5}\s+)([가-힣]+(?:특별시|광역시|특별자치시|특별자치도|도)\s+[^\n/]{4,60})"),
    ("disclosure", "공개구분", r"((?:대시민\s*)?(?:비공개|부분\s*공개|공개)(?:\s*\([^)]{0,20}\))?)\s*$"),
]


def extract_fields(text: str) -> list[dict]:
    t = normalize(text)
    lines = t.splitlines()
    found: dict[str, dict] = {}

    def put(key, label, value):
        value = re.sub(r"\s{2,}", " ", value).strip(" :：")
        value = re.sub(r"\s+([.,)\]])", r"\1", value)
        value = re.sub(r"([(\[])\s+", r"\1", value)
        if value and key not in found:
            found[key] = {"key": key, "label": label, "value": value}

    # 시행/접수: "시행 총무과-1234 (2026. 11. 6.)"
    for kind, label in (("시\\s*행", "시행"), ("접\\s*수", "접수")):
        m = re.search(kind + r"\s*[:：]?\s*(" + DOCNO + r")?\s*\(?\s*(" + DATE + r")?", t)
        if m and (m.group(1) or m.group(2)):
            key = "exec" if label == "시행" else "recv"
            if m.group(1):
                put(key + "_no", f"{label} 문서번호", re.sub(r"\s", "", m.group(1)))
            if m.group(2):
                put(key + "_date", f"{label}일자", m.group(2))

    for key, label, pat in FIELD_RULES:
        for l in lines:
            m = re.search(pat, l)
            if m:
                put(key, label, m.group(1))
                break

    # 붙임이 여러 줄(1. ... 2. ...)인 경우 이어 붙이기
    if "attach" in found:
        idx = next((i for i, l in enumerate(lines) if re.search(r"붙\s*임", l)), None)
        if idx is not None:
            more = []
            for l in lines[idx + 1: idx + 8]:
                if not l.strip():
                    continue
                if re.match(r"^\s*\d{1,2}\.\s*\S", l) and not re.search(r"끝\.?\s*$", found["attach"]["value"]):
                    more.append(l.strip())
                else:
                    break
            if more:
                found["attach"]["value"] += " / " + " / ".join(more)
        found["attach"]["value"] = re.sub(r"\s*끝\s*\.?\s*$", "", re.sub(r"\s+\.", ".", found["attach"]["value"]))

    # 발신명의: "○○○장", "○○○청장" 단독 줄 (뒤쪽에서 찾기)
    for l in reversed(lines):
        s = l.strip()
        if re.fullmatch(r"[가-힣A-Za-z0-9·\s]{2,30}(?:장|청장|총장|원장|대표|위원장|이사장|교육감|군수|구청장|시장|도지사)(?:\s*\(?(?:직인|인)\)?)?", s):
            put("sender", "발신명의", re.sub(r"\s*\(?(직인|인)\)?$", "", s))
            break

    # 문서 안의 모든 날짜 (참고용)
    dates = list(dict.fromkeys(re.sub(r"\s+", " ", d).strip() for d in re.findall(DATE, t)))
    if dates:
        put("dates", "등장 날짜", ", ".join(dates[:8]))

    order = ["exec_no", "exec_date", "recv_no", "recv_date", "receiver", "via", "title", "attach", "sender",
             "drafter", "reviewer", "approver", "tel", "fax", "email", "zip", "address", "disclosure", "dates"]
    return [found[k] for k in order if k in found]


# ---------------------------------------------------------------- 개인정보 마스킹

PII_RULES = [
    ("rrn", "주민/외국인등록번호", re.compile(r"(?<!\d)(\d{6})\s*[-–]\s*([1-8])\d{6}(?!\d)"), lambda m: f"{m.group(1)}-{m.group(2)}******"),
    ("card", "카드번호", re.compile(r"(?<!\d)(\d{4})[\s\-](\d{4})[\s\-](\d{4})[\s\-](\d{4})(?!\d)"), lambda m: f"{m.group(1)}-****-****-{m.group(4)}"),
    ("phone", "전화번호", re.compile(r"(?<!\d)(01[016789]|0[2-6][1-5]?|070)\s*[-.)]?\s*(\d{3,4})\s*[-.]\s*(\d{4})(?!\d)"), lambda m: f"{m.group(1)}-{'*' * len(m.group(2))}-{m.group(3)}"),
    ("email", "이메일", re.compile(r"([A-Za-z0-9._%+\-]{1,3})([A-Za-z0-9._%+\-]*)(@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})"), lambda m: m.group(1) + "*" * len(m.group(2)) + m.group(3)),
    ("account", "계좌번호", re.compile(r"(계좌(?:번호)?\s*[:：]?\s*(?:[가-힣]+은행\s*)?)(\d[\d\-]{7,20}\d)"), lambda m: m.group(1) + re.sub(r"\d", "*", m.group(2)[:-4]) + m.group(2)[-4:]),
    ("passport", "여권번호", re.compile(r"(?<![A-Za-z0-9])([MSRODG])(\d{3})([A-Z]?\d{4,5})(?![A-Za-z0-9])"), lambda m: m.group(1) + m.group(2) + "*" * len(m.group(3))),
]


def mask_pii(text: str, kinds: list[str] | None = None) -> tuple[str, dict]:
    counts: dict[str, int] = {}
    for key, label, rx, repl in PII_RULES:
        if kinds and key not in kinds:
            continue
        text, n = rx.subn(repl, text)
        if n:
            counts[label] = n
    return text, counts


# ---------------------------------------------------------------- 항목체계 → 개요(마크다운)

LEVELS = [
    (re.compile(r"^(\d{1,2}\.)\s*(.*)"), 0),
    (re.compile(r"^([가-하]\.)\s*(.*)"), 1),
    (re.compile(r"^(\d{1,2}\))\s*(.*)"), 2),
    (re.compile(r"^([가-하]\))\s*(.*)"), 3),
    (re.compile(r"^(\(\d{1,2}\))\s*(.*)"), 4),
    (re.compile(r"^(\([가-하]\))\s*(.*)"), 5),
    (re.compile(r"^([①-⑳])\s*(.*)"), 6),
    (re.compile(r"^([㉮-㉻])\s*(.*)"), 7),
    (re.compile(r"^([□■])\s*(.*)"), 0),
    (re.compile(r"^([○●◦])\s*(.*)"), 1),
    (re.compile(r"^([-–])\s+(.*)"), 2),
    (re.compile(r"^([·•])\s*(.*)"), 3),
    (re.compile(r"^(※)\s*(.*)"), -1),
]


def to_outline(text: str) -> str:
    out = []
    for raw in clean_text(text, merge_lines=True).splitlines():
        l = raw.strip()
        if not l:
            out.append("")
            continue
        m = re.match(r"^제\s*목\s*[:：]?\s*(.+)", l)
        if m:
            out.append(f"# {m.group(1)}")
            continue
        for rx, lvl in LEVELS:
            m = rx.match(l)
            if m:
                if lvl < 0:
                    out.append(f"> ※ {m.group(2)}")
                else:
                    out.append("  " * lvl + f"- **{m.group(1)}** {m.group(2)}")
                break
        else:
            out.append(l)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ================================================================ AI (Ollama)

AI_SYSTEM = "당신은 한국어 공문서·기록물 전문 편집자입니다. 지시를 정확히 따르고, 결과물만 출력합니다. 서두나 설명을 붙이지 않습니다."

AI_TASKS = {
    "fix": "다음은 OCR로 인식한 한국어 문서입니다. 인식 오류(잘못 인식된 글자, 깨진 글자, 띄어쓰기 오류)만 바로잡으세요.\n"
           "- 내용을 추가·삭제·요약·재구성하지 마세요.\n- 줄 구성과 항목 기호(1. 가. 1) 등)를 유지하세요.\n"
           "- 확신이 없는 부분은 원문 그대로 두세요.\n- 교정된 전체 텍스트만 출력하세요.",
    "summary": "다음 문서의 핵심을 개조식으로 요약하세요.\n- 첫 줄: 한 문장 요지\n- 이어서 '- '로 시작하는 핵심 항목 3~7개\n"
               "- 날짜·금액·기관명·기한은 반드시 포함\n- 원문에 없는 내용은 쓰지 마세요.",
    "table": "다음 텍스트에서 표 형태의 데이터를 찾아 마크다운 표로 정리하세요. 표가 아닌 부분은 원문 그대로 두세요. 숫자와 단위를 바꾸지 마세요.",
    "structure": "다음 공문서를 JSON으로 구조화하세요. 키: 문서번호, 시행일자, 수신, 경유, 제목, 본문요약(3문장 이내), "
                 "주요항목(배열), 붙임(배열), 발신명의, 담당자, 연락처. 없는 값은 빈 문자열 또는 빈 배열. 원문에 없는 값은 만들지 마세요.",
    "translate": "다음 한국어 문서를 자연스러운 영어로 번역하세요. 문단·항목 구조를 유지하세요.",
    "plain": "다음 문서를 공문서 용어를 모르는 사람도 이해할 수 있게 쉬운 말로 풀어 쓰세요. 사실관계(날짜·금액·대상)는 그대로 유지하세요.",
}


def ai_refine(text: str, task: str, model: str, custom: str | None = None) -> str:
    from .engines import ollama_chat
    if task == "custom":
        if not custom:
            raise ValueError("지시 내용을 입력하세요.")
        instr = custom
    else:
        instr = AI_TASKS.get(task)
        if not instr:
            raise ValueError(f"알 수 없는 작업: {task}")
    prompt = f"{instr}\n\n<문서>\n{text}\n</문서>"
    fmt = "json" if task == "structure" else None
    out = ollama_chat(model, prompt, system=AI_SYSTEM, fmt=fmt).strip()
    if task == "structure":
        try:
            out = json.dumps(json.loads(out), ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            pass
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", out).strip()
    return out


# ================================================================ 진입점

def run_rule(mode: str, text: str, opts: dict) -> dict:
    if mode == "clean":
        return {"text": clean_text(text, opts.get("merge", True), opts.get("dropPageNo", True), opts.get("spacing", True))}
    if mode == "mask":
        t, counts = mask_pii(text, opts.get("kinds"))
        return {"text": t, "counts": counts}
    if mode == "outline":
        return {"text": to_outline(text)}
    if mode == "body":
        from .extract import extract_body
        return {"text": extract_body(text)}
    if mode == "fields":
        return {"fields": extract_fields(text)}
    if mode == "normalize":
        return {"text": normalize(text)}
    raise ValueError(f"알 수 없는 재가공: {mode}")
