"""웹버전(GitHub Pages) 만들기: app/static/index.html → 저장소 루트 index.html.
브라우저 안에서 동작하도록 web/rules.js(규칙 엔진)와 web/shim.js(서버 흉내)를 끼워 넣는다.
사용: python tools/build_web.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "app" / "static" / "index.html").read_text("utf-8")
inject = '<script src="web/rules.js"></script>\n<script src="web/shim.js"></script>\n<script>\n"use strict";'
assert '<script>\n"use strict";' in src
out = src.replace('<script>\n"use strict";', inject, 1)
out = out.replace('<meta name="viewport"', '<meta name="description" content="IUX OCR 웹 — 설치 없이 브라우저에서 스캔 문서·공문서를 텍스트로. 문서는 서버로 전송되지 않습니다.">\n<meta name="viewport"', 1)
(ROOT / "index.html").write_text(out, "utf-8")
(ROOT / ".nojekyll").write_text("", "utf-8")
print("index.html 생성 완료")
