# IUX OCR — 작업 안내서

## 한 줄 요약
스캔 문서·공문서를 **인터넷 없이 내 PC에서** 텍스트로 바꾸고 정리·항목 추출·마스킹·재가공하는 OCR 도구. 설치형(exe)·포터블·웹 3가지.

- 웹: https://ocr.iux.kr (GitHub Pages, `CNAME`) / 설치형: GitHub Releases
- 기획·경쟁사 분석: `docs/기획서.md`

## 확정된 결정
| 항목 | 결정 |
|---|---|
| 동작 | 로컬 우선. 서버는 `127.0.0.1`에서만 열림, 문서가 PC 밖으로 안 나감 |
| 핵심 흐름 | 본문 + 사용자가 정한 항목을 추출 → 항목별 확인 → 개별/전체 복사로 업무 시스템에 옮김 (helper.js로 업무 화면 자동 입력) |
| 설치형 | Windows exe. `main` 푸시 → GitHub Actions가 빌드해 Releases에 올림 |
| 웹 | Tesseract.js(브라우저). 정확·AI 모드·폴더 일괄·검색 PDF는 설치형 전용 |

## 파일
```
app/server.py       로컬 서버 (API)
app/engines.py      OCR 엔진 (Tesseract / PaddleOCR / AI)
app/preprocess.py   전처리 (기울기·이진화)
app/extract.py      항목 추출  · app/refine.py 재가공 · app/exporters.py 저장(TXT/Word/Excel/PDF)
app/static/index.html   화면 원본 (설치형·웹 공용)
app/static/helper.js    업무 화면 자동 입력 도우미
index.html          웹버전 (tools/build_web.py가 app/static/index.html로 생성 — 직접 고치지 말 것)
web/shim.js         웹버전에서 서버 API를 브라우저로 대신 (IndexedDB, Tesseract.js, pdf.js)
web/rules.js        웹버전 항목 추출 규칙
web/vendor/         Tesseract·pdf.js·SheetJS·언어 데이터 (수정 금지)
install/            설치 프로그램(Inno Setup), 오프라인 번들
.github/workflows/build-windows.yml   exe 자동 빌드
```

## 자주 하는 수정
- **화면 수정:** `app/static/index.html` 고침 → `python tools/build_web.py`로 루트 `index.html` 다시 만들기 → 둘 다 커밋
- **항목 규칙:** 설치형은 `app/extract.py`, 웹은 `web/rules.js` (둘을 같이 맞출 것)
- 환경 변수: `IUX_OCR_PORT`, `IUX_OCR_WORKSPACE`, `TESSERACT_CMD`, `OLLAMA_HOST`(AI 교정)

## 확인
`run_windows.bat`(또는 `./run_mac_linux.sh`)로 띄워 샘플 스캔 1장 인식 → 항목 추출 → 저장까지. 웹은 ocr.iux.kr 또는 루트 `index.html`.

---

## 공통 작업 원칙 (IUX 프로젝트 전체)
- **언어:** 화면 문구·문서·커밋 메시지는 한국어. 쉬운 말, 기술 용어 최소화.
- **퍼블리싱 시안:** 순수 HTML/CSS + 간단한 스크립트만 넣는다. 서버·DB 연동 코드는 개발 단계에서 따로 붙인다.
- **아이콘:** Lucide. 디자인 시스템을 따로 정하지 않았으면 스포티파이 스타일(다크/화이트 + 포인트 컬러)을 기본으로 쓴다.
- **배포:** GitHub `main` → Vercel 자동 배포. 도메인은 가비아 `iux.kr`의 서브도메인(`*.iux.kr` CNAME → Vercel)을 쓴다.
- **DB:** Supabase 무료 플랜은 활성 프로젝트 2개까지라, 새 프로젝트를 만들지 말고 기존 `iux-cloud`(studio) 프로젝트에 테이블을 얹는 것을 먼저 검토한다.
- **비밀값:** 공개 저장소다. 서비스 키·AI 키·비밀번호는 절대 커밋하지 않는다. 키는 Vercel 환경 변수나 Supabase secrets에 둔다. Supabase anon(public) 키만 코드에 넣을 수 있다.
- **확인:** 고친 뒤에는 브라우저 콘솔 오류 0개, 모바일(390px)과 PC 화면을 눈으로 확인하고 나서 완료라고 한다.
- **브랜치:** `main`은 배포 중인 상태. 큰 작업이나 검증 전 작업은 브랜치(`wip/…`, `claude/…`)에 올리고, `main`을 강제 푸시(-f)로 덮어쓰지 않는다.
- **전체 프로젝트 목록:** https://github.com/handgoldking1234/iux-app/blob/main/docs/PROJECTS.md
