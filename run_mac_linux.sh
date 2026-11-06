#!/usr/bin/env bash
# IUX OCR 실행 (macOS / Linux)
# 사전 설치: macOS → brew install tesseract tesseract-lang / Ubuntu → sudo apt install tesseract-ocr tesseract-ocr-kor
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "[처음 실행] 프로그램 환경을 준비합니다..."
  python3 -m venv .venv || exit 1
  if [ -d wheels ]; then
    .venv/bin/python -m pip install --no-index --find-links wheels -r requirements.txt || exit 1
  else
    .venv/bin/python -m pip install -r requirements.txt || exit 1
  fi
fi
exec .venv/bin/python -m app.server "$@"
