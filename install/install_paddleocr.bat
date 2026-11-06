@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
title IUX OCR - PaddleOCR 설치

if not exist ".venv\Scripts\python.exe" (
  echo  먼저 run_windows.bat 을 한 번 실행해 기본 환경을 만드세요.
  pause
  exit /b 1
)

echo.
echo  정확 모드(PaddleOCR 한국어 PP-OCRv5)를 설치합니다. 약 1GB, 인터넷 필요.
echo.
".venv\Scripts\python.exe" -m pip install "paddlepaddle>=3.0" "paddleocr>=3.0"
if errorlevel 1 (
  echo  설치 실패. 64비트 Python 3.9~3.12 인지 확인하세요.
  pause
  exit /b 1
)

echo.
echo  한국어 모델을 미리 받아 둡니다 (이후 오프라인 사용 가능)...
".venv\Scripts\python.exe" -c "from paddleocr import PaddleOCR; PaddleOCR(lang='korean', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False); print('완료')"
echo.
echo  설치가 끝났습니다. 프로그램을 다시 실행하면 '정확' 엔진이 켜집니다.
pause
