@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
title IUX OCR - 오프라인 설치 묶음 만들기

rem 인터넷 되는 PC에서 실행 → wheels 폴더 생성
rem 프로그램 폴더 전체를 USB로 옮기면, 인터넷 없는 PC에서도 run_windows.bat 으로 설치됩니다.
rem (두 PC의 Python 버전이 같아야 합니다)

set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -m pip download -r requirements.txt -d wheels
echo.
echo  wheels 폴더가 만들어졌습니다. Tesseract 설치 파일도 함께 USB에 담아 가세요.
pause
