@echo off
chcp 65001 >nul
cd /d "%~dp0"
title IUX OCR

set "PY=python"
where py >nul 2>nul && set "PY=py -3"

if exist ".venv\Scripts\python.exe" goto RUN

echo.
echo  [처음 실행] 프로그램 환경을 준비합니다. 1~3분 걸립니다...
echo.
%PY% -m venv .venv
if errorlevel 1 (
  echo  파이썬을 찾을 수 없습니다. python.org 에서 Python 3.10 이상을 설치하세요.
  echo  설치할 때 "Add python.exe to PATH" 를 꼭 체크하세요.
  pause
  exit /b 1
)

if exist "wheels\" (
  echo  wheels 폴더의 오프라인 설치 파일로 설치합니다.
  ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheels -r requirements.txt
) else (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 (
  echo  패키지 설치에 실패했습니다. 인터넷 연결을 확인하세요.
  rmdir /s /q .venv
  pause
  exit /b 1
)

:RUN
".venv\Scripts\python.exe" -m app.server
pause
