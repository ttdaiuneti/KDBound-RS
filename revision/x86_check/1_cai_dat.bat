@echo off
rem Buoc 1: tao moi truong Python rieng va cai thu vien (can Internet, chi lam mot lan)
cd /d "%~dp0"
set PY=
py -3.12 --version >nul 2>&1 && set PY=py -3.12
if not defined PY py -3.11 --version >nul 2>&1 && set PY=py -3.11
if not defined PY py -3.10 --version >nul 2>&1 && set PY=py -3.10
if not defined PY python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set PY=python
if not defined PY (
  echo Khong tim thay Python 3.10, 3.11 hoac 3.12. Cai Python 3.12 tu python.org roi chay lai.
  pause
  exit /b 1
)
echo Dung %PY%
%PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -c "import sys,struct,platform; print(sys.version.split()[0], struct.calcsize('P')*8, 'bit', platform.machine())"
python -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 (
  echo Khong co ban cai san cho Python nay. Can Python 3.12 64-bit tren may Intel/AMD.
  pause
  exit /b 1
)
echo.
echo Xong buoc 1. Chay tiep 2_kiem_tra.bat
pause
