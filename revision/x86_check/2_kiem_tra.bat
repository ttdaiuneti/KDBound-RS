@echo off
rem Buoc 2: kiem tra file, phien ban thu vien va chay thu mot don vi (khoang 1 giay)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run_check.py check
pause
