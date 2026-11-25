@echo off
rem Buoc 3: chay toan bo cac don vi roi viet bang so sanh (1-2 gio). Bi ngat thi chay lai, cac don vi da xong duoc bo qua.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run_check.py run all
python run_check.py summary
echo.
echo Xong. Mang thu muc results ve.
pause
