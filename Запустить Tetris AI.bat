@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing_venv
set "PROJECT_ROOT=%CD%"
set "RUNTIME_ROOT=%PROJECT_ROOT%\.runtime"
for %%D in (cache temp logs checkpoints pycache pip-cache torch-cache cuda-cache inductor-cache triton-cache numba-cache matplotlib data python-user-base) do if not exist "%RUNTIME_ROOT%\%%D" mkdir "%RUNTIME_ROOT%\%%D"
set "TEMP=%RUNTIME_ROOT%\temp"
set "TMP=%RUNTIME_ROOT%\temp"
set "TMPDIR=%RUNTIME_ROOT%\temp"
set "PIP_CACHE_DIR=%RUNTIME_ROOT%\pip-cache"
set "PYTHONPYCACHEPREFIX=%RUNTIME_ROOT%\pycache"
set "TORCH_HOME=%RUNTIME_ROOT%\torch-cache"
set "XDG_CACHE_HOME=%RUNTIME_ROOT%\cache"
set "CUDA_CACHE_PATH=%RUNTIME_ROOT%\cuda-cache"
set "TORCHINDUCTOR_CACHE_DIR=%RUNTIME_ROOT%\inductor-cache"
set "TRITON_CACHE_DIR=%RUNTIME_ROOT%\triton-cache"
set "NUMBA_CACHE_DIR=%RUNTIME_ROOT%\numba-cache"
set "MPLCONFIGDIR=%RUNTIME_ROOT%\matplotlib"
set "PYTHONUSERBASE=%RUNTIME_ROOT%\python-user-base"
set "PYTHONNOUSERSITE=1"
set "PYGAME_HIDE_SUPPORT_PROMPT=1"
".venv\Scripts\python.exe" -m tetris_ai
if errorlevel 1 pause
exit /b %errorlevel%
:missing_venv
echo Локальное окружение .venv не найдено.
echo Сначала запустите "Нажми на меня чтобы подготовиться к запуску.bat".
pause
exit /b 1
