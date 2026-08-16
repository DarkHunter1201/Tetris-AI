@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
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
set "PYTHON_CMD="
for %%V in (3.12 3.13 3.11 3.10 3.14) do call :try_launcher %%V
if not defined PYTHON_CMD call :try_python
if not defined PYTHON_CMD goto python_error
if not exist ".venv\Scripts\python.exe" %PYTHON_CMD% -m venv .venv
if errorlevel 1 goto venv_error
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel "setuptools<81"
if errorlevel 1 goto install_error
nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu_torch
".venv\Scripts\python.exe" -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
if not errorlevel 1 goto install_project
echo Не удалось установить CUDA-сборку PyTorch. Выполняется установка CPU-сборки.
:cpu_torch
".venv\Scripts\python.exe" -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto install_error
:install_project
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto install_error
".venv\Scripts\python.exe" -W "ignore:pkg_resources is deprecated as an API:UserWarning" -c "import pygame, psutil, pynvml, torch, tetris_ai; print('Tetris AI', tetris_ai.__version__); print('PyTorch', torch.__version__); print('CUDA', torch.cuda.is_available())"
if errorlevel 1 goto import_error
echo.
echo Установка Tetris AI 1.2.2 rc успешно завершена.
echo Теперь запустите файл "Запустить Tetris AI.bat".
pause
exit /b 0
:python_error
echo Не найден Python 3.10 или новее. Установите 64-bit Python и повторите запуск.
pause
exit /b 1
:try_launcher
if defined PYTHON_CMD exit /b 0
py -%1 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -%1"
exit /b 0
:try_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=python"
exit /b 0
:venv_error
echo Не удалось создать локальное окружение .venv.
pause
exit /b 1
:install_error
echo Не удалось установить зависимости. Проверьте интернет и файл .runtime\logs при наличии.
pause
exit /b 1
:import_error
echo Зависимости установлены, но проверка импортов завершилась ошибкой.
pause
exit /b 1
