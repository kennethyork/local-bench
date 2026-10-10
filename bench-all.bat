@echo off
rem Benchmark everything that fits this GPU, then refresh the GitHub Pages copy (docs\).
rem
rem   bench-all.bat              your installed models first, then the whole Ollama library
rem   bench-all.bat mine         only your installed models (Ollama + Hugging Face cache)
rem   bench-all.bat library      only the Ollama library (pull -> test -> delete, one at a time)
rem
rem Resumable: close the window any time and run it again; finished models are skipped.
rem Your own models are never deleted. Other options pass through, e.g.:
rem   bench-all.bat --governor balanced          (faster, warmer)
rem   bench-all.bat library --dry-run            (show the library plan only)
rem   bench-all.bat library --family qwen3,gemma3
rem Library-only options (--dry-run, --family, --max-models, --keep, --retry-failed, --refresh)
rem need the "library" mode.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set "PY=.venv\Scripts\python.exe"

set "MODE=all"
if /i "%~1"=="mine" set "MODE=mine"
if /i "%~1"=="library" set "MODE=library"
set "ARGS="
if "%MODE%"=="all" (set "ARGS=%*") else (call :drop_first %*)

if not "%MODE%"=="library" (
  echo.
  echo ===== Step 1: your installed models =====
  "%PY%" -m localbench run --publish %ARGS%
)
if not "%MODE%"=="mine" (
  echo.
  echo ===== Step 2: the Ollama library =====
  "%PY%" -m localbench library --publish %ARGS%
)

echo.
echo Done or stopped. Dashboard: results\report.html   GitHub Pages copy: docs\index.html
pause
exit /b

:drop_first
shift
:collect
if "%~1"=="" goto :eof
set "ARGS=%ARGS% %1"
shift
goto collect
