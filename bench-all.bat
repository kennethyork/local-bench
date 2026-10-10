@echo off
rem Benchmark every local Ollama model from ollama.com that's compatible with this GPU,
rem then refresh the GitHub Pages copy (docs\).
rem
rem   bench-all.bat              the ollama.com library: pull -> test -> delete, one model at a time
rem                              (models you already have are reused, never deleted)
rem   bench-all.bat mine         only models already installed (e.g. custom ones not on ollama.com)
rem   bench-all.bat both         installed models first, then the library
rem
rem Resumable: close the window any time and run it again; finished models are skipped.
rem Compatible models that don't fit right now because other apps hold VRAM are deferred, not dropped.
rem Options pass through, e.g.:
rem   bench-all.bat --governor balanced          (faster, warmer)
rem   bench-all.bat --dry-run                    (show the plan only)
rem   bench-all.bat --family qwen3,gemma3        (only some families)
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set "PY=.venv\Scripts\python.exe"

set "MODE=library"
if /i "%~1"=="mine" set "MODE=mine"
if /i "%~1"=="both" set "MODE=both"
if /i "%~1"=="library" set "MODE=library"
set "ARGS="
if /i "%~1"=="mine" (call :drop_first %*) else if /i "%~1"=="both" (call :drop_first %*) else if /i "%~1"=="library" (call :drop_first %*) else (set "ARGS=%*")

if not "%MODE%"=="library" (
  echo.
  echo ===== Installed models =====
  "%PY%" -m localbench run --publish %ARGS%
)
if not "%MODE%"=="mine" (
  echo.
  echo ===== ollama.com library =====
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
