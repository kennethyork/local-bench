@echo off
rem Benchmark every local Ollama model from ollama.com that's compatible with this GPU,
rem then refresh the GitHub Pages copy (docs\).
rem
rem   bench-all.bat              continuous benchmarking of the ollama.com library: pull -> test -> delete,
rem                              one model at a time (models you already have are reused, never deleted).
rem                              When it's caught up it re-checks ollama.com every 6 hours for new models
rem                              and keeps going; close the window to stop.
rem   bench-all.bat once         one pass through the library, then stop
rem   bench-all.bat mine         only models already installed (e.g. custom ones not on ollama.com)
rem   bench-all.bat both         installed models first, then the library
rem
rem Resumable: close the window any time and run it again; finished models are skipped.
rem Compatible models that don't fit right now because other apps hold VRAM are deferred, not dropped.
rem Options pass through, e.g.:
rem   bench-all.bat --governor quiet             (fans kept under 55%%, slower)
rem   bench-all.bat --dry-run                    (show the plan only)
rem   bench-all.bat --family qwen3,gemma3        (only some families)
setlocal
rem The governor's watchdog changes the GPU power limit, which Windows only allows as administrator.
net session >nul 2>&1
if errorlevel 1 (
  echo Asking for administrator rights so the watchdog can control the GPU power limit...
  powershell -NoProfile -Command "$a = '%*'; try { if ($a) { Start-Process -FilePath '%~f0' -ArgumentList $a -Verb RunAs -ErrorAction Stop } else { Start-Process -FilePath '%~f0' -Verb RunAs -ErrorAction Stop }; exit 0 } catch { exit 1 }"
  if not errorlevel 1 exit /b
  echo Not elevated - running anyway, without the watchdog. Only the 80C hard stop protects the GPU.
)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set "PY=.venv\Scripts\python.exe"

set "MODE=library"
if /i "%~1"=="mine" set "MODE=mine"
if /i "%~1"=="both" set "MODE=both"
if /i "%~1"=="library" set "MODE=library"
if /i "%~1"=="once" set "MODE=once"
set "ARGS="
if /i "%~1"=="mine" (call :drop_first %*) else if /i "%~1"=="both" (call :drop_first %*) else if /i "%~1"=="library" (call :drop_first %*) else if /i "%~1"=="once" (call :drop_first %*) else (set "ARGS=%*")

set "FOREVER=--forever"
if "%MODE%"=="once" (set "MODE=library" & set "FOREVER=")

if not "%MODE%"=="library" (
  echo.
  echo ===== Installed models =====
  "%PY%" -m localbench run --publish %ARGS%
)
if not "%MODE%"=="mine" (
  echo.
  echo ===== ollama.com library =====
  "%PY%" -m localbench library --publish %FOREVER% %ARGS%
)

echo.
echo ===== Freeing the GPU =====
"%PY%" -m localbench unload
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
