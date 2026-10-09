@echo off
rem Benchmark EVERY Ollama library model that fits this GPU: pull -> test -> delete -> next.
rem Resumable: close the window any time, double-click again to continue where it stopped.
rem Your own models are never deleted. Progress: results\library_state.json
rem Options pass through, e.g.:
rem   bench-library.bat --governor balanced      (faster, warmer)
rem   bench-library.bat --dry-run                (just show the plan)
rem   bench-library.bat --family qwen3,gemma3    (only some families)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
".venv\Scripts\python.exe" -m localbench library --publish %*
echo.
echo Stopped or finished. Dashboard: results\report.html   GitHub Pages copy: docs\index.html
pause
