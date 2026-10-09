@echo off
rem Benchmark every local model that fits this GPU, then refresh the GitHub Pages copy.
rem Double-click to run. Extra options pass straight through, e.g.:
rem   bench-all.bat --governor balanced
rem   bench-all.bat --redo
rem Already-benchmarked models are skipped, so rerunning only picks up new models.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
".venv\Scripts\python.exe" -m localbench run --publish %*
echo.
echo Done. Dashboard: results\report.html   GitHub Pages copy: docs\index.html
pause
