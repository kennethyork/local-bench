@echo off
rem Free the GPU: unload every Ollama model and stop anything LocalBench left behind.
rem Don't run it while a benchmark is running (that would unload the model being tested).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m localbench unload
pause
