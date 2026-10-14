@echo off
REM kayEi AmpStudio from source (Windows): creates a local venv on first run.
REM For end users: use the portable zip from the GitHub releases instead.
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install -q -r requirements.txt
)
.venv\Scripts\python -m amp_qt.app %*
