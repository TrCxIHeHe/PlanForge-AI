@echo off
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Follow the setup steps in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest -q
