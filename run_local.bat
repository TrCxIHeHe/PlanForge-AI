@echo off
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Follow the setup steps in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
