@echo off
rem Запуск CashLens AI из любой папки: переходим в папку скрипта.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Создаю окружение .venv ...
  python -m venv .venv || goto :error
  .venv\Scripts\python -m pip install -r requirements.txt || goto :error
)
echo Открой http://localhost:8000
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
goto :eof
:error
echo Ошибка установки. Проверь, что Python 3.11+ установлен и доступен как "python".
pause
