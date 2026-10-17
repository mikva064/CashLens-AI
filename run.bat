@echo off
rem Запуск CashLens AI из любой папки: переходим в папку скрипта.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Создаю окружение .venv на Python 3.12 ...
  py -3.12 -m venv .venv || goto :error
  .venv\Scripts\python -m pip install -e ".[data,service]" || goto :error
)
if "%CASHLENS_ANALYZER%"=="" set CASHLENS_ANALYZER=mock
echo Анализатор: %CASHLENS_ANALYZER%
echo Открой http://localhost:8000
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
goto :eof
:error
echo Ошибка установки. Нужен Python 3.12: py install 3.12
pause
