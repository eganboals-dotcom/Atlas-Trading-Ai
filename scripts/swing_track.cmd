@echo off
REM Project Atlas - swing forward tracker (research only; places no orders).
REM Double-click after 4:20 pm ET (1:20 pm Pacific) on any trading day. Downloads daily bars (read-only,
REM about 1.5 minutes) and scores the frozen rules in configs\swing_track.yaml against Hold since 2026-10-29.
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo Python environment not found. See README "Setup".
  pause
  exit /b 1
)
"%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track.yaml
echo.
echo Report: artifacts\reports\swing_track\tracker.md
pause
