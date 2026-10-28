@echo off
REM Project Atlas - swing forward tracker (research only; places no orders).
REM Double-click after 4:20 pm ET (1:20 pm Pacific) on any trading day. Downloads daily bars (read-only,
REM about 1.5 minutes) and scores the frozen rules in configs\swing_track.yaml against Hold since 2026-10-29.
REM "swing_track.cmd auto" is the unattended form (for a scheduled task): no pause, output appended to
REM artifacts\reports\swing_track\runs.log.
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo Python environment not found. See README "Setup".
  if /i not "%~1"=="auto" pause
  exit /b 1
)
if /i "%~1"=="auto" (
  if not exist artifacts\reports\swing_track mkdir artifacts\reports\swing_track
  echo === %DATE% %TIME% >> artifacts\reports\swing_track\runs.log
  "%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track.yaml >> artifacts\reports\swing_track\runs.log 2>&1
  exit /b %ERRORLEVEL%
)
"%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track.yaml
echo.
echo Report: artifacts\reports\swing_track\tracker.md
pause
