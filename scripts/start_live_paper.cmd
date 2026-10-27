@echo off
REM Project Atlas - live paper trading (simulated orders only; prices from Alpaca IEX).
REM Double-click, or run from a terminal. Runs the preflight, then today's session, and opens
REM the live leaderboard page. Re-running on the same day resumes the session.
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo Python environment not found. See README "Setup".
  pause
  exit /b 1
)
echo === Preflight ===
"%PY%" -m atlas.cli --log-level ERROR paper-check --config configs\paper_live.yaml
if errorlevel 1 (
  echo.
  echo Preflight failed - fix the FAIL lines above, then run this again.
  pause
  exit /b 1
)
echo.
echo === Live paper session (Ctrl+C to stop cleanly) ===
"%PY%" -m atlas.cli --log-level WARNING paper --config configs\paper_live.yaml --open-dashboard --auto-resume
echo.
echo Session ended. Daily report: artifacts\paper\live_^<date^>\daily_report.md
echo Leaderboard: "%PY%" -m atlas.cli paper-status --config configs\paper_live.yaml
pause
