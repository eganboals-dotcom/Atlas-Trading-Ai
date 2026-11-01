@echo off
REM Project Atlas - daily research after the close (research only; places no orders).
REM Double-click after 4:20 pm ET (1:20 pm Pacific) on any trading day. Runs the news desk
REM (configs\news_desk.yaml: new headlines, reader notes, next briefing), then the forward trackers:
REM the frozen swing rules (configs\swing_track.yaml; downloads daily bars, about 1.5 minutes) and the
REM news bots (configs\swing_track_news.yaml) and the gross-profitability rule
REM (configs\swing_track_fund.yaml; bars and SEC filings, about 4 minutes), and the 300-company
REM rules (configs\swing_track_u300.yaml; about 15 minutes) and the earnings and short-interest
REM signals (configs\swing_track_signals.yaml; FINRA files only) and the v12 rules
REM (configs\swing_track_v12.yaml; new headlines for the 300), each scored against
REM Hold since 2026-10-29.
REM Last, the pre-FOMC drift tracker (configs\fomc_track_v21.yaml; SPY minute bars, under a minute) and the
REM earnings-quality tracker (configs\quality_track_v24.yaml; the 300-company bars above, and SEC filings each quarter) and the
REM rotation tracker (configs\rotation_track_v30.yaml; the 300-company bars above, and its ETFs, under a minute) and the
REM ETF-only rotation (configs\etf_rotation_track_v32.yaml; the same ETFs). Then one page across all of them.
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
  "%PY%" -m atlas.cli --log-level WARNING news-desk --config configs\news_desk.yaml >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track.yaml >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_news.yaml --no-fetch >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_fund.yaml >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_u300.yaml >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_signals.yaml >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_v12.yaml >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" scripts\v21_fomc_track.py >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" scripts\v24_quality_track.py >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" scripts\v30_rotation_track.py >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" scripts\v32_etf_rotation_track.py >> artifacts\reports\swing_track\runs.log 2>&1
  "%PY%" scripts\tracker_summary.py >> artifacts\reports\swing_track\runs.log 2>&1
  exit /b %ERRORLEVEL%
)
"%PY%" -m atlas.cli --log-level WARNING news-desk --config configs\news_desk.yaml
"%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track.yaml
"%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_news.yaml --no-fetch
"%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_fund.yaml
"%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_u300.yaml
"%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_signals.yaml
"%PY%" -m atlas.cli --log-level WARNING swing-track --config configs\swing_track_v12.yaml
"%PY%" scripts\v21_fomc_track.py
"%PY%" scripts\v24_quality_track.py
"%PY%" scripts\v30_rotation_track.py
"%PY%" scripts\v32_etf_rotation_track.py
"%PY%" scripts\tracker_summary.py
echo.
echo Reports: artifacts\reports\swing_track\tracker.md, artifacts\reports\swing_track_news\tracker.md
echo          artifacts\reports\swing_track_fund\tracker.md, artifacts\reports\swing_track_u300\tracker.md
echo          artifacts\reports\swing_track_signals\tracker.md, artifacts\reports\swing_track_v12\tracker.md
echo          artifacts\reports\fomc_track\tracker.md, artifacts\reports\quality_track\tracker.md
echo          artifacts\reports\rotation_track\tracker.md, artifacts\reports\etf_rotation_track\tracker.md
echo          artifacts\reports\summary\latest.md (all of them on one page)
echo          and the next briefing in artifacts\reports\news_desk\latest.md
pause
