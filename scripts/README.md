# Scripts

Everything runs from the repo root with `.venv/Scripts/python.exe`. Research only: no script places an order.

## At the top of this folder

These stay here because the Windows scheduled task and the desktop shortcut point at them.

| File | What it does |
| --- | --- |
| `start_live_paper.cmd`, `start_live_paper.ps1` | Start the paper trading session |
| `swing_track.cmd` | The daily research run, every tracker in order |
| `nightly_wake.ps1` | The scheduled task's job: runs `swing_track.cmd auto`, with `-Trade` also the day's paper session, then shuts down |
| `open_leaderboard.ps1` | Opens the leaderboard for that session, at sign-in and at 5:40 am |
| `set_morning_task.ps1` | Moves the scheduled task to 5:35 am with `-Trade` and `-After sleep`; run once as administrator |

## trackers/

Frozen rules, run every night. Their rules never change. Each one refuses to run if its rule does.

| File | What it tracks |
| --- | --- |
| `v21_fomc_track.py` | The drift before Fed decisions |
| `v24_quality_track.py` | Earnings quality |
| `v30_rotation_track.py` | The stock and ETF rotation |
| `v32_etf_rotation_track.py` | The ETF-only rotation |
| `v36_etf_equal_track.py` | The ETF rotation's 13 ETFs held equally |
| `tracker_summary.py` | One page across all of them, with the investing bots; also writes `bots.json` for the leaderboard's Investing bots tab |

## research/trading/

One-off tests of shorter-term ideas. Each ran once, and its result is in the README.

| File | Test | Result |
| --- | --- | --- |
| `v12_controls.py`, `v12_robustness.py` | Controls and break attempts for the v12 rules | See README |
| `v13_open_fade.py` | Fading the opening 30 minutes | Failed |
| `v14_stocks_in_play.py` | Stocks-in-play breakout | Failed |
| `v15_noise_area.py`, `v15_followup.py`, `v15_overlay.py`, `v15b_noise_area_breadth.py` | Noise Area breakout | Failed; the overlay was the one lead |
| `v16_late_day_flow.py` | Late-day flow | Failed |
| `v17_low_turnover.py` | Lower-turnover versions | Failed |
| `v18_holdout.py` | The overlay on the locked months | Failed |
| `v19_pairs.py` | Pairs trading | Failed |
| `v20_others.py`, `v20_followup.py` | Seasonality, trend, and the pre-Fed drift | Only the drift passed; it became v21 |

## research/investing/

One-off tests of slower ideas.

| File | Test | Result |
| --- | --- | --- |
| `v22_bottleneck.py` | Bottlenecks in growing sectors | Failed 3 of 5 |
| `v23_filings.py`, `v23_followup.py` | Four rules from SEC filings | Earnings quality passed; it became v24 |
| `v25_new_companies.py` | Earnings quality on new companies | Failed 5 of 7 |
| `v26_deflated_sharpe.py` | Allowing for how many rules were tried | Under the bar |
| `v27_finrl_rotation.py` | FinRL's Adaptive Rotation | Failed 3 of 6 |
| `v28_trading_costs.py` | What a trade really costs | A measurement |
| `v29_monthly_rotation.py`, `v29_followup.py` | The rotation, monthly | Passed 5 of 6; it became v30 |
| `v31_point_in_time.py` | The list with the failed companies in it | Earnings quality 4 of 6; rotation 2 of 2 |
| `v33_random_picks.py` | The rotation against chance | Ranking failed; choosing groups passed |
| `v34_etf_rotation_history.py` | The ETF-only rotation on history, after it was frozen | An estimate: +17.1% a year vs SPY's +16.4%, mostly 2025 |
| `v35_lazy_prices.py` | Annual reports that barely changed ("Lazy Prices") | Failed 3 of 6: right direction, too small, lost to SPY |
| `v37_forecasters.py` | Eight price forecasters: naive, ARIMA, random forest, XGBoost, LSTM, TimesFM 2.5, TiRex-2 | All failed; the best passed 3 of 6, none beat SPY |
| `v38_reversal.py` | Last week's losers, on v31's list with the failed companies in | Failed 4 of 7: more return, but as much more risk; lost to SPY and to costs |

`research/build_report.py` draws the charts for the write-up in `docs/report/`.

Scripts in different folders import each other, so a few of them add one `sys.path` line near the top. The trackers import from `research/investing/` and `research/investing/` imports from `research/trading/`, so moving one of those files means updating that line and `swing_track.cmd`.
