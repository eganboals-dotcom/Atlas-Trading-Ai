# CLAUDE.md — operating Project Atlas

Atlas is a research and **paper-trading** platform for evolved trading agents. The full spec is
`docs/PROJECT_SPECIFICATION.docx`; the status and how-tos are in `README.md`; the latest readiness
review is `docs/READINESS_REVIEW.md` (verdict: NOT READY).

## Hard rules

- **No live trading.** There is no live-order path, and none may be added without the user's explicit,
  separate approval for Phase 7/8. Never add trading endpoints, broker SDKs, order-placing code, or a
  network client outside `src/atlas/feeds/alpaca.py`. The tests in `tests/test_paper_cannot_trade_live.py`
  enforce this; never weaken them to make something pass.
- **Keys stay in `.env`.** Alpaca market-data keys live only in `.env` (git-ignored). Never ask the user to
  paste keys into chat, never print, `cat`, or log `.env`, and never commit it. If keys are missing, tell
  the user to add them to `.env` themselves.
- **Don't change a running session.** Never edit `configs/paper_live.yaml`, `configs/risk_live.yaml`, or
  checkpoints while a session runs. Risk limits are the user's to change.
- **Permanent bots.** `Flip` (coin flip) and `Hold` (buy-and-hold benchmark) are in every paper session
  (`PERMANENT_BASELINES` in `src/atlas/paper/session.py`). Never remove them; they are the yardsticks
  every strategy is judged against.
- **Recorded data stays local.** Live sessions save market data under `data/` (git-ignored). Never commit it;
  Alpaca's data terms apply. It is the real-market dataset for training new bots (`ParquetProvider` on
  `data/processed/alpaca_iex_live`).
- **Personal info.** Never commit `.env`, keys, account numbers, local paths, or the user's email. Commits use the
  GitHub noreply address set in this repo's git config.
- **Frozen bots.** Bots don't train during paper sessions. Training happens offline (`evolve`,
  `walkforward`, `train-ppo`).
- **Fund.** Each bot trades only its share of `fund.total_capital`. Its capital compounds day to day through
  `artifacts/paper/fund_ledger.json`. Never edit the ledger by hand. Use `--reset-fund` only when the user asks
  for a fresh split; it archives the old ledger.
- **Honest results.** The bots were evolved on synthetic data. Don't describe paper P&L as evidence of
  an edge.

## Environment

Windows, Python 3.12 venv at `.venv`. Always run from the repo root with `.venv/Scripts/python.exe`.

```bash
.venv/Scripts/python.exe -m pytest            # full suite, about 25 s
```

## Live paper trading (Alpaca prices, simulated orders)

A session runs from 09:30 to 16:00 ET, about 6.5 hours. That is far longer than a foreground tool call.

1. **Preflight.** This is safe at any time; it checks keys, clock, bots, and the page, and does not trade.
   `.venv/Scripts/python.exe -m atlas.cli paper-check --config configs/paper_live.yaml`
2. **Start.** The user's preferred way is double-clicking `scripts/start_live_paper.cmd`, which runs in its
   own window. If the user asks you to start it, run it as a **background** process so it is not killed by a
   tool timeout:
   `.venv/Scripts/python.exe -m atlas.cli --log-level WARNING paper --config configs/paper_live.yaml --auto-resume`
   It waits for the open by itself and opens the live leaderboard window (a native, read-only app window).
   Reopen it any time with `paper-window --config configs/paper_live.yaml`.
3. **Monitor.** This is read-only. Use `paper-status` for a one-shot leaderboard; `paper-watch` is the live,
   refreshing view, and is what the leaderboard window runs:
   `.venv/Scripts/python.exe -m atlas.cli paper-status --config configs/paper_live.yaml`
   The raw snapshot is `artifacts/paper/live_<YYYY-MM-DD>/live.json`, and alerts are in `alerts.jsonl` there.
4. **Stop cleanly.** This is resumable; re-running step 2 the same day continues the session:
   `.venv/Scripts/python.exe -m atlas.cli paper-stop --config configs/paper_live.yaml`
5. **Emergency halt.** Only when the user asks, or on clear malfunction. It halts trading, and the user
   must review before restarting:
   `.venv/Scripts/python.exe -m atlas.cli paper-kill --config configs/paper_live.yaml --reason "<why>"`
6. **After the close.** Read `artifacts/paper/live_<date>/daily_report.md`. Reopen the leaderboard window
   with `paper-watch --config configs/paper_live.yaml`; the optional browser page opens with
   `.venv/Scripts/python.exe -m atlas.cli dashboard`.

### If the preflight or start fails

- *"set APCA_API_KEY_ID and APCA_API_SECRET_KEY"*: the user must add their Alpaca **paper-account** keys to
  `.env`.
- *"Alpaca rejected the market-data credentials"*: the keys are wrong or revoked; the user regenerates them.
- *"local clock is … behind"*: Windows Settings → Time & language → Date & time → Sync now.
- *"already exists; pass --resume"*: use `--auto-resume`, which the launcher already does.
- *"kill switch engaged"*: the user reviews why (`alerts.jsonl`), then restarts with `--reset-kill-switch`.
- *Weekend/holiday*: the market is closed. Holidays are not in the calendar, so on a holiday the session
  just sees no fresh data and stays in safe state.
- *Port 8501 busy* (browser page only): an earlier page is still running; open http://localhost:8501.
- *"already running (process N)"*: today's session is already running in another window. Use that window; don't
  start a second one.
- *Safe state "feed disconnected"*: normal during network blips. The session retries with backoff for about
  11 minutes before the kill switch engages. Check the user's internet connection.
