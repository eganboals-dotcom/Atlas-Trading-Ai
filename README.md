# Project Atlas

A research platform that tests trading and investing rules honestly, on days they never saw, and tracks the few that
survive forward on real prices with pretend money. **Nothing here trades real money, and there is no code that can.**

Atlas began as a 50-agent evolutionary day-trading AI. Those bots lost to trading costs, so the work moved to slower
investing rules, each tested against written-down rules before its result was seen. Five rules are now frozen and
scored every night on new market days. The day-trading bots still run each weekday on live prices as a working
system, not as evidence of an edge. How the platform was built, and the full write-up of every test, is in
[docs/HISTORY.md](docs/HISTORY.md).

> **Safety:** no broker integration and no order-placing code exist. Only five read-only modules may use the network
> (Alpaca market data, a local AI model, the optional Claude reader, SEC EDGAR and FINRA), and
> `tests/test_paper_cannot_trade_live.py` enforces it.

## Where it stands (November 2026)

- **No rule has beaten the market convincingly.** 38 numbered tests on 2019 to 2026 history. A few beat holding the
  same companies, but none reliably beat SPY, the index anyone can buy.
- **Five rules are frozen and tracked forward.** Their verdicts come after about five years of new days, around 2031.
  Until then their numbers say almost nothing.
- **The investing bots** show each frozen rule as an account of $100,000 of pretend money next to the same money in SPY.
- **The day-trading bots** trade a $100,000 paper fund on live Alpaca prices every weekday. They were evolved on
  synthetic data and lose about what they pay in spreads.

## What it found

**[The write-up: Why beating the market is so hard](docs/report/README.md)**

- **Backtests flatter.** Of 809,823 bots tried in 39 walk-forward experiments, the best bot did worse on unseen days
  than in training in 36 of them.
- **Costs eat short-term trading.** On live prices the trading bots lost almost exactly what they paid in spreads.
- **Most textbook strategies lose to simply holding.** The biggest winners held three stocks at a time and fell over
  50% at worst; the cautious versions of the ideas that passed pre-registered tests beat holding by under a point a
  year, an edge that takes decades of data to tell apart from luck.
- **Machine learning didn't change that.** In v37, eight forecasters (ARIMA, a random forest, XGBoost, an LSTM, and
  the foundation models TimesFM 2.5 and TiRex-2) all failed, and none beat SPY. The foundation models did worse than
  holding.
- **The one new pattern was risk, not edge.** Last week's losers beat last week's winners by 16 points a year (v38),
  but swung half again as much as holding, lost to SPY, and fell below holding at 10 bps a trade.

[![Training scores vs unseen days](docs/report/figures/1_backtest_vs_unseen.png)](docs/report/README.md)

## The frozen rules

Each rule's code and settings are fingerprinted, and its tracker refuses to run if either changes. The verdict for
each is written in its config before any result. Backtest figures are on history from 2019-01-24 to 2026-01-29, after
costs, against SPY's +16.6% a year.

| Tracker | The rule | Backtest a year | Scored from | Verdict after |
|---|---|---|---|---|
| v21 Fed drift | Hold SPY in the hours before each Fed decision | an event rule | 2026-11-02 | 40 decisions |
| v24 Earnings quality | The fifth of 300 large companies whose profit is best backed by cash, reset every 63 sessions | +23.1% (survivors only; +24.3% with failed companies added) | 2026-11-02 | 1,260 sessions |
| v30 Rotation | The strongest groups of stocks and ETFs (growth, real assets, defensive), in four weekly sleeves | +30.7% for the monthly form; the tracked form never ran on history | 2026-11-02 | 1,260 sessions |
| v32 ETF rotation | The same group choice with ETFs only | +17.1%, an estimate made after freezing | 2026-11-02 | 1,260 sessions |
| v36 ETFs held equally | The rotation's 13 ETFs in equal amounts, reset each quarter; it claims steadier returns than SPY, not more | +15.9%, an estimate seen before freezing | 2026-11-06 | 1,260 sessions |

Most of these figures come from lists of companies that survived to 2026, which flatters every rule; v31 rebuilt the
list with the companies that failed, and earnings quality still beat holding there but passed only 4 of 6 tests. The configs are
[fomc_track_v21](configs/fomc_track_v21.yaml), [quality_track_v24](configs/quality_track_v24.yaml),
[rotation_track_v30](configs/rotation_track_v30.yaml), [etf_rotation_track_v32](configs/etf_rotation_track_v32.yaml) and
[etf_equal_track_v36](configs/etf_equal_track_v36.yaml).

Beside them, the swing trackers score earlier rules the same way each night: quality, momentum, trend, earnings
surprises, short interest and the news desk, whose readers (fixed keyword rules and a free AI model on this PC) turn
headlines into a briefing before each open.

## The two kinds of bots

**Investing bots.** `artifacts/reports/summary/latest.md`, written by `scripts/trackers/tracker_summary.py`, is the one
page for everything: each frozen rule as a $100,000 account beside the same money in SPY, what each holds, every
tracked rule against Hold and SPY, and anything that needs a person to look. The leaderboard window's Investing bots tab
shows the same, with a chart since the start and, during market hours, an estimate from the latest trades.

**Day-trading bots.** Ten bots share a $100,000 paper fund, $10,000 each to start, and each one's capital compounds
with its own results in `artifacts/paper/fund_ledger.json`. Six are evolved (Liquidity, Regime, MomentumEvo, TrendFollow,
Pullback, SmartHold, frozen in [bots/live_lineup_v2.json](bots/live_lineup_v2.json) and explained in
[bots/README.md](bots/README.md)), three are textbook baselines (Momentum, MeanRevert, Random), and Flip, a coin flip,
is permanent. Hold, the buy-and-hold benchmark, runs beside them with a fresh share each day. Prices are Alpaca's free
real-time IEX quotes and trades, polled; every order and fill is simulated by Atlas's own paper broker, and positions
are sold at the bid at the close. Each day's report is `artifacts/paper/live_<date>/daily_report.md`, and the
leaderboard window shows the session live.

## The daily routine

The Windows scheduled task "Atlas daily research" runs at 5:35 am Pacific:

1. The PC wakes from sleep, or is started by the motherboard's 5:30 alarm.
2. On weekdays it starts the paper session, which waits for the 6:30 am open and ends at 1:00 pm. The leaderboard
   window opens at 5:40 am and at sign-in.
3. It runs the nightly research (`scripts/swing_track.cmd auto`): new prices, the news desk, every tracker, then the
   summary page.
4. When the session ends it puts the PC back to sleep, unless someone is using it.

Logs are in `artifacts/reports/swing_track/runs.log`, with `paper_morning.log` beside it.

## The research log

Each test since v13 ran once, with its pass tests written in its config before its result was seen; v26, v28 and v34
measure something rather than pass or fail a rule. A failed test stays failed. The full write-ups are in [docs/HISTORY.md](docs/HISTORY.md) and the scripts are mapped in
[scripts/README.md](scripts/README.md).

| Test | The idea | Result |
|---|---|---|
| v1 to v11 | Evolved and PPO day-trading bots, then swing rules on ETFs and stock lists | Bots lost to costs; quality, earnings surprises and low short interest passed modestly and are tracked |
| v12 | Earnings surprises read from headlines | Passed, but three stocks drove most of it; tracked |
| v13 to v19 | Opening fade, stocks-in-play breakout, Noise Area, late-day flow, low turnover, the holdout, pairs | All failed |
| v20 | Seasonality, trend, the pre-Fed drift | Only the drift passed; frozen as v21 |
| v22 | Bottlenecks in growing sectors | Failed 3 of 5 |
| v23 | Four rules from SEC filings | Earnings quality passed; frozen as v24 |
| v25 | Earnings quality on newer companies | Failed 5 of 7 |
| v26 | How much survives the number of rules tried | Under the bar |
| v27 | FinRL's Adaptive Rotation | Failed 3 of 6 |
| v28 | What a trade really costs | About 5 bps a side for stocks at the open, 1 bp for ETFs |
| v29 | The rotation, monthly | Passed 5 of 6; frozen as v30 |
| v31 | The list with the failed companies in it | Earnings quality 4 of 6; rotation 2 of 2 |
| v33 | The rotation against chance | Ranking failed; choosing groups passed |
| v34 | The ETF-only rotation on history, after freezing | +17.1% a year vs SPY's +16.4%, mostly 2025 |
| v35 | Annual reports that barely changed ("Lazy Prices") | Failed 3 of 6 |
| v37 | Eight price forecasters, naive to foundation models | All failed; none beat SPY |
| v38 | Last week's losers, with the failed companies in | Failed 4 of 7 |

v26 counted 25 rules tried; with at least 14 more since, the bar for telling a result from luck is a little higher than
it shows.

## Setup

Python 3.12 on Windows, in a `.venv` at the repo root:

```bash
py -3.12 -m venv .venv
.venv/Scripts/activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
```

Keys live only in `.env`, which is git-ignored and copied from `.env.example`:

| Key | For |
|---|---|
| `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | Alpaca market data, from a free paper-trading account |
| `SEC_USER_AGENT` | The contact the SEC requires on every request, as `<name> <email>` |
| `ANTHROPIC_API_KEY` | Optional: the Claude news reader, which uses paid credits and is off by default |

The local news reader needs [Ollama](https://ollama.com) with the model named in
[configs/news_desk.yaml](configs/news_desk.yaml); without it that reader is skipped and catches up later. The paper
session also needs a synced PC clock: it refuses one more than 3 s behind exchange time.

## Commands

All run from the repo root with `.venv/Scripts/python.exe`. The paper session, started by double-clicking
`scripts/start_live_paper.cmd` or by the scheduled task:

```bash
python -m atlas.cli paper-check --config configs/paper_live.yaml                  # preflight, trades nothing
python -m atlas.cli paper --config configs/paper_live.yaml --auto-resume          # the session, about 6.5 hours
python -m atlas.cli paper-status --config configs/paper_live.yaml                 # one-shot leaderboard
python -m atlas.cli paper-window --config configs/paper_live.yaml                 # reopen the leaderboard window
python -m atlas.cli paper-stop --config configs/paper_live.yaml                   # clean, resumable stop
python -m atlas.cli paper-kill --config configs/paper_live.yaml --reason "..."    # emergency halt only
```

The nightly research and the one-page summary:

```bash
scripts/swing_track.cmd
python scripts/trackers/tracker_summary.py
```

Each research test is one script with its config, for example
`python scripts/research/investing/v38_reversal.py run` with
[configs/reversal_v38.yaml](configs/reversal_v38.yaml). The tests, about 40 s:

```bash
python -m pytest
```

The original platform's commands (synthetic data, evolution, walk-forward, PPO, the readiness review) still work and
are described in [docs/HISTORY.md](docs/HISTORY.md).

## Rules the code keeps

- **Paper only.** Configs accept only `mode: paper` with live orders off, and the only broker is the simulated one. A
  live-order path would need separate, explicit approval.
- **Five network modules,** each read-only and pinned to one job: `src/atlas/feeds/alpaca.py`, `local_llm.py`
  (loopback only), `anthropic_api.py`, `sec.py` and `finra.py`.
- **Frozen means frozen.** A tracker refuses to run if its rule's fingerprint changes.
- **Flip and Hold are permanent** in every paper session, as the yardsticks.
- **Recorded data stays local.** `data/` and `artifacts/` are git-ignored; Alpaca's data terms apply.
- **Paper P&L isn't evidence.** The trading bots were evolved on synthetic data.

## Known limitations

- **Survivors.** Most tests use 300 companies that all survived to 2026. Only v31 and v38 include the ones that failed.
- **One window, many tries.** About 40 rules have been run on the same 2019 to 2026 days, so one passing by luck is
  likely; only the frozen trackers' new days are clean.
- **Polled IEX data.** IEX shows only its own book, usually wider than the national best price, so paper fills are
  conservative approximations, not executions.
- **No holiday calendar.** On a market holiday the session sees no fresh data and stays in its safe state.
- **Few days so far.** The trackers have days, not years; nothing before a verdict is a result.

## Repository layout

```
configs/           every rule and test's settings, one YAML each; paper_live.yaml and risk_live.yaml run the session
src/atlas/
  feeds/           the only network modules, all read-only
  paper/           paper config, live feed, simulated broker, risk manager, session, report
  swing/           daily bars, backtests, rule families, fundamentals from SEC filings
  news/            the news desk: headlines, readers, the briefing
  dashboard/       the leaderboard window and the optional browser page, read-only
  agents/, evolution/, evaluation/, learning/, environment/, data/   the original platform
bots/              the frozen day-trading lineup
scripts/           launchers and the scheduled task's scripts; trackers/ (frozen rules); research/ (one-off tests)
tests/             pytest suites, about 480 tests
docs/              the write-up (report/), HISTORY.md, the original specification, the readiness review
data/, artifacts/  local data and generated reports (git-ignored)
```

The original specification is [docs/PROJECT_SPECIFICATION.md](docs/PROJECT_SPECIFICATION.md), and the readiness review,
whose verdict is NOT READY, is [docs/READINESS_REVIEW.md](docs/READINESS_REVIEW.md).
