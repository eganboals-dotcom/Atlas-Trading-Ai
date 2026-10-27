# Project Atlas

50-agent evolutionary trading AI with genetic optimization, reinforcement learning, and realistic market simulation.

Atlas is a research platform for evolving and evaluating algorithmic trading agents with
genetic selection, reinforcement learning, event-driven market simulation, and strict risk
controls. Strategies are tested on historical data and live-data paper trading before any
separately approved live deployment. **Profitability is not assumed; it must be demonstrated
with robust out-of-sample evidence.**

The full specification is in [docs/PROJECT_SPECIFICATION.docx](docs/PROJECT_SPECIFICATION.docx).

> **Safety:** this codebase contains no broker integration and no live-order code. Live
> trading is out of scope until a separately approved Phase 8.

## Status: Phase 6 — readiness review

| Phase | Scope | Status |
|---|---|---|
| 1 | Events, synthetic data, validation, deterministic replay, costs/latency, ledger | Done |
| 2 | Agent interface, 50 non-RL agents, baselines, evolution, lineage | Done |
| 3 | Historical data, walk-forward evaluation, metrics | Done |
| 4 | PPO policy family | Done |
| 5 | Live-data paper trading, risk manager, dashboard | Done |
| 6 | Readiness review (no live trading) | **Done (awaiting review)**: verdict NOT READY |
| 7–8 | Disabled broker adapter, live activation | Not started; each needs separate explicit approval |

## Setup

Requires **Python 3.12**.

```bash
py -3.12 -m venv .venv            # macOS/Linux: python3.12 -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
```

The lockfile includes the optional `rl` extra (PyTorch CPU build, Gymnasium), which PPO
needs. That is why the command adds PyTorch's package index. It also includes the
`dashboard` extra (Streamlit, Plotly). Everything else runs without them, and the
affected tests are skipped when they are missing.

No API keys or credentials are needed. Synthetic data is generated locally, and
historical data is imported from files you supply (see "Historical data" below).

## Run

Run from the repository root.

```bash
# Validate the clean synthetic dataset
python -m atlas.cli validate-data --config configs/data.yaml

# See validation catch injected duplicates, out-of-order, crossed, missing, and stale data
python -m atlas.cli validate-data --config configs/data_with_anomalies.yaml --strict

# Deterministic synthetic replay with the plumbing demo strategy
python -m atlas.cli simulate --config configs/simulator.yaml
```

`simulate` prints a JSON summary and writes a run record to `artifacts/reports/<run_id>/`:

| File | Contents |
|---|---|
| `summary.json` | result digest, dataset hash, seeds, validation summary, order/fill counts, final portfolio |
| `config_snapshot.yaml` | the exact simulator and data configs used |
| `fills.jsonl` | every simulated fill, with the venue quote it executed against |
| `audit.jsonl` | every order decision (with the visible quote and its age) and every execution outcome |
| `equity_curve.csv` | equity sampled about once per second of simulated time |
| `validation_issues.json` | every data issue found |

Re-running with the same configs produces the same `result_digest`.

Exit codes: `0` OK, `1` data validation failure, `2` configuration or checkpoint error.

### Evolution

```bash
# Run the configured number of generations (5) for the experiment in configs/evolution.yaml
python -m atlas.cli evolve --config configs/evolution.yaml

# Continue the same experiment from its latest checkpoint, up to 10 generations in total
python -m atlas.cli evolve --config configs/evolution.yaml --resume --generations 10
```

* **Speed.** One generation is 50 agents × 3 windows of 10 minutes. With `workers: 0` it
  uses all CPU cores but one, and takes about 2 s on a 16-core machine.
* **Existing experiments.** An experiment that already has checkpoints is never
  overwritten. Resuming with different settings is refused, because the config
  fingerprint no longer matches.

Outputs:

| Path | Contents |
|---|---|
| `artifacts/checkpoints/<experiment>/gen_NNN.json` | full state for resuming: population, random-generator state, archives, lineage, history, baselines. Contains no wall-clock data. |
| `artifacts/reports/evolution/<experiment>/generation_NNN.json` | status, frozen fitness config + fingerprint, parents, selection notes, per-family stats, baseline comparison, full leaderboard |
| `artifacts/reports/evolution/<experiment>/leaderboard_NNN.csv` | one row per agent: score, metrics, eligibility and the reasons behind it |
| `artifacts/reports/evolution/<experiment>/lineage.jsonl` | every agent ever created, with its parents, origin and exact parameter changes |
| `artifacts/reports/evolution/<experiment>/summary.json` | history, top champions, number of evaluations run |

### Historical data

Real market data is not included. The pipeline below runs end to end on synthetic files
exported in a vendor-style format. For a real vendor, copy
`configs/import_synthetic_demo.yaml`, point it at the vendor's CSV export, map the
column names, and record the license terms and where the timestamps come from.

```bash
# 1. Write 30 synthetic sessions as raw CSVs (raw data is never overwritten)
python -m atlas.cli export-synthetic --config configs/data.yaml --start 2026-06-01 --sessions 30 --duration-s 600 --symbols SPY QQQ --out data/raw/synthetic_demo

# 2. Import them into normalized Parquet sessions plus manifest.json (processed data is immutable)
python -m atlas.cli import-data --config configs/import_synthetic_demo.yaml

# 3. Check every session file against its manifest hash, then validate its events
python -m atlas.cli validate-data --dataset data/processed/synthetic_demo
```

**The manifest** records:
* source, feed, license, and where timestamps come from (exchange, vendor, local or
  synthetic)
* the SHA-256 hash of every raw and processed file
* per-session row counts, symbols and validation summary
* the point-in-time universe, taken from an optional `date,symbol` file; symbols the
  universe lists but the data lacks are recorded
* corporate actions, from an optional file
* the chronology assumption used to merge separate quote and trade files

A processed file whose hash no longer matches its manifest is refused.

**Prices** are raw and unadjusted. Every session is simulated on its own, starting flat
in cash, so no position spans a split or dividend and no adjustment is needed. Double
adjustment cannot happen. Features that span several days would need adjusted prices
first.

### Walk-forward evaluation

```bash
# Nested walk-forward on 30 in-memory synthetic sessions
python -m atlas.cli walkforward --config configs/walkforward.yaml

# The same procedure on the imported Parquet dataset
python -m atlas.cli walkforward --config configs/walkforward_parquet.yaml

# Rebuild and print the Markdown report of a run
python -m atlas.cli report --run-id <run_id>

# Stress-test an evolution checkpoint's champions on later sessions under every cost scenario
python -m atlas.cli evaluate --config configs/walkforward.yaml --checkpoint <path/to/gen_NNN.json>

# Only once the design is frozen: evaluate the final candidate on the outer holdout
python -m atlas.cli evaluate --config configs/walkforward.yaml --run-id <run_id> --holdout
```

**How it works**
([splits.py](src/atlas/evaluation/splits.py),
[walkforward.py](src/atlas/evaluation/walkforward.py)):
* **Ordering.** Sessions are kept in date order and never shuffled. The last
  `holdout_sessions` sessions form the outer holdout.
* **Folds.** The remaining sessions are cut into rolling or expanding folds:
  *train → embargo gap → validation*. Validation periods move forward fold by fold.
* **Each fold:**
  1. A fresh population evolves on the training sessions only.
  2. The top champions are scored again on those same sessions (in-sample).
  3. The champions and every baseline are scored on the validation sessions under the
     optimistic, base and pessimistic cost scenarios (out-of-sample).
* **Report contents** (`artifacts/reports/walkforward/<run_id>/report.{json,md}` and
  `design.json`):
  * champion out-of-sample returns with bootstrap confidence intervals, next to the
    baselines
  * the in-sample vs out-of-sample gap, as an estimate of selection bias
  * the number of selection trials
  * results by session, symbol and volatility regime
  * Sharpe, Sortino, profit factor, win rate, fill rate, exposure, and decision latency
    at p50/p95/p99
  * a data-quality table and the limitations
* **Emphasis.** Base and pessimistic costs are the main evidence. Optimistic results are
  shown only for contrast.

**Holdout protection** ([holdout.py](src/atlas/evaluation/holdout.py)):
* `walkforward` cannot load holdout sessions; trying raises an error, and a test checks
  that they are never read.
* `evaluate --holdout` works only when the current configs and data reproduce the
  frozen design's fingerprint.
* Every holdout access is written to `artifacts/holdout_ledger.jsonl`.
* Evaluating a *different* design on the same holdout is refused unless you pass
  `--allow-holdout-reuse`. Any reuse is recorded, and from then on the report warns that
  the holdout is no longer untouched.

### PPO (reinforcement learning)

```bash
# Walk-forward with PPO trained per fold, then compared with the evolution champions and baselines
python -m atlas.cli walkforward --config configs/walkforward_ppo.yaml

# Training stability: 3 seeds on fold 0's training sessions, each scored on fold 0's validation sessions
python -m atlas.cli train-ppo --config configs/walkforward_ppo.yaml --fold 0 --seeds 3
```

**Environment** ([env.py](src/atlas/learning/env.py)). This is a Gymnasium environment
running on the same replay engine as everything else; the engine now also has a
step-by-step mode for this.
* **Episode and step.** One episode is one training session, and one step is one
  decision at the usual 500 ms interval.
* **Action.** For each symbol the policy picks a target of 0, 10% or 20% of equity. That
  target goes through the same adapter, limits, latency and fill model as every
  evolutionary agent, so PPO gets no execution advantage.
* **Reward.** The reward is `log(V_t / V_{t-1})` after all fills and costs, scaled to
  bps.
* **Exit cost at the close.** At the end of an episode, the estimated cost of closing
  open positions is subtracted. This removes the shortcut of holding into the close to
  avoid paying the exit spread.

**Inputs** ([observation.py](src/atlas/learning/observation.py)):
* the causal per-symbol features, the position as a fraction of equity, the cash
  fraction, and how far through the session it is
* normalization statistics fitted **only on that fold's training sessions**, then
  versioned and stored in every checkpoint

**Training** ([ppo.py](src/atlas/learning/ppo.py)):
* PPO with clipping and GAE, using one shared per-symbol network
* seeded and single-threaded, so the same seed reproduces the same weights (tested)

Each update logs:
* policy and value loss, entropy, approximate KL, clip fraction, explained variance
* the action distribution
* **training reward against realized P&L**, as a check for reward hacking
* fills per episode

Warnings are raised for KL spikes, action collapse, never trading, and non-finite
losses.

**Frozen agent** ([policy.py](src/atlas/learning/policy.py)). Evaluation uses a
`PolicyAgent` that follows the common agent contract:
* **Pinned weights.** The record references its weights as `path#sha256`, and tampered
  weights are refused.
* **Frozen behavior.** It always picks the highest-scoring action and never learns
  during evaluation.

**Fair comparison** (from `configs/walkforward_ppo.yaml`):
* **Same training data.** For each fold, PPO trains on the fold's training sessions only.
* **Same scoring.** It is scored on the same validation sessions and cost scenarios as
  the evolution champions and the baselines.
* **Never auto-promoted.** PPO is a separate, versioned family (`ppo-v1`) and is never
  bred into the 50-agent population. The frozen design's final candidate comes from
  evolution unless `final_candidate_source: ppo` is set explicitly, and doing that
  changes the design fingerprint.

### Paper trading

Paper mode replays a session through the live-feed interface. It uses the same
validation, features and frozen agents as research, but orders go only through the
independent risk manager into Atlas's internal `PaperBroker`. **Paper mode has no live
trading endpoint, no live-order credentials, and no route that turns a paper order into
a real order.**

```bash
# Needs an evolution checkpoint first (configs/paper.yaml points at gen_005 of `evolve`)
python -m atlas.cli evolve --config configs/evolution.yaml
python -m atlas.cli paper --config configs/paper.yaml

# After a crash or stop: continue the same session from its last saved state
python -m atlas.cli paper --config configs/paper.yaml --resume

# Emergency stop of a running or stopped session (an operator must reset it before restarting)
python -m atlas.cli paper-kill --session-dir artifacts/paper/paper_2026-10-19 --reason "manual review"
python -m atlas.cli paper --config configs/paper.yaml --resume --reset-kill-switch

# Read-only dashboard (accounts, P&L, costs, risk events, fills, latency, alerts, walk-forward reports)
streamlit run src/atlas/dashboard/app.py -- --paper-dir artifacts/paper --reports-dir artifacts/reports/walkforward
```

**Safety checks at startup** ([config.py](src/atlas/paper/config.py),
[session.py](src/atlas/paper/session.py)):
* **Config.** The command refuses to start unless `mode` is exactly `paper`,
  `live_orders_enabled` is exactly `false`, and every limit in `configs/risk.yaml` is
  present and valid (none has a default). Unknown keys, such as credential fields, are
  rejected.
* **Broker.** The only broker factory builds a `PaperBroker`, and the session refuses any
  broker whose `is_live` is not `False`. No live-broker module exists. Only the
  read-only market-data module (`feeds/alpaca.py`) may use the network or read keys, and
  it can only send GET requests to two data endpoints. No trading host or orders path
  appears anywhere in the source, and a replay session runs with sockets disabled (all
  tested).
* **Health checks.** Before starting, it checks that:
  * the candidate checkpoint loads, and its genome or pinned weights match
  * the population or schema version matches
  * the feed's symbols are all in `allowed_symbols`
  * the session isn't a duplicate
  * no kill switch is engaged

**Accounts.** There is one frozen **champion**, up to five **shadows**, and an automatic
buy-and-hold **benchmark**. Each has its own paper broker, ledger and risk state.

**Paper broker** ([broker.py](src/atlas/paper/broker.py)):
* **Order states.** proposed, accepted, partially filled, filled, cancelled, rejected
  and expired.
* **Same fill rules as backtests.** Orders arrive after simulated latency and fill
  immediately-or-cancel at the top of the book. The arrival price is the latest
  *received* quote stamped at or before the arrival time.

**Risk manager** ([risk.py](src/atlas/paper/risk.py)). It checks every proposal after
the agent, using its own immutable copy of the limits:
* **Per order:** allowed symbols; order size in shares and dollars (reduced to fit);
  position limit per symbol (rejected); gross and net exposure (reduced, then rejected);
  orders per minute.
* **Per account:** a daily-loss or drawdown breach halts the account and cancels its
  working orders.
* **Global safe state:** stale or missing data (including total feed silence), a feed
  disconnect, or a clock anomaly stops all new actions and cancels working orders. Each
  raises an alert and clears automatically once the condition clears.
* **Reconnects:** retries are limited, and running out engages the kill switch.
* **Kill switch:** triggered by a `KILL` file, the `paper-kill` command, or a critical
  failure. It stays engaged across restarts.

**Session outputs** (`artifacts/paper/<session>/`):
* `audit.jsonl` — every decision (with market snapshot and quote ages), risk decision,
  simulated order, fill (with the fill assumption, fees and resulting cash and
  positions), cancel, data issue, clock anomaly, disconnect and kill
* `alerts.jsonl`
* `state.json` — saved atomically
* `equity.csv`
* `daily_report.{json,md}` — net P&L against the benchmark, drawdown, turnover, fills,
  broker and risk rejections, decision latency, feed gaps, disconnects, safe-state
  periods, and model/version identifiers

**Restart.** State is saved every `state_flush_events` events and at the end. After a
crash, `--resume`:
* restores every ledger and risk state, and the kill switch
* cancels orders that were working at the crash instead of assuming they filled
* cuts the audit log back to the last save, then replays the feed from that point, so
  nothing is skipped or written twice (tested)
* refuses if the configuration has changed. Fault-injection settings are excluded from
  that check, so drills can be resumed.

**Feeds** ([feed.py](src/atlas/paper/feed.py)):
* **Interface.** `LiveFeed` is the interface every feed implements. There are two: the
  live Alpaca data feed (see below) and `ReplayLiveFeed`.
* **Speed.** `speed: 0` runs on a virtual clock as fast as possible; `speed: 1` runs in
  real time.
* **Fault injection.** It can simulate disconnects and failed reconnects, data gaps,
  a clock going backwards, timestamps from the future, and crashes, for drills.

### Paper trading on live prices (Alpaca)

Paper trading on **live market prices** from Alpaca: seven frozen bots plus a
buy-and-hold benchmark, with a live page that opens automatically. **Market data only.**
Atlas reads prices from two GET endpoints on `data.alpaca.markets` and contains no code
that can place an order at Alpaca or anywhere else. Every order and fill is simulated by
Atlas's internal paper broker.

**One-time setup**
1. Create a free Alpaca account and open its **paper trading** account. Generate the
   paper account's API keys there; paper accounts get their own keys.
2. Copy `.env.example` to `.env` (git-ignored) and fill in `APCA_API_KEY_ID` and
   `APCA_API_SECRET_KEY`.
3. Make sure the PC clock is synced (Windows: Settings → Time & language → Sync now).
   Startup refuses a clock more than 3 s behind exchange time.
4. Have an evolution checkpoint. The bots come from
   `artifacts/checkpoints/synthetic_baseline_v1/gen_005.json`; create it with
   `python -m atlas.cli evolve --config configs/evolution.yaml` if it is missing.

**Each trading day** (start it any time before or during market hours): double-click
`scripts/start_live_paper.cmd`. It runs the preflight, then the session, in its own
window, and re-running it the same day resumes the session. Or use the commands directly:

```bash
python -m atlas.cli paper-check --config configs/paper_live.yaml     # preflight; does not trade
python -m atlas.cli paper --config configs/paper_live.yaml --auto-resume    # opens the leaderboard window
python -m atlas.cli paper-window --config configs/paper_live.yaml    # reopen the leaderboard window
python -m atlas.cli paper-watch --config configs/paper_live.yaml     # live leaderboard in this terminal
python -m atlas.cli paper-status --config configs/paper_live.yaml    # one-shot leaderboard
python -m atlas.cli paper-stop --config configs/paper_live.yaml      # clean, resumable stop
```

When running Atlas through Claude Code, see [CLAUDE.md](CLAUDE.md): sessions last about 6.5
hours, so they run in their own window or in the background.

**What happens when you run it:**
* **Startup checks.** The keys work, today is a weekday, the clock is in sync, and the
  bots load.
* **The leaderboard window opens.** It is a native dark trading-screen window
  (`paper-window`; Edge WebView2, no web server, no internet resources) and refreshes
  every second. It shows:
  * fund equity, today's and total P&L, the leader, trade count, and comparison against
    buy-and-hold
  * a live SPY/QQQ ticker
  * ranked bots with each one's capital, equity, today's and total P&L, sparkline, fills,
    positions and status
  * an equity chart and a live trade feed

  It stays open with the final standings after the close. Without the `window` extra,
  the terminal view (`paper-watch`) is used instead.
* **Permanent bots.** **Flip**, the coin-flip bot, and **Hold**, the buy-and-hold benchmark,
  are added to every paper session automatically. A config can't remove or duplicate
  them, and both names are reserved.
* **Bot names.** Each bot's `name` in the config (Liquidity-A, Liquidity-B, Regime,
  Momentum, MeanRevert, Random) is what appears on the leaderboard, in reports and in
  the fund ledger.
* **Coin-flip bot.** `baseline_coin_flip` flips a fair coin per symbol every 5 minutes:
  heads holds 1 share, tails holds none. It is long-or-flat because Atlas does not short.
  The flips are seeded by time, so a day replays identically but every day is new. A
  strategy worth keeping should beat it over time, after costs.
* **Fund.** `fund.total_capital` (here $100,000) is split equally across the 7 bots on
  day 1, $14,285.71 each, and each bot trades only its own share. At the close, open
  positions are sold at the bid, after costs. Each bot starts the next day with exactly
  what it ended with, so its capital grows and shrinks with its own P&L. The ledger is
  `artifacts/paper/fund_ledger.json`. Changing the bot lineup requires `--reset-fund`,
  which archives the old ledger rather than deleting it. The benchmark gets a fresh
  equal share each day. Atlas does not read your Alpaca balance; it only reads market
  data.
* **Trading hours.** The session waits for 09:30 ET, trades until 16:00 ET, then writes
  the daily report.

**The optional browser page** (`python -m atlas.cli dashboard`, or `--open-dashboard`, at
`http://localhost:8501`), refreshed every 2 seconds:
* a leaderboard ranking the bots by net P&L, against buy-and-hold
* each bot's realized and unrealized P&L, fees, fills, positions and halt status
* P&L over time
* recent simulated trades
* a per-bot detail view

**Reliability for an all-day run:**
* **Network trouble.** Reconnects back off (2 s, 4 s, 6 s and so on, capped at 30 s per
  attempt), so about 11 minutes of lost internet is ridden out in safe state. Only
  after that does the kill switch engage.
* **Bad responses.** Malformed responses and rate-limit replies count as temporary
  feed hiccups.
* **Bot errors.** A bot that raises an error is halted on its own; the others keep
  trading.
* **Sleep.** The PC is kept awake while a live session runs; the screen can still turn
  off.
* **Saving progress.** A provisional fund entry is saved about once a minute, so a day
  is never lost even if the window is closed and never resumed.
* **Duplicate launches.** Only one process can run a session; a second launcher
  refuses instead of corrupting files.
* **Tested:** a full simulated day (09:30-16:00 at real polling cadence) with a 4-minute
  outage, malformed responses and rate-limit errors. It finished without a kill, used
  about 62 s of CPU in total and 52 MB of memory, and averaged 89 requests a minute.

**Stopping and restarting:**
* Stop cleanly with Ctrl+C or `paper-stop`; use `paper-kill` only as an emergency halt.
* Continue the same day with `--resume`.
* Reopen the page any time with `python -m atlas.cli dashboard`.

**Feed details** ([alpaca.py](src/atlas/feeds/alpaca.py)), checked against Alpaca's
official docs on 2026-10-26:
* **Data:** the free plan provides real-time **IEX** quotes and trades. IEX's top of book
  is usually wider than the national best bid/offer, so paper fills are conservative.
* **Polling:** latest quotes every second and trades every 2 s, which is 90 requests a
  minute against a 200 per minute limit. Only changed quotes and new trades are used.
* **Risk settings** (`configs/risk_live.yaml`) are tuned for polled data: a 30 s stale
  limit, 2 s decisions and a 3 s clock tolerance.

**These bots were evolved on synthetic data,** so their live results show how the system
behaves, not whether a strategy has an edge.

### Readiness review

```bash
# Collect evidence, run the failure drills and the test suite, score the gates, write the report
python -m atlas.cli readiness --config configs/readiness.yaml
```

The latest review is committed at [docs/READINESS_REVIEW.md](docs/READINESS_REVIEW.md).
Its current verdict is **NOT READY**: 4 required gates pass, 9 fail, 5 have no
evidence yet, and 4 await a human sign-off.

**What it checks** ([readiness.py](src/atlas/readiness.py)):
* **Data fidelity:** real licensed data, timestamp provenance, point-in-time universe.
* **Out-of-sample results:** base and pessimistic confidence intervals above zero,
  beating cash, enough sessions, selection gap.
* **Outer holdout:** evaluated exactly once for the frozen design.
* **Paper trading:** sessions on a live feed, at least 20 sessions, net P&L.
* **Risk incidents.**
* **Operational reliability:** 9 failure drills against the shipped paper config, the
  full test suite, latency, and the vendor feed adapter.
* **Broker capabilities:** 10 unverified items, each with its official documentation
  link.
* **Governance:** independent risk, legal/regulatory/tax, broker terms, and runbook
  sign-offs. These are recorded in `configs/readiness.yaml` with reviewer, date and
  document.

Missing evidence counts as **MISSING**, never as a pass. The verdict is "READY FOR HUMAN
REVIEW" only if every required gate passes. Even then it **enables nothing**: a broker
adapter (Phase 7) and real money (Phase 8) each need separate explicit approval.

## Test

```bash
python -m pytest
```

The suites cover:

| File | Covers |
|---|---|
| `test_data_validation.py` | duplicate, out-of-order, missing/NaN, impossible, crossed, stale, clock-anomaly, and out-of-session events |
| `test_no_lookahead.py` | strategies see only received events; corrupting future data leaves past decisions unchanged |
| `test_portfolio_accounting.py` | cash, average cost, realized/unrealized P&L, fees, no margin or shorting, plus a property test that the ledger always reconciles |
| `test_execution.py` | fills at bid/ask (never mid), slippage, fees, latency-driven adverse fills, partial fills, limits that only touch the last trade not filling, stale-quote rejection |
| `test_reproducibility.py` | same seed gives an identical run; no state leaks between runs |
| `test_config_and_cli.py` | strict config validation (unknown keys rejected) and CLI end to end |
| `test_agents.py` | genome operators, 10 agents per family, baselines kept separate, features (returns, trade signing, flow windows, staleness), illegal-target and limit handling, every family reconciles, no shared state |
| `test_fitness_selection.py` | metric definitions, the documented score, every gate, no breach offset by return, diverse parents, never padding with ineligible agents, offspring mix and lineage, archives |
| `test_evolution.py` | exactly 50 agents per generation (5 × 10), full lineage, an interrupted-and-resumed run identical to an uninterrupted one, refusal to overwrite or resume on changed config, hold and explore policies, parallel results identical to serial |
| `test_ingest.py` | lossless synthetic → CSV → Parquet round trip, immutable raw and processed data, refusal of tampered files, vendor column mapping, ISO timestamps, universe and corporate-action files |
| `test_walkforward.py` | date-ordered splits with embargo and rotating validation, holdout guard and ledger (reuse refused unless forced), cost scenarios, round-trip metrics, best-of-many overfitting the selection data, a walk-forward run that never reads holdout files, then a single frozen-design holdout evaluation |
| `test_risk_controls.py` | every per-order rule; daily-loss and drawdown halts; every risk field required, invalid values rejected, limits immutable; session-level halts, stale-feed and clock-anomaly safe states, bounded reconnects, kill switch on exhausted retries, manual kill that blocks orders and survives restart, disallowed symbols, model mismatch |
| `test_paper_cannot_trade_live.py` | config accepts only `mode: paper` with live orders off, rejects missing safety settings and credential fields, refuses configs that bypassed validation, accepts only `PaperBroker`, no live-broker module, no network imports or environment credentials, a full session with sockets disabled and every fill marked simulated |
| `test_paper_session.py` | accounts, audit and daily report; deterministic sessions; crash then resume with nothing skipped or duplicated and working orders cancelled; resume refused on changed config |
| `test_readiness.py` | current evidence scores NOT READY, missing evidence reported as MISSING, gates pass only with complete positive evidence and sign-offs, a reused holdout fails, all 9 failure drills pass, unknown sign-offs rejected |
| `test_alpaca_feed.py` | Alpaca feed against a fake API and clock: exact nanosecond timestamps, keys from the environment or `.env` and never printed, waits for the open and stops at the close, emits only new data, stays under the rate limit, disconnects and reconnects; a full live session with the leaderboard snapshot; live dates must be today; the live page renders |
| `test_dashboard.py` | the dashboard imports nothing that can trade, file loaders, the Streamlit app renders |
| `test_learning.py` (skipped without the `rl` extra) | step-by-step engine matches the callback engine, normalizer fitted on given sessions only, environment spaces, reward adding up to the realized log return, sessions cycled, observations unaffected by future data, deterministic PPO with full diagnostics, frozen and pinned policy agent (parallel equals serial, tampered weights refused), PPO in walk-forward never promoted automatically, explicit PPO candidate through the holdout |

## How agents and evolution work

**Agents.**
* Every agent implements `initialize`, `observe`, `act`, `learn`, `save_checkpoint`,
  `load_checkpoint` and `metadata` ([base.py](src/atlas/agents/base.py)).
* `act` returns a *target position* per symbol. Agents never place orders: an adapter
  turns the change in target into bounded market orders. It allows one working order per
  symbol and never acts on stale data.
* The adapter clips targets to hard limits (`limits` in the config) and records any
  illegal target as a violation.
* A fresh agent instance is built from its record for every run.

**Families.** Ten agents each, defined in [families.py](src/atlas/agents/families.py):

| Family | Rule |
|---|---|
| Momentum | buy after a strong enough up-move over a chosen horizon |
| Mean reversion | buy after a down-move that is large relative to recent volatility, unless the order book is leaning heavily to the sell side |
| Order flow | buy on a blend of net buying pressure in recent trades and top-of-book size imbalance |
| Liquidity | buy only into tight, deep markets with support on the bid; sell when liquidity worsens |
| Regime-adaptive | momentum when the market is calm; mean reversion (or stay out) when it's volatile |

**Baselines.** These are scored every run, but they are never bred or selected:
* cash (never trades)
* equal-weight buy-and-hold
* simple momentum
* simple mean reversion
* seeded random control

**Features** ([features.py](src/atlas/agents/features.py)). Every feature is built only
from data the agent has already received:
* spread and top-of-book sizes
* returns over 0.5 s, 1 s, 5 s and 30 s
* short- and long-term volatility, and their ratio
* trade flow signed as buys or sells, over 1 s, 5 s and 30 s windows

**Fitness** ([fitness.py](src/atlas/evolution/fitness.py)).
* Hard gates are checked first:
  * maximum drawdown
  * maximum turnover
  * no illegal actions
  * at least 30 fills and a trade in every window — otherwise the agent is labeled
    "insufficient evidence"
* An agent that fails any gate gets no score, so a high return can never offset a breach.
* Agents that pass are scored with
  `J = w_R·R/a_R − w_D·D/a_D − w_V·V/a_V − w_T·T/a_T − w_S·S/a_S`. The terms are mean net
  return, worst drawdown, downside deviation, turnover and instability across windows.
  Each is divided by a fixed anchor declared in the config, so the terms are comparable.
* Weights, anchors and gates are frozen in the config, and their fingerprint is logged
  with every generation.

**Generation** ([runner.py](src/atlas/evolution/runner.py)):
1. Every agent is evaluated on every window.
2. Agents are gated and scored.
3. The champion and novelty archives are updated.
4. Up to 5 parents are chosen. Near-clones are skipped, and at most 2 parents may come
   from one family; if a rule has to be relaxed to fill the slots, the report says so.
5. Each parent produces 10 children: 1 exact copy, 6 small mutations (some crossed with
   another parent of the same family first), 2 large mutations and 1 fresh random agent.
6. The new generation is checkpointed and reported.

If fewer than 5 agents pass the gates, nothing weak is promoted. Under `hold` the run
stops and keeps the current population. Under `explore` the open slots get fresh random
agents, labeled as such.

## How the simulator works

* **Time.** All timestamps are integer nanoseconds, UTC. Each event keeps its source time
  (`event_ts`) and its receive time (`receive_ts`) separately.
* **Two clocks.**
  * A strategy sees an event only once it has been received. If an event has no receive
    time, the configured feed latency is added to its source time.
  * The simulated venue knows the true market by source time.
* **Orders.**
  * An order decided at time *t* reaches the venue at *t* + decision latency + order latency.
  * It executes against the venue quote at arrival, so the price can move against it in
    between.
  * Buys pay the ask and sells receive the bid. Slippage and optional spread widening are
    always adverse.
* **Fill model.**
  * Orders are immediate-or-cancel (IOC) against the top of the book.
  * A fill is capped by `max_participation` × the displayed size, available cash, and the
    current position (long-only).
  * A limit order fills only if it is marketable after slippage. A limit price that merely
    touches the last trade never fills.
* **Stale data.**
  * Orders are rejected if the visible quote is stale at decision time, or if the venue
    quote is stale at arrival.
  * Quotes are never forward-filled.
* **Validation.**
  * Defective events are dropped, never repaired or reordered.
  * Stale gaps are flagged.
  * `invalid_event_policy: fail` aborts the run instead of dropping.
* **Determinism.** All randomness comes from seeded generators, and every tie in event
  ordering is broken by a fixed key.

## What the first evolution run shows

The shipped config was run for 10 generations on synthetic random-walk data.

* **No evolved agent beats cash.** The best agent scored −0.035 at +0.3 bps per window,
  in-sample. The cash baseline earns 0 bps. Every baseline that trades loses money,
  between −9 and −34 bps per window. This is the correct result: a random walk has no
  edge, so crossing the spread only costs money.
* **Selection pushes agents to trade as little as the gate allows.** The leaders sit at
  exactly the 30-fill minimum. An early run with a 6-fill gate selected lucky agents that
  had made almost no trades. That is why the evidence gate was raised, and the
  per-family parent cap added.
* **Three families take over.** By generation 6, momentum, regime-adaptive and liquidity
  hold most of the 50 slots. Mean reversion and order flow survive only as occasional
  fresh random agents. The champion and novelty archives keep a record of the families
  that died out.
* **Scores are in-sample.** Selection reuses the same 3 windows every generation, so the
  scores overfit them by construction. Phase 3 adds walk-forward folds by date and an
  untouched holdout period to measure that.

## What the first walk-forward run shows

This is `configs/walkforward.yaml`: 30 synthetic sessions, of which 6 are held out, split
into 6 folds with 4 generations each. It took about 40 s on 16 cores. Returns are the
mean per validation session for each fold's top champion, with a 90% bootstrap
confidence interval:

| Cost scenario | Champion, out-of-sample | Cash | Random control |
|---|---|---|---|
| optimistic | +0.10 bps [−0.24, +0.52] | 0 | −6.2 |
| **base** | **−0.39 bps [−0.73, −0.02]** | 0 | −9.7 |
| **pessimistic** | **−2.92 bps [−3.78, −2.17]** | 0 | −26.3 |

* **No champion beats cash out of sample under base or pessimistic costs.** That is the
  correct answer on random-walk data.
* **Selection bias.** Champions average −0.13 bps in-sample and −0.39 bps out-of-sample.
  The +0.26 bps gap is the selection bias the walk-forward exists to expose.
* **Selection trials.** There were 4,800 agent-session evaluations, and every one is
  counted in the report.
* **Data path check.** The same run on the imported Parquet dataset gives exactly the
  same numbers, which shows the import path loses nothing.
* **Holdout.** The 6 holdout sessions have not been evaluated.

## What the first PPO run shows

This is `configs/walkforward_ppo.yaml`, with the same 6 folds as above and 49,152 PPO
steps per fold. It took about 3 minutes. Returns are out-of-sample, per validation
session:

| Cost scenario | PPO | Evolution champion | Cash |
|---|---|---|---|
| **base** | **−2.72 bps [−4.73, −0.68]** | −0.39 bps [−0.73, −0.02] | 0 |
| **pessimistic** | **−3.47 bps [−5.45, −1.46]** | −2.92 bps [−3.78, −2.17] | 0 |

* **PPO loses to cash and to the evolution champion.** It trails the champion in 6 of 6
  folds under base costs. Under pessimistic costs it was ahead of the champion in 3 of 6
  folds, but it still lost money in every fold, so it was **not promoted**.
* **What PPO learned.** In every fold it ends up choosing "half position" about 98% of
  the time. It learns to stop the costly churning of random exploration, which costs
  about −145 bps per episode early in training. But it settles on a constant small
  position rather than cash.
* **Training stability.** Three seeds on fold 0 train to different weights, but all
  three settle on the same constant action, so they score identically
  (−2.93 bps, spread 0). The result does not depend on the seed, and at this budget the
  policy is that degenerate constant position.
* **No sign of reward hacking.** Training reward tracks realized P&L to within 0.14–0.25
  bps per episode.
* **No sign of training instability.** The largest per-update KL was 0.015, and there
  were no warnings.
* **Small in-sample/out-of-sample gap.** −2.60 bps against −2.72 bps, much smaller than
  evolution's, because a constant position has little to overfit.
* **Slower decisions.** PPO takes about 230 µs per decision at p99, against about 24 µs
  for the rule agents. Both are measured on this machine.

## What the first paper session shows

This is `configs/paper.yaml`: a 30-minute synthetic session on 2026-10-19 with 23,555
events, a champion from evolution checkpoint `gen_005`, two shadows and the benchmark.
It took about 1 s on the virtual clock.

| Account | Model | Return | vs buy & hold | Fills |
|---|---|---|---|---|
| champion | g003-a05 (liquidity) | −5.38 bps | +18.8 bps | 16 |
| shadow-1 | g004-a04 (liquidity) | −5.33 bps | +18.9 bps | 16 |
| shadow-2 | random control | −34.12 bps | −9.9 bps | 121 |
| benchmark | buy & hold | −24.20 bps | 0 | 2 |

* **The champion lost money but beat buy-and-hold.** It loses less because it holds
  little exposure while the market falls. One synthetic day is no evidence of edge.
* **Validation dropped 33 late quotes** that arrived after a newer quote for the same
  symbol. The paper feed checks order separately for each symbol's quotes and trades,
  because real feeds deliver those channels independently.
* **Drills behave as designed:**
  * A disconnect with two failed reconnects recovers on the third attempt.
  * When retries run out, the kill switch engages.
  * A crash followed by `--resume` processes the same 23,555 events as an uninterrupted
    run.

## Known limitations

* **Synthetic data only.** It tests plumbing and carries no information about real markets.
  No result from it is evidence of profitability, and the demo strategy is not a research
  strategy.
* **Session calendar.** Only the regular session (09:30–16:00 ET, weekdays) is modeled.
  Holidays, early closes, halts, and corporate actions are not modeled yet.
* **Fill model.** Only IOC against top of book. There are no resting limit orders, queue
  position, or depth beyond Level 1, and no market impact beyond the fixed slippage in
  basis points.
* **Fees.** The fee schedule in `configs/simulator.yaml` is an illustrative placeholder.
* **Prices.** Prices and cash are floats. The ledger reconciles to within 1e-6 × initial cash.
* **No real data yet.** No vendor data has been imported. The importer handles generic
  CSVs, but each vendor's format, entitlements, timestamp quality and redistribution
  terms must be checked before use.
* **No holidays.** Synthetic sessions run on consecutive weekdays, so market holidays
  such as 2026-06-19 and 2026-07-03 are included.
* **Selection uses base costs only.** Evolution selects under the base cost scenario.
  All three scenarios are applied only when scoring out-of-sample.
* **Few validation sessions.** Out-of-sample evidence is 12 validation sessions, which
  are not independent of each other, so the confidence intervals are approximate.
* **Noisy Sharpe and Sortino.** They are annualized from 1-minute returns within
  10-minute sessions.
* **Machine-dependent latency.** Decision latency is wall-clock time on this machine. It
  excludes any network or broker latency.
* **Risk manager runs in paper mode only.** Backtests and walk-forward rely on agent
  limits plus the fitness gates.
* **Small PPO budget.** PPO trains on about 41 ten-minute episodes per fold, which is
  small for reinforcement learning. The action set is three fixed target levels with
  market orders only. The results describe this setup, not reinforcement learning in
  general.
* **No PPO hyperparameter search.** A tuned search would be one more selection trial
  and would need its own walk-forward accounting.
* **PPO uses base costs only.** It trains under the base cost scenario.
* **Polled IEX data.** The live feed polls Alpaca's IEX quotes; it is not a streamed
  consolidated feed. Quotes are up to about a second old when used, and IEX shows only
  IEX's own book.
* **Silence is detected late.** A feed that goes completely silent is noticed when the
  next event arrives; the safe state is then backdated to the moment data went stale. A
  real-time vendor adapter must deliver heartbeats or time out so a silent feed is
  noticed without waiting for the next event.
* **Paper fills are an approximation.** They use the latest *received* quote at arrival,
  so they include feed latency. They are simulations, not executions.
* **Agent memory resets on restart.** Positions survive a restart, but in-memory agent
  state such as entry times starts fresh.
* **Halted accounts keep their positions.** Nothing is flattened automatically; the
  positions wait for manual review.
* **Alerts go only to the console and a file.** Nothing sends messages externally.

## Repository layout

```
configs/           data.yaml, data_with_anomalies.yaml, simulator.yaml, evolution.yaml,
                   walkforward.yaml, walkforward_parquet.yaml, walkforward_ppo.yaml,
                   import_synthetic_demo.yaml, paper.yaml, risk.yaml, paper_live.yaml, risk_live.yaml,
                   readiness.yaml
src/atlas/
  cli.py           command-line entry point
  config.py        typed, strict configuration models
  simulation.py    run orchestration and run records
  timeutils.py     ns timestamps, session bounds
  logging_utils.py JSON structured logging
  data/            events, synthetic generator, validation, providers, Parquet storage, CSV import
  environment/     replay engine, market view, orders, costs, latency, portfolio, demo strategy
  agents/          agent contract, features, genome operators, 5 families, baselines, adapter
  evaluation/      metrics, splits, holdout guard, cost scenarios, walk-forward, reports
  learning/        Gymnasium env, observation normalizer, PPO trainer, frozen policy agent, seed stability
  brokers/         broker interface only (no live adapter exists)
  feeds/           Alpaca market-data feed (read-only; the only module allowed network access)
  paper/           paper config, live-feed interface + replay feed, paper broker, risk manager, session, report
  dashboard/       read-only Streamlit app and file loaders
  readiness.py     Phase 6 review: evidence, failure drills, gates, broker open items
  evolution/       evaluation (parallel), fitness gates, selection, reproduction, archives, runner
tests/             pytest suites
data/              raw/ processed/ manifests/ (licensed data is never committed)
artifacts/         checkpoints/ reports/ (generated, git-ignored)
docs/              project specification, READINESS_REVIEW.md
```
