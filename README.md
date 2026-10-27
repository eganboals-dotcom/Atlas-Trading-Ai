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

## Status: Phase 1 — simulation foundation

| Phase | Scope | Status |
|---|---|---|
| 1 | Events, synthetic data, validation, deterministic replay, costs/latency, ledger | **Done (awaiting review)** |
| 2 | Agent interface, 50 non-RL agents, baselines, evolution, lineage | Not started |
| 3 | Historical data, walk-forward evaluation, metrics | Not started |
| 4 | PPO policy family | Not started |
| 5 | Live-data paper trading, risk manager, dashboard | Not started |
| 6–8 | Readiness review, disabled broker adapter, live activation | Not started; each needs separate approval |

## Setup

Requires **Python 3.12**.

```bash
py -3.12 -m venv .venv            # macOS/Linux: python3.12 -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

No API keys or credentials are needed. All Phase 1 data is generated locally.

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

Exit codes: `0` OK, `1` data validation failure, `2` configuration error.

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

## Known limitations (Phase 1)

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
* **Not built yet.** There is no risk manager (Phase 5), no agent population or evolution
  (Phase 2), and no historical-data adapter or walk-forward evaluation (Phase 3).

## Repository layout

```
configs/           data.yaml, data_with_anomalies.yaml, simulator.yaml
src/atlas/
  cli.py           command-line entry point
  config.py        typed, strict configuration models
  simulation.py    run orchestration and run records
  timeutils.py     ns timestamps, session bounds
  logging_utils.py JSON structured logging
  data/            events, synthetic generator, validation
  environment/     replay engine, market view, orders, costs, latency, portfolio, demo strategy
tests/             pytest suites
data/              raw/ processed/ manifests/ (licensed data is never committed)
artifacts/         checkpoints/ reports/ (generated, git-ignored)
docs/              project specification
```
