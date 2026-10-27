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

## Status: Phase 2 — agents and evolution

| Phase | Scope | Status |
|---|---|---|
| 1 | Events, synthetic data, validation, deterministic replay, costs/latency, ledger | Done |
| 2 | Agent interface, 50 non-RL agents, baselines, evolution, lineage | **Done (awaiting review)** |
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
* **Fixed evaluation windows.** Phase 2 evaluates on fixed synthetic windows. Window
  rotation, nested walk-forward and an outer holdout come in Phase 3.
* **Single cost scenario.** Agents are evaluated under one cost scenario. Pessimistic,
  base and optimistic scenarios come in Phase 3.
* **Limits, not a risk manager.** Agent limits clip positions and order sizes. Daily loss
  limits, order-rate limits and the kill switch are part of the Phase 5 risk manager.
* **Coarse metrics.** Drawdown uses equity sampled about once per second plus every
  fill. Sharpe, Sortino and profit factor arrive in Phase 3.
* **Not built yet.** There is no PPO (Phase 4) and no historical-data adapter
  (Phase 3).

## Repository layout

```
configs/           data.yaml, data_with_anomalies.yaml, simulator.yaml, evolution.yaml
src/atlas/
  cli.py           command-line entry point
  config.py        typed, strict configuration models
  simulation.py    run orchestration and run records
  timeutils.py     ns timestamps, session bounds
  logging_utils.py JSON structured logging
  data/            events, synthetic generator, validation
  environment/     replay engine, market view, orders, costs, latency, portfolio, demo strategy
  agents/          agent contract, features, genome operators, 5 families, baselines, adapter
  evaluation/      run metrics
  evolution/       evaluation (parallel), fitness gates, selection, reproduction, archives, runner
tests/             pytest suites
data/              raw/ processed/ manifests/ (licensed data is never committed)
artifacts/         checkpoints/ reports/ (generated, git-ignored)
docs/              project specification
```
