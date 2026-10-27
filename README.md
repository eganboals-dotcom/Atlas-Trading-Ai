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

## Status: Phase 4 — PPO

| Phase | Scope | Status |
|---|---|---|
| 1 | Events, synthetic data, validation, deterministic replay, costs/latency, ledger | Done |
| 2 | Agent interface, 50 non-RL agents, baselines, evolution, lineage | Done |
| 3 | Historical data, walk-forward evaluation, metrics | Done |
| 4 | PPO policy family | **Done (awaiting review)** |
| 5 | Live-data paper trading, risk manager, dashboard | Not started |
| 6–8 | Readiness review, disabled broker adapter, live activation | Not started; each needs separate approval |

## Setup

Requires **Python 3.12**.

```bash
py -3.12 -m venv .venv            # macOS/Linux: python3.12 -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps
```

The lockfile includes the optional `rl` extra (PyTorch CPU build, Gymnasium), which PPO
needs. That is why the command adds PyTorch's package index. Everything else runs
without it, and `test_learning.py` is skipped when PyTorch is missing.

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
* **Limits, not a risk manager.** Agent limits clip positions and order sizes. Daily loss
  limits, order-rate limits and the kill switch are part of the Phase 5 risk manager.
* **Small PPO budget.** PPO trains on about 41 ten-minute episodes per fold, which is
  small for reinforcement learning. The action set is three fixed target levels with
  market orders only. The results describe this setup, not reinforcement learning in
  general.
* **No PPO hyperparameter search.** A tuned search would be one more selection trial
  and would need its own walk-forward accounting.
* **PPO uses base costs only.** It trains under the base cost scenario.
* **Not built yet.** Paper trading, the live data provider and the full risk manager
  come in Phase 5.

## Repository layout

```
configs/           data.yaml, data_with_anomalies.yaml, simulator.yaml, evolution.yaml,
                   walkforward.yaml, walkforward_parquet.yaml, walkforward_ppo.yaml,
                   import_synthetic_demo.yaml
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
  evolution/       evaluation (parallel), fitness gates, selection, reproduction, archives, runner
tests/             pytest suites
data/              raw/ processed/ manifests/ (licensed data is never committed)
artifacts/         checkpoints/ reports/ (generated, git-ignored)
docs/              project specification
```
