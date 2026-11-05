# PROJECT ATLAS

*A plain-text copy of [PROJECT_SPECIFICATION.docx](PROJECT_SPECIFICATION.docx), the original build spec from October 2026. For
where the project stands now, see the [README](../README.md).*

50-Agent Evolutionary Trading AI

Complete build specification for Claude Code Architecture • Learning • Selection • Runtime • Paper trading • Future live deployment

> **Non-negotiable deployment rule**
>
> Build simulation first and paper trading second. A real-money adapter may be designed behind a strict interface, but it must be disabled and must not be implemented as an active order path during the initial project. Live trading requires a separate explicit approval, readiness review, and additional tests.

Version 3.0 | Final architecture, learning, evaluation, deployment, and repository-profile review | October 26, 2026

## GitHub repository bio (copy/paste)

### GitHub repository description (paste into the repository About section)

50-agent evolutionary trading AI with genetic optimization, reinforcement learning, and realistic market simulation.

### Optional README summary

Project Atlas is a research platform for evolving and evaluating algorithmic trading agents with genetic selection, reinforcement learning, event-driven market simulation, and strict risk controls. Strategies are tested on historical data and live-data paper trading before any separately approved live deployment. Profitability is not assumed; it must be demonstrated with robust out-of-sample evidence.

### Suggested GitHub topics

algorithmic-trading, reinforcement-learning, evolutionary-algorithms, quantitative-finance, market-simulation, backtesting, paper-trading, python

## 0. Instructions for Claude Code

Treat this document as the source of truth. The repository owner will create the Git repository and open it in Claude Code. Do not generate the entire system in one pass.

- First inspect the repository and report what exists. Do not overwrite useful work.
- Create a phased implementation plan, then implement Phase 1 only. Run tests and summarize changed files, test results, known limitations, and next steps.
- Do not move to the next phase without explicit approval from the user.
- Use synthetic data first. Do not require API keys for tests or basic demos.
- Never claim a strategy is profitable based on synthetic data or backtests alone.
- No live order placement in Phases 1–7. Paper trading must be technically unable to place real orders.
- If API documentation or broker capabilities are unclear, isolate the integration behind an adapter and document the unresolved item rather than guessing.
- Use type hints, small modules, clear comments, dependency pinning, automated tests, structured logs, and a useful README.

> **Critical correction for the live-trading plan**
>
> A Schwab real-time market-data credential is not by itself a complete live-trading setup. API-product access, OAuth authorization, app approval, account permissions, terms, rate limits, and supported order types must be verified. Schwab documents a developer sandbox for testing supported API methods with sample data, but it must not be assumed to be a realistic paper-trading execution simulator for this strategy. Keep Atlas paper fills simulated internally, and verify exact current product capabilities before any future broker adapter is implemented.

## 1. Mission and scope

Project Atlas is a research platform that evolves a population of 50 algorithmic agents for liquid U.S. stocks and ETFs. Agents compete on identical market data and execution assumptions. The system selects promising eligible parents, generates descendants, and optionally trains policy agents using reinforcement learning. The target horizon is sub-second to seconds, but the system must explicitly state what the available feed and timestamps can support.

- Initial universe: a small, configurable set of highly liquid U.S. ETFs, expanding only after validation. Start long-only and cash-funded: no margin, leverage, short selling, or options unless a separately reviewed phase adds and tests those risks.
- Primary output: reproducible evidence about strategy performance, robustness, risk, and execution—not a promise of profit.
- Development sequence: synthetic tests → historical replay → walk-forward testing → live-data paper trading → separate live-readiness review.
- No real orders, live-order credentials, or live-order endpoint access in Phases 1–7. Phase 7 may only build a disabled adapter and test it against mocks or an officially documented sandbox. Enabling production orders is a separate Phase 8 decision requiring explicit approval.
- A retail brokerage API is not automatically suitable for true high-frequency trading. API rate limits, order latency, data granularity, and broker policies may make sub-second strategies impractical.

## 2. Architecture and runtime topology

| Process / service | Runs when | Responsibilities | Failure behavior |
|---|---|---|---|
| Offline research runner | On demand | Load datasets, train policies, backtest agents, compute metrics | Fail the run, preserve logs/checkpoint, never trade |
| Evolution coordinator | Offline after each evaluation cycle | Rank agents, select parents, create descendants, version generations | Do not promote if evaluation is incomplete or invalid |
| Paper-trading process | During configured market hours | Consume live data, score frozen candidate(s), simulate orders/fills, monitor risk | Fail closed on stale data, disconnects, bad clock, or risk breach |
| Risk manager | Inside simulator and paper process | Enforce independent limits on every proposed action/order | Reject action and raise alert; kill switch on critical failures |
| Dashboard/reporting | As needed / during paper sessions | Show agents, P&L, costs, risk, fills, latency, logs | Read-only; cannot bypass risk controls |
| Future live adapter | Not active in initial project | Potentially translate approved orders to broker API | Must remain disabled until separate authorization and review |

Initially run everything on one developer machine, with separate CLI commands/process configurations for research and paper trading. Do not add distributed infrastructure before the single-machine version is correct. Training and live-data paper trading must not share mutable model weights.

## 3. Suggested stack

| Area | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | Pinned dependencies and lockfile |
| Data processing | NumPy, Polars | Causal feature computation |
| Storage | Parquet + DuckDB | Immutable raw data, dataset manifests and hashes |
| Model training | PyTorch | Optional; simple rule-based agents are valid baselines |
| RL environment | Gymnasium | Explicit observation/action/reward API |
| RL baseline | PPO | Experimental candidate, not assumed best |
| Evolution | Custom code or DEAP | Selection, mutation, diversity, lineage, checkpoint/resume |
| Tracking | MLflow or structured JSON/CSV | Every run stores code/data/config/seed/metrics |
| Dashboard | Streamlit + Plotly | Research and paper monitoring |
| Testing | pytest | Unit, integration, property, leakage, and safety tests |

## 4. Data ingestion and quality

### Required event schema

- Every quote/trade event should contain symbol, source event timestamp, local receive timestamp when available, source/feed, sequence number when available, event type, and relevant venue/condition metadata. If exact event ordering cannot be determined, record the ambiguity; do not invent chronology.
- Quotes should contain bid/ask prices and sizes; trades should contain price, size, and relevant conditions.
- Record timestamps in UTC internally, plus market session/calendar metadata. Preserve source timestamps rather than replacing them with processing time.
- Validate ordering, duplicates, missing values, impossible prices/sizes, stale quotes, out-of-session data, and feed disconnects.
- Do not forward-fill stale quotes for executable prices. Mark features invalid or suspend trading until valid data resumes.
- Store raw data immutably; processed data should include source, license/entitlement notes, schema version, content hash, corporate-action treatment, and point-in-time universe information when available. Track whether timestamps are exchange/source timestamps, vendor timestamps, or local receive timestamps.

### Data fidelity levels

| Data level | Can support | Cannot prove by itself |
|---|---|---|
| OHLCV bars | Prototype, slower strategy research | Sub-second signals, spread-aware fills, queue position |
| Timestamped trades + Level 1 quotes | Quote-aware event-driven research | Exact depth/queue behavior or exact fills |
| Level 2/order-book updates | Depth/imbalance features and improved fill simulation | Guaranteed queue position or perfect historical fills |
| Replay-grade sequenced events | More detailed execution replay | Future live fill certainty or zero market impact |

The chosen provider’s entitlements, historical coverage, timestamp quality, symbol limits, redistribution/use rights, and streaming limits must be verified before committing to a data design. Schwab’s official portal lists market-data and accounts/trading product areas; access to one does not guarantee access to the other. Treat available timestamp precision and feed latency as measured constraints, not assumptions.

## 5. Agent population and behavior

| Family | Count | Initial strategy hypothesis |
|---|---|---|
| Short-term momentum | 10 | Continuation over short horizons where data quality supports it |
| Mean reversion | 10 | Reversion after brief dislocations; guard against adverse selection |
| Order-flow imbalance | 10 | Buy/sell pressure imbalance; requires reliable trade/quote data |
| Liquidity-aware | 10 | Spread- and depth-sensitive behavior; no assumed queue priority |
| Regime-adaptive | 10 | Adjust behavior to volatility and liquidity regimes |
| Total active population | 50 | Five parents may each generate ten descendants for the next generation |
| Fixed evaluation baselines | Outside population | Cash/no-trade, buy-and-hold where relevant, simple momentum/reversion, randomized control; never bred or selected |
| PPO / reinforcement learning | Phase 4 onward | Added only after simulator and evaluation gates pass; population composition and version must be logged |

### Common agent contract

Every agent must implement a stable interface equivalent to:

- initialize(config, seed) — deterministic setup
- observe(market_state, portfolio_state) — receive causal, validated features
- act(observation) — return a constrained target position or action proposal
- learn(experience_batch) — optional offline update; disabled in frozen paper candidates
- save_checkpoint(path) / load_checkpoint(path) — reproducible state
- metadata() — agent ID, family, generation, parents, code/config version

An agent proposes; the independent risk manager decides whether the proposed action is permitted. Agents must never place orders directly. The initial 50-agent population consists of five non-RL strategy families; PPO is added only after Phase 4 begins. Fixed evaluation baselines sit outside the competitive population and never get bred into it.

## 6. Features, action space, and inventory

### Features

- Spread in basis points, bid/ask sizes, quote age, and spread change.
- Mid-price returns over 100 ms, 500 ms, 1 s, and 5 s only if feed timestamps justify those horizons.
- Depth imbalance at selected levels when depth data exists.
- Trade volume and signed order-flow imbalance with a documented signing method.
- Short-term volatility, liquidity regime, market session, halt/stale-data flags.
- Current position, cash, average entry price, unrealized P&L, recent turnover, and time since last trade.
- Feature normalization must be fit on training data only and versioned.

### Actions

Prefer a target inventory or bounded discrete action (buy, sell, hold, cancel) for the initial version. Position sizing, order type, order size, and time-in-force must be explicit. The risk layer imposes maximum position, per-order size, gross exposure, and trading-rate limits.

## 7. How agents learn

### Stage 1: rule-based and parameterized agents

Start with deterministic rules and tunable parameters. This validates the observation pipeline, action interface, simulator, metrics, and evolutionary process before neural training adds complexity.

### Stage 2: evolutionary search

1. Each agent is evaluated on the same chronological training windows and cost assumptions.
2. Invalid agents are disqualified for accounting bugs, illegal actions, leakage, or risk-limit breaches.
3. Rank eligible candidates using net performance, drawdown, turnover, stability, and diversity—not raw return alone.
4. Select up to five parents per generation. Prefer a diversity-aware or Pareto selection method so all parents are not near-clones.
5. Create 50 descendants: nominally ten per parent. Use a configurable mix of exact elite copies, small mutations, larger mutations, and occasional fresh immigrants. Make the counts sum to 50 and log the actual method.
6. Mutate strategy thresholds, feature subsets, hyperparameters, and neural weights where applicable. Crossover only compatible architectures.
7. Keep a champion archive and novelty archive. Never delete all record of a potentially useful prior strategy.
8. A generation is a full evaluation cycle across multiple sessions/windows; never choose survivors from one day’s return alone. Select up to five eligible parents, and have each produce exactly ten descendants for a 50-agent next generation. Descendants are usually mutated variants, not ten identical copies. Keep archived champions separate from the active 50-agent population. If fewer than five parents pass the eligibility gates, do not quietly duplicate weak candidates to fill slots; hold promotion and report the failure.

### Stage 3: PPO / reinforcement learning

- Implement PPO only after the simulator, rule-based evolutionary loop, and historical evaluation pipeline pass their exit gates. Until Phase 4, all 50 competing agents are non-RL agents; PPO candidates can be introduced in a later, explicitly versioned population.
- Collect experience from training episodes only; keep validation and final test data separate.
- Reward can start from log portfolio-value change after all simulated fills and costs: r_t = log(V_t / V_(t-1)).
- Optional penalties for turnover, inventory, or drawdown require careful scaling; avoid double-counting costs already reflected in portfolio value.
- Monitor reward hacking by comparing training reward with realized net P&L and action distributions.
- Compare PPO against non-RL evolutionary agents and simple baselines under identical conditions.

### What an agent is learning

Agents do not learn merely because the system tells them a profitable agent survived. In evolutionary search, parameter configurations are selected and mutated based on measured outcomes. In PPO, the policy is updated using rewards from simulated trajectories. Both can overfit the training environment; unseen data and realistic execution costs are essential.

## 8. Selection, fitness, and generation governance

Use eligibility gates before fitness ranking. Example research score:

J = w_R · R̃_net − w_D · D̃_max − w_V · Ṽ_downside − w_T · T̃_turnover

R_net is net return after costs; D_max is maximum drawdown; V_downside is a documented downside-risk measure; and T_turnover is turnover. The tilde indicates each metric has been transformed to a documented, dimensionless scale using fixed, predeclared anchors or a training-only transform. Do not mix raw percentages, volatility, and dollar turnover in one score. Freeze weights and transforms before candidate comparison, log them, and apply hard eligibility gates before ranking; high return must never offset a hard risk breach.

- Use hard limits for maximum drawdown, position, order size, turnover, stale data, and invalid behavior.
- Require minimum sample size/trading activity before ranking; label insufficient evidence rather than treating it as a winner.
- Use nested chronological evaluation: inner walk-forward folds may guide evolution, while an outer holdout remains untouched until strategy design and selection rules are frozen. Rotate or diversify validation windows and track every selection trial because repeatedly selecting winners on the same windows overfits them. Add pessimistic/base/optimistic cost scenarios and report uncertainty where feasible.
- Record selection reason, score components, parents, mutations, and rejected-candidate reasons.
- If no candidate meets the minimum gates, do not force a winner; preserve the current champion or run a new exploratory generation.

## 9. Simulator and order-fill model

| Component | Minimum requirement |
|---|---|
| Chronology | Deterministic event ordering; a decision cannot use a future-arriving quote |
| Spread | Buy/sell at plausible bid/ask or book-derived prices; no automatic midpoint fill |
| Latency | Separate feed receive, feature, decision, order submission, and simulated arrival times |
| Slippage | Configurable scenarios; include adverse price movement between decision and arrival |
| Limit orders | Partial fills and queue uncertainty; touching a limit does not guarantee execution |
| Liquidity | Constrain fills to plausible available liquidity and order size |
| Fees | Configurable broker/venue/regulatory costs as applicable |
| Portfolio ledger | Cash, shares, average cost, realized/unrealized P&L, fees, equity, exposure |
| Market events | Halt, open/close, early close, stale feed, symbol changes, splits/dividends as relevant |
| Audit | Every action, risk decision, order proposal, simulated fill, and metric is logged |

Use pessimistic/base/optimistic execution assumptions for sensitivity analysis. Base and pessimistic results should be emphasized; optimistic assumptions must not be presented as the main evidence.

## 10. Evaluation, baselines, and test discipline

| Metric | Required definition/reporting |
|---|---|
| Net return | After modeled fees, spread, slippage, and fill effects |
| Sharpe / Sortino | Document annualization and sampling assumptions |
| Maximum drawdown | Peak-to-trough decline in equity |
| Profit factor | Gross profit / gross loss with edge cases defined |
| Turnover | Define denominator and report total/average |
| Fill rate | Filled orders or quantity divided by a clearly defined submitted denominator |
| Net edge per trade | Mean and distribution after costs; include uncertainty |
| Exposure | Average/peak inventory and gross/net exposure |
| Latency | p50/p95/p99 measured local processing; separately document network/broker latency |
| Stability | By day, symbol, volatility regime, and cost scenario |
| Baselines | Cash/no-trade, buy-and-hold where relevant, simple momentum, simple mean reversion, randomized control |

- Use chronological training, validation, and final test splits; do not randomly shuffle time-series rows.
- Fit all normalization and feature-selection statistics on training data only.
- Use walk-forward testing across multiple time windows.
- Keep a final untouched test set until the design is frozen.
- Track every search and selection trial. Repeatedly choosing winners on the same validation period can overfit it; use inner walk-forward folds for selection and keep a separate outer holdout untouched until the design is frozen.
- For the initial scope, agents are long-only and cash-funded. Short selling, borrow availability/fees, leverage, margin, and options are out of scope until separately specified and tested.
- Point-in-time data is mandatory for credible historical claims: include delisted securities and historically correct universe membership when available, handle splits/dividends without double adjustment, and document missing historical records. If the data vendor cannot support this, report the limitation instead of claiming unbiased stock-level results.
- Do not treat every high-frequency trade as statistically independent; returns and fills can be highly correlated.

## 11. Runtime: where the agents run

### Development and training

- Run the first version locally on the developer machine. A GPU is optional for 50 lightweight agents; training size and neural architecture determine the need.
- Use a command-line orchestrator to run experiments, for example: `python -m atlas.cli simulate`, `python -m atlas.cli evolve`, `python -m atlas.cli evaluate`.
- Parallelize independent backtests only after deterministic behavior is proven. Each worker receives immutable data and a unique run ID.
- Store checkpoints and reports on disk; do not rely on in-memory state surviving a restart.

### Paper-trading runtime

- Run a separate process with its own config and logs; it consumes live market data but uses only the internal simulated broker.
- Freeze the selected champion model for a paper session. Do not update weights during a session unless a specific experiment is explicitly designed and labeled for that purpose.
- Initially run one champion plus a small shadow group for comparison. Keep all 50 agents in research evaluation; do not assume all 50 should trade live simultaneously.
- A paper candidate must be selected from frozen, versioned checkpoints and pass pre-session health checks.
- On disconnect, stale data, timestamp anomalies, process crash, or risk breach: stop proposing new orders, cancel simulated working orders, alert, and enter safe state.

## 12. Paper trading design

Paper trading means live market data enters the same feature and policy pipeline, but proposed orders go to an internal paper broker that simulates fills. It does not mean sending orders to Schwab and hoping they are harmless. Schwab’s developer sandbox, where available, is for API-method testing with sample data and must not be treated as the strategy’s fill simulator. The paper broker must be separate from any future live broker adapter.

- Paper broker supports order lifecycle states: proposed, accepted, partially filled, filled, cancelled, rejected, expired.
- Log market snapshot and quote age at decision, action, simulated order, fill assumption, fees, and resulting portfolio state.
- Simulate realistic order-arrival delay and use a documented fill model; distinguish simulated fills from actual executions.
- Daily reports include net P&L, benchmark, drawdown, turnover, fills, rejects, latency, feed gaps, and model/version identifiers.
- Paper mode has no live trading endpoint, no live-order credentials, and no route to convert a paper order into a real order.

## 13. Future Schwab integration and live-money gate

> **Important broker reality check**
>
> Schwab’s official developer portal separates Market Data Production from Accounts and Trading Production. It also documents a developer sandbox for testing supported API methods with engineered sample data and simulated OAuth; that is not automatically a realistic end-to-end brokerage paper-trading engine with live market conditions, exchange order routing, partial fills, or queue position. Do not mistake API-method sandbox responses for strategy paper trading. Confirm which sandbox methods and products are available for the specific Trader API app before relying on them. Keep Atlas paper fills simulated internally.

### Architecture now

Define a `MarketDataProvider` interface and a `BrokerExecution` interface, but implement only the historical provider, optional live-data provider, and `PaperBroker` initially. A future `SchwabBroker` adapter should be a separate module and should not be reachable from paper-mode code.

### Before live trading could be considered

- Verify current Schwab official API products, terms, OAuth requirements, market-data entitlements, account/trading permissions, rate limits, streaming limits, and supported order types. Confirm the specific product’s documented sandbox coverage; a general API test sandbox is not the same as a production-like paper broker.
- Confirm the API and account are appropriate for the intended strategy. Retail broker APIs may not support the latency or order throughput needed for true sub-second HFT.
- Complete an extended forward paper-trading period and a documented independent readiness review; one profitable month is not enough by itself.
- Use a separate live config and credentials store, default-off deployment flag, explicit human confirmation, and a separate release process.
- Require hard caps: maximum order size, position, gross exposure, daily loss, order rate, and allowed symbols; include a kill switch and manual broker-side recovery plan.
- Reconcile broker order/fill/account activity against internal state and handle duplicate submissions, timeouts, rejects, partial fills, and disconnect recovery safely.
- Start with minimal exposure only if approved; continuously monitor and stop on anomalies. Do not use money that cannot be lost.
- Review applicable legal, regulatory, tax, and broker terms before any automated live deployment.

The initial implementation must not place real orders. Phase 7 may create a disabled adapter but must not send production orders. A Phase 8 live activation requires explicit user approval, a current official API review, passing safety tests, and a separate release decision.

## 14. Risk controls

| Control | Expected behavior |
|---|---|
| Max position | Reject any action that would exceed per-symbol inventory limit |
| Max order size | Cap each proposed order before simulation |
| Max gross/net exposure | Reject or reduce action that exceeds portfolio limits |
| Daily loss / drawdown | Stop new trading actions when configured threshold is reached |
| Order-rate limit | Limit repeated orders/cancellations; avoid runaway loops |
| Stale/missing data | No new actions; enter safe state |
| Feed/API disconnect | Stop actions and alert; reconnect with bounded retries |
| Clock anomaly | Stop actions until event ordering is trustworthy |
| Model mismatch | Reject model if schema/version does not match expected features |
| Kill switch | Immediate stop of new actions; clear simulated working orders and log event |

Risk limits are configuration-driven, validated at startup, and cannot be changed by an agent. Any future live broker must enforce risk checks again immediately before order submission.

## 15. Repository structure

atlas-trading-ai/ ├── README.md ├── pyproject.toml ├── .env.example ├── .gitignore ├── docs/ │   └── PROJECT_SPECIFICATION.docx ├── configs/ │   ├── data.yaml │   ├── simulator.yaml │   ├── evolution.yaml │   ├── risk.yaml │   └── paper.yaml ├── data/ │   ├── raw/                 # never commit licensed market data │   ├── processed/ │   └── manifests/ ├── src/atlas/ │   ├── cli.py │   ├── data/                # ingestion, validation, features, storage │   ├── agents/              # interface, baselines, policies, population │   ├── environment/         # events, simulator, portfolio, execution │   ├── learning/            # PPO and training │   ├── evolution/           # fitness, selection, mutation, generations │   ├── evaluation/          # metrics, walk-forward, reports │   ├── paper/               # live data, paper broker, risk, monitor │   ├── brokers/             # interfaces only initially; no live adapter │   └── dashboard/ ├── experiments/ ├── artifacts/ │   ├── checkpoints/ │   └── reports/ └── tests/     ├── test_data_validation.py     ├── test_no_lookahead.py     ├── test_portfolio_accounting.py     ├── test_execution.py     ├── test_evolution.py     ├── test_reproducibility.py     ├── test_risk_controls.py     └── test_paper_cannot_trade_live.py

## 16. Configuration examples

Use YAML or TOML configuration, validate it on startup, and save a snapshot with every experiment. Keep the initial strategy long-only and cash-funded; market data, broker mode, and risk limits must be explicit. Example fields:

| Config | Fields to include |
|---|---|
| data.yaml | provider, symbols, timezone, feed, historical range, allowed data age, source/license notes |
| simulator.yaml | initial_cash, fees, spread model, slippage model, latency distributions, fill model, session calendar |
| evolution.yaml | population_size=50, parent_count=5, offspring_per_parent=10, generation_window, mutation rates, diversity settings, seed |
| risk.yaml | max_position, max_order_size, max_exposure, max_daily_loss, max_drawdown, max_order_rate, stale_data_ms |
| paper.yaml | mode=paper, live_orders_enabled=false, candidate_checkpoint, allowed_symbols, session schedule, alerts |

> **Fail-safe configuration**
>
> The paper command must refuse to start unless mode is explicitly paper, live_orders_enabled is false, and all required risk settings are present and valid. Research/simulation commands use their own non-broker configuration. Any missing or invalid safety setting fails closed. The initial codebase must not contain a functional live-order implementation.

## 17. Testing requirements

- Deterministic replay with same data/config/seed.
- No-lookahead tests using intentionally future-shifted values and timestamp assertions.
- Portfolio accounting reconciles cash, positions, fills, fees, and equity.
- Limit order does not fill solely because last price touches limit.
- Stale feed, missing quote, duplicate event, out-of-order event, halt, and reconnect behavior.
- Risk manager rejects invalid/oversized orders and stops on loss limit.
- Evolution always yields exactly 50 agents, logs lineage, and resumes from checkpoint.
- No mutable state leakage between agents.
- Paper mode cannot instantiate a live broker or call a live-order endpoint.
- API credentials are never required for tests; secrets are not logged or committed.
- Integration test for process shutdown and safe restart.

## 18. Six-week prototype build plan (estimate)

| Timeline | Deliverables | Exit gate |
|---|---|---|
| Week 1 | Repo, synthetic data, event models, deterministic simulator, portfolio ledger, cost models, tests | Accounting and chronology tests pass |
| Week 2 | 50-agent non-RL population, baselines, fitness scaling, mutation, selection, lineage, checkpointing | One reproducible generation completes |
| Week 3 | Historical-data adapter, data-quality reports, chronological splits, nested walk-forward evaluation, execution stress tests | No lookahead; final holdout remains untouched; performance report includes limitations |
| Week 4 | Optional PPO family, reward validation, training stability and inference-latency tests | PPO compared fairly with non-RL agents; no automatic promotion |
| Week 5 | Live-data adapter, internal paper broker, risk manager, dashboard, alerts, forward-test launch | Paper mode works without any live-order path; safety integration tests pass |
| Week 6 | Operational hardening, failure drills, results review, candidate-readiness report | Open issues documented; readiness decision kept separate from live activation |

This six-week schedule is a planning estimate, not a deadline or a profitability timeline. It may extend substantially depending on market-data access, data quality, fill-model complexity, hardware, and failed tests. Do not compress or skip an exit gate to meet a calendar date.

## 19. CLI commands to implement

| Command | Purpose |
|---|---|
| `python -m atlas.cli validate-data --config configs/data.yaml` | Validate data schema, timestamps, gaps, and manifest |
| `python -m atlas.cli simulate --config configs/simulator.yaml` | Run deterministic historical replay |
| `python -m atlas.cli evolve --config configs/evolution.yaml` | Run one or more evolutionary generations |
| `python -m atlas.cli evaluate --checkpoint <path>` | Run walk-forward and stress evaluations |
| `python -m atlas.cli paper --config configs/paper.yaml` | Run paper mode only; simulated orders only |
| `python -m atlas.cli report --run-id <id>` | Build a report for an experiment |

Names may be adjusted, but the README must document equivalent runnable commands.

## 20. Claude Code Phase 1 prompt

> You are the senior engineer implementing Project Atlas. Read docs/PROJECT_SPECIFICATION.docx fully before coding.
>
> Start with PHASE 1 ONLY. First inspect the repository and report existing files. Do not overwrite useful work.
>
> Implement:
>
> - Python package skeleton, pyproject.toml, README, .gitignore, and typed configuration models.
>
> - Typed timestamped quote/trade events, including source and receive timestamps where available.
>
> - Synthetic data generator that needs no API key.
>
> - Deterministic chronological replay engine.
>
> - Portfolio ledger for cash, positions, fills, fees, realized/unrealized P&L, and equity.
>
> - Configurable spread, fee, slippage, and latency assumptions.
>
> - Validation for duplicate, out-of-order, stale, missing, and impossible events.
>
> - CLI demo for a small synthetic replay.
>
> - pytest tests for accounting, deterministic replay, event chronology, invalid data, and no lookahead.
>
> Mandatory constraints:
>
> - No RL yet.
>
> - No broker integration yet.
>
> - No live trading code or live-order endpoint.
>
> - Do not require credentials.
>
> - Use type hints, useful errors, and structured logs.
>
> - Run the tests, fix failures, and report the exact test command/results.
>
> - Update README with setup, run, test, and limitations.
>
> Do not start Phase 2 until the user approves the Phase 1 summary.

## 21. Later-phase prompts

### Phase 2 — Agents and evolution

Implement the common agent interface and 50 initial non-RL agents across the five initial families (10 per family). Keep fixed evaluation baselines outside the competing population. Add fitness components with fixed scaling, risk eligibility gates, five-parent selection, ten descendants per parent, mostly mutated offspring, diversity preservation, lineage, checkpoint/resume, and reproducible experiment reports. Do not implement PPO yet; reserve a clean interface so PPO can join a versioned population after Phase 4. Use synthetic data for plumbing tests; do not make profitability claims.

### Phase 3 — Historical data and evaluation

Add a configurable historical data adapter, data manifests, causal features, point-in-time universe/corporate-action handling where data permits, chronological splits, nested walk-forward evaluation, execution cost scenarios, and complete metrics. Keep the final outer holdout untouched. Test lookahead and repeated-validation overfit. Compare every agent with fixed baselines and report data/coverage limitations.

### Phase 4 — PPO

Only after Phases 1–3 exit gates pass, add the Gymnasium environment and PPO as an optional, versioned policy family. Train only on training windows. Keep selection folds separate from the final outer test. Compare PPO and non-RL agents with identical execution costs. Track reward hacking, action distributions, training stability, and inference latency. Do not promote PPO automatically because it is newer or more complex.

### Phase 5 — Paper trading

Add a live-data provider adapter and an internal paper broker. No real order endpoint or live-order credentials may be used. Add order lifecycle simulation, stale-data protection, risk controls, audit logs, dashboard, kill switch, and integration tests proving paper mode cannot trade live.

### Phase 6 — Readiness review only

Generate a readiness report covering out-of-sample performance, data fidelity, cost sensitivity, paper results, risk, operational reliability, and open issues. Do not implement live trading in this phase. Present unresolved broker API terms/capabilities and wait for explicit approval.

### Phase 7 — Disabled future broker adapter (new approval required)

Only after explicit approval and current official API documentation are supplied, design a separate broker adapter with secure OAuth/token handling, order preview where available, idempotent order submission, order/fill reconciliation, rate limits, strict risk gates, default-off configuration, audit logs, and kill switch. In Phase 7, test only with mocks and officially documented non-production sandbox capabilities; do not submit production orders. Keep PaperBroker unchanged. Any real-money activation is a separate Phase 8 release decision with explicit approval, operational runbook, limited exposure, and verified broker support. Do not claim retail API latency is suitable for HFT without measurement.

### Phase 8 — Real-money activation (not included in this build; new explicit approval required)

This document does not authorize live trading. Before any live orders are enabled, require a separate deployment plan; documented evidence from the outer holdout and extended forward paper trading; broker/account permissions confirmed from current official documentation; independent risk review; secure credential/token handling; monitored startup with minimal approved exposure; broker-side kill/recovery procedure; reconciliation tests; alerting; and an explicit, informed human release approval. If any gate is missing, live orders remain disabled.

## 22. Definition of done

- 50 agents can be evaluated and evolved reproducibly.
- The simulator is deterministic and its accounting is tested.
- Agents are selected based on documented risk-adjusted criteria and unseen validation windows.
- The system records how each agent was initialized, trained, selected, mutated, and promoted.
- Reports compare agents with baselines after realistic execution costs.
- Paper trading uses live market data but internal simulated orders only.
- Paper mode is technically unable to place live orders.
- The project clearly states data, latency, and fill-model limitations.
- No real-money trading is enabled without a separate approved phase and readiness review.

> **Success definition**
>
> Atlas is successful as a research platform when it reproducibly evolves agents, evaluates them fairly, survives failure scenarios, and provides honest evidence about out-of-sample performance. Profitability is an empirical question, not a software acceptance criterion.

## 23. Official references to re-check before broker work

Verify current documentation at the time of implementation; pages and permissions can change. Checked for this revision on October 26, 2026.

- Schwab Trader API — Individual (products, market data, accounts/trading, order entry, order preview): https://developer.schwab.com/products/trader-api--individual
- Schwab Developer Portal — Test in Sandbox (documented API-method sandbox and sample data): https://developer.schwab.com/user-guides/apis-and-apps/test-in-sandbox
- Schwab Developer Portal — OAuth authentication: https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth
- Schwab Developer Portal — Become an Individual Developer: https://developer.schwab.com/user-guides/individual-developer/become-individual-developer
