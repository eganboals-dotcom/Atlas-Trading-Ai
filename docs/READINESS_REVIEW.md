# Project Atlas — readiness review

Generated 2026-10-27T01:38:03Z · atlas 0.1.0

## Verdict: **NOT READY**

> This verdict is a recommendation for a human decision. It enables nothing. Any broker adapter (Phase 7) or real-money activation (Phase 8) needs separate explicit approval.

Required gates: 4 pass, 9 fail, 5 missing, 4 pending sign-off.

**Live trading is not implemented and remains disabled.** Nothing in this review changes that.

## Gates

| Status | Area | Criterion | Evidence |
|---|---|---|---|
| FAIL | Data fidelity | Research uses real, licensed historical market data | provider synthetic, timestamps synthetic |
| FAIL | Data fidelity | Exchange/vendor timestamps with measured precision and feed latency | timestamp provenance: synthetic; precision and latency not yet measured on a real feed |
| MISSING | Data fidelity | Point-in-time universe including delisted symbols | fixed configured symbols |
| FAIL | Out-of-sample | base costs: champion OOS mean-return CI lower bound > +0.00 bps | mean -0.39 bps, CI [-0.73 bps, -0.02 bps], n=12 |
| FAIL | Cost sensitivity | pessimistic costs: champion OOS mean-return CI lower bound > +0.00 bps | mean -2.92 bps, CI [-3.78 bps, -2.17 bps], n=12 |
| FAIL | Out-of-sample | Champion beats the cash baseline out of sample (base costs) | champion -0.39 bps vs cash +0.00 bps |
| FAIL | Out-of-sample | At least 60 out-of-sample sessions | 12 validation sessions |
| PASS | Out-of-sample | Selection gap ≤ 1.0 bps, trials counted | gap +0.26 bps; 4800 agent-session evaluations |
| MISSING | Outer holdout | Frozen design evaluated once on the untouched outer holdout | holdout not evaluated for wf_synthetic_v1-20261027T013509Z |
| FAIL | Paper trading | Paper sessions ran on a live market-data feed | 1 session(s), 0 on a live feed (others were replays) |
| FAIL | Paper trading | At least 20 forward paper sessions on live data | 0 live-data session(s) |
| FAIL | Paper trading | Champion net P&L positive over the paper period | total champion net P&L -53.83 over 1 session(s); beat benchmark in 1 |
| PASS | Risk | No unexplained risk halts or kill-switch events in paper sessions | halts none; kills none |
| PASS | Operational reliability | All failure drills pass | 9/9 passed |
| PASS | Operational reliability | Full automated test suite passes | 218 passed in 23.59s |
| PASS (info) | Operational reliability | Local p99 decision latency well below the decision interval | max p99 6 µs |
| MISSING | Operational reliability | Network and broker order round-trip latency measured | no broker connection exists (by design until Phase 7) |
| MISSING | Operational reliability | Vendor live-feed adapter with heartbeats/timeouts implemented and tested | only the replay feed exists; vendor streaming specs unverified |
| MISSING | Broker | Broker API products, permissions, order types, and limits verified from current official documentation | 10 open items (see below) |
| PENDING | Governance | Broker terms and data-use review | no sign-off recorded |
| PENDING | Governance | Independent risk review | no sign-off recorded |
| PENDING | Governance | Legal, regulatory, and tax review | no sign-off recorded |
| PENDING | Governance | Operational runbook and broker-side recovery plan reviewed | no sign-off recorded |

## Out-of-sample performance and cost sensitivity

Walk-forward run `wf_synthetic_v1-20261027T013509Z`. Champion mean per-session net return on validation sessions:

| Scenario | Mean | 90% CI |
|---|---|---|
| optimistic | +0.10 bps | [-0.24 bps, +0.52 bps] |
| base | -0.39 bps | [-0.73 bps, -0.02 bps] |
| pessimistic | -2.92 bps | [-3.78 bps, -2.17 bps] |

Baselines (base): cash +0.00 bps, buy & hold -12.03 bps. In-sample -0.13 bps vs out-of-sample -0.39 bps (selection gap +0.26 bps).

PPO (`wf_synthetic_ppo_v1-20261027T010441Z`): OOS base -2.72 bps, pessimistic -3.47 bps; beat the evolution champion in 0/6 folds (base). Not promoted. The final candidate comes from evolution; PPO is never promoted automatically.

## Paper trading

| Session | Feed | Account | Net P&L | vs benchmark | Halted |
|---|---|---|---|---|---|
| paper_2026-10-19 | replay | champion | -53.83 | +18.82 bps | no |
| paper_2026-10-19 | replay | shadow-1 | -53.29 | +18.87 bps | no |
| paper_2026-10-19 | replay | shadow-2 | -341.21 | -9.92 bps | no |
| paper_2026-10-19 | replay | benchmark | -241.99 | +0.00 bps | no |

## Failure drills

| Drill | Scenario | Expected | Observed | Result |
|---|---|---|---|---|
| baseline | clean session | completes, no kill switch | 23555 events, kill switch None | PASS |
| disconnect_recovers | feed drops at event 1000; 2 reconnects fail | safe state, recovers on attempt 3, session continues | disconnects 1, reconnect attempts 3, kill None | PASS |
| reconnect_exhausted | feed drops and never comes back | kill switch after 3 attempts; session stops | kill switch: feed lost: 3 reconnect attempts failed; events processed 1000 | PASS |
| stale_feed | 10 s of total feed silence at event 2000 | safe state from the moment data went stale; no orders until fresh data | 1 stale safe-state periods, longest 8151 ms, orders during them: 0 | PASS |
| clock_anomaly | local clock runs backwards; an event is stamped 10 s in the future | both detected; safe state until ordering is trustworthy again | clock anomalies 2, clock safe-state periods 2 | PASS |
| crash_and_resume | process dies at event 2345 (last state flush at 2000), then --resume | ledgers restored, in-flight orders cancelled, no events skipped or double-logged | restarts 1, events 23555 (clean run 23555), duplicated order records 0 | PASS |
| manual_kill | operator writes the KILL file mid-session | no new orders, working orders cancelled, restart refused until reset | kill: manual: readiness drill; orders after kill 0; restart refused without reset: True | PASS |
| daily_loss_halt | daily-loss limit tightened to 0.01% | breaching accounts halt; no orders after the halt | halted accounts ['benchmark', 'champion', 'shadow-1', 'shadow-2']; orders after halt 0 | PASS |
| live_config_refused | config with live_orders_enabled: true, mode: live, or an api_key | paper command refuses to start | unsafe configs refused: 3/3 | PASS |

Test suite: 218 passed in 23.59s (exit code 0).

## Unresolved broker and API items

Each must be answered from the broker's **current official documentation** (and, where noted, by measurement) before any adapter is designed. None is assumed.

| Item | Question | Reference |
|---|---|---|
| API product access | Is the Trader API (Individual) approved for this account, covering both the Market Data Production and the Accounts and Trading Production products? Access to one does not imply the other. | https://developer.schwab.com/products/trader-api--individual |
| App approval and OAuth | App registration status, the OAuth authorization-code flow, access and refresh token lifetimes, re-authorization cadence, and secure token storage requirements. | https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth |
| Account permissions | Which order types, instruments, and account types are enabled. Initial scope is long-only, cash-funded US ETFs, with no margin, shorting, or options. | https://developer.schwab.com/products/trader-api--individual |
| Supported order types and time-in-force | Is IOC (or an equivalent) supported? What limit/market semantics, order preview, and replace/cancel behavior apply? Atlas assumes IOC fills. | https://developer.schwab.com/products/trader-api--individual |
| Rate limits | Request and order-submission limits per minute, and the throttling and error behavior. These bound order frequency and may rule out sub-second strategies. | https://developer.schwab.com/products/trader-api--individual |
| Market-data entitlements and streaming | Real-time vs delayed Level 1 data, symbol and connection limits, message formats, heartbeat behavior, and the precision and semantics of exchange, vendor, and local timestamps. | https://developer.schwab.com/products/trader-api--individual |
| Sandbox coverage | Which API methods the developer sandbox supports for this app. It returns engineered sample data with simulated OAuth, so it is not a realistic paper-trading fill simulator. | https://developer.schwab.com/user-guides/apis-and-apps/test-in-sandbox |
| Terms of use | Whether the terms permit automated trading by this account holder, and any restrictions on data redistribution, storage, and derived use. | https://developer.schwab.com/user-guides/individual-developer/become-individual-developer |
| Latency and throughput | Measured order round-trip latency and fill-report latency. No retail API should be assumed suitable for HFT without measurement. | measurement required |
| Reconciliation and recovery | How fills, orders, and positions can be queried for reconciliation; duplicate-submission and idempotency behavior; and broker-side manual kill and recovery procedures. | https://developer.schwab.com/products/trader-api--individual |

## Open issues

* [FAIL] Data fidelity: Research uses real, licensed historical market data. provider synthetic, timestamps synthetic
* [FAIL] Data fidelity: Exchange/vendor timestamps with measured precision and feed latency. timestamp provenance: synthetic; precision and latency not yet measured on a real feed
* [MISSING] Data fidelity: Point-in-time universe including delisted symbols. fixed configured symbols
* [FAIL] Out-of-sample: base costs: champion OOS mean-return CI lower bound > +0.00 bps. mean -0.39 bps, CI [-0.73 bps, -0.02 bps], n=12
* [FAIL] Cost sensitivity: pessimistic costs: champion OOS mean-return CI lower bound > +0.00 bps. mean -2.92 bps, CI [-3.78 bps, -2.17 bps], n=12
* [FAIL] Out-of-sample: Champion beats the cash baseline out of sample (base costs). champion -0.39 bps vs cash +0.00 bps
* [FAIL] Out-of-sample: At least 60 out-of-sample sessions. 12 validation sessions
* [MISSING] Outer holdout: Frozen design evaluated once on the untouched outer holdout. holdout not evaluated for wf_synthetic_v1-20261027T013509Z
* [FAIL] Paper trading: Paper sessions ran on a live market-data feed. 1 session(s), 0 on a live feed (others were replays)
* [FAIL] Paper trading: At least 20 forward paper sessions on live data. 0 live-data session(s)
* [FAIL] Paper trading: Champion net P&L positive over the paper period. total champion net P&L -53.83 over 1 session(s); beat benchmark in 1
* [MISSING] Operational reliability: Network and broker order round-trip latency measured. no broker connection exists (by design until Phase 7)
* [MISSING] Operational reliability: Vendor live-feed adapter with heartbeats/timeouts implemented and tested. only the replay feed exists; vendor streaming specs unverified
* [MISSING] Broker: Broker API products, permissions, order types, and limits verified from current official documentation. 10 open items (see below)
* [PENDING] Governance: Broker terms and data-use review. no sign-off recorded
* [PENDING] Governance: Independent risk review. no sign-off recorded
* [PENDING] Governance: Legal, regulatory, and tax review. no sign-off recorded
* [PENDING] Governance: Operational runbook and broker-side recovery plan reviewed. no sign-off recorded
* [INFO] PPO family: OOS base -2.72 bps; Not promoted. The final candidate comes from evolution; PPO is never promoted automatically.
* [INFO] The risk manager runs in paper mode only; backtests use agent limits and fitness gates.
* [INFO] Session calendars do not model holidays, early closes, or halts.
* [INFO] Fill model: IOC against top of book only; no queue position, depth, or market impact beyond fixed slippage.

## What would change the verdict

1. Import licensed historical data, with exchange/vendor timestamps and point-in-time coverage, and rerun the walk-forward evaluation on it.
2. Find a design whose out-of-sample confidence interval clears zero under base **and** pessimistic costs on enough sessions. Then freeze it and evaluate the outer holdout exactly once.
3. Build and test a vendor live-feed adapter with heartbeats, then run an extended forward paper-trading period on live data.
4. Answer every broker item above from official documentation, and record the independent risk, legal/regulatory/tax, broker-terms, and runbook sign-offs.
5. Only then request explicit approval for Phase 7 (a disabled adapter, tested against mocks and the documented sandbox). Phase 8 (real money) is a separate decision.
