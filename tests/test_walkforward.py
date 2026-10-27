from __future__ import annotations

import json
import statistics
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from helpers import ScriptedStrategy, data_config, quote, sim_config

from atlas.agents.base import AgentRecord
from atlas.agents.baselines import baseline_records
from atlas.agents.genome import sample_genome
from atlas.agents.registry import FAMILY_CLASSES
from atlas.config import AgentLimits, EvolutionConfig
from atlas.data import providers as providers_mod
from atlas.data.providers import SyntheticProvider
from atlas.environment.orders import OrderIntent, Side
from atlas.environment.replay import ReplayEngine
from atlas.evaluation.holdout import HoldoutAccessError, HoldoutGuard, HoldoutLedger
from atlas.evaluation.metrics import minute_returns, run_metrics, sharpe_sortino
from atlas.evaluation.scenarios import DEFAULT_SCENARIOS, apply_scenario
from atlas.evaluation.splits import SplitConfig, SplitError, plan_splits
from atlas.evaluation.walkforward import WalkForwardConfig, evaluate_holdout, run_walkforward
from atlas.evolution.evaluation import evaluate_records

DAYS = [date(2026, 6, d) for d in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16)]


# ------------------------------------------------------------------ splits


def test_rolling_splits_are_chronological_with_embargo_and_holdout() -> None:
    cfg = SplitConfig(holdout_sessions=2, train_sessions=3, validation_sessions=2, step_sessions=2,
                      embargo_sessions=1)
    plan = plan_splits(list(reversed(DAYS)), cfg)  # input order must not matter
    assert plan.holdout == tuple(DAYS[-2:])
    assert max(plan.development) < min(plan.holdout)
    assert len(plan.folds) == 3
    for f in plan.folds:
        assert len(f.train) == 3 and len(f.validation) == 2
        gap = DAYS.index(f.validation[0]) - DAYS.index(f.train[-1])
        assert gap == 2  # one embargo session in between
        assert not set(f.train) & set(plan.holdout) and not set(f.validation) & set(plan.holdout)
    assert [f.validation[0] for f in plan.folds] == [DAYS[4], DAYS[6], DAYS[8]]  # rotating validation


def test_expanding_splits_grow_training() -> None:
    cfg = SplitConfig(holdout_sessions=2, train_sessions=2, validation_sessions=1, step_sessions=2,
                      scheme="expanding")
    plan = plan_splits(DAYS, cfg)
    assert all(f.train[0] == DAYS[0] for f in plan.folds)
    assert [len(f.train) for f in plan.folds] == [2, 4, 6, 8]


def test_split_errors() -> None:
    with pytest.raises(SplitError):
        plan_splits(DAYS[:3], SplitConfig(holdout_sessions=3, train_sessions=1, validation_sessions=1,
                                          step_sessions=1))
    with pytest.raises(SplitError):
        plan_splits(DAYS, SplitConfig(holdout_sessions=2, train_sessions=9, validation_sessions=2,
                                      step_sessions=1))
    with pytest.raises(SplitError, match="duplicate"):
        plan_splits(DAYS + DAYS[:1], SplitConfig(holdout_sessions=2, train_sessions=2, validation_sessions=1,
                                                 step_sessions=1))


# ------------------------------------------------------------------ holdout


def test_holdout_guard_and_ledger(tmp_path: Path) -> None:
    p = SyntheticProvider(data_config(), DAYS[0], len(DAYS), duration_s=60)
    guard = HoldoutGuard(p, tuple(DAYS[-2:]))
    assert len(guard.refs(DAYS[:3])) == 3
    with pytest.raises(HoldoutAccessError):
        guard.refs([DAYS[0], DAYS[-1]])

    ledger = HoldoutLedger(tmp_path / "ledger.jsonl")
    assert len(guard.unlock(ledger, "ds", "design-A", "test")) == 2
    guard.unlock(ledger, "ds", "design-A", "same design again is allowed")
    with pytest.raises(HoldoutAccessError, match="already used"):
        guard.unlock(ledger, "ds", "design-B", "a new design")
    guard.unlock(ledger, "ds", "design-B", "explicit reuse", allow_reuse=True)
    entries = ledger.entries()
    assert len(entries) == 3 and entries[-1]["reuse"] is True
    guard.unlock(ledger, "other-dataset", "design-B", "different dataset is independent")


# --------------------------------------------------------------- scenarios


def test_scenarios_change_costs_as_declared() -> None:
    base = sim_config(slippage={"bps": 0.5}, fees={"sell_regulatory_bps": 1.0})
    pess = apply_scenario(base, DEFAULT_SCENARIOS["pessimistic"])
    assert pess.slippage.bps == 2.0 and pess.spread.widen_bps == 1.0
    assert pess.fees.sell_regulatory_bps == pytest.approx(1.5)
    assert pess.fill.max_participation == 0.25
    assert apply_scenario(base, DEFAULT_SCENARIOS["base"]) == base


def test_pessimistic_costs_hurt_a_trading_baseline() -> None:
    p = SyntheticProvider(data_config(), DAYS[0], 2, duration_s=120, symbols=["SPY"])
    rec = [r for r in baseline_records() if r.family == "baseline_random"]
    sim = sim_config(stale_quote_ms=2_000.0)
    base = evaluate_records(rec, p.refs(), sim, AgentLimits(), 500)[0]
    pess = evaluate_records(rec, p.refs(), apply_scenario(sim, DEFAULT_SCENARIOS["pessimistic"]), AgentLimits(), 500)[0]
    assert sum(m.net_return for m in pess.windows) < sum(m.net_return for m in base.windows)


# ------------------------------------------------------------------ metrics


def test_round_trip_profit_factor_and_fill_rate() -> None:
    events = [quote(0), quote(100), quote(200, 101.00, 101.02), quote(300, 101.00, 101.02),
              quote(400, 99.00, 99.02), quote(500, 99.00, 99.02)]
    script = [(0, OrderIntent("SPY", Side.BUY, 10)), (150, OrderIntent("SPY", Side.SELL, 10)),
              (250, OrderIntent("SPY", Side.BUY, 10)), (450, OrderIntent("SPY", Side.SELL, 10))]
    result = ReplayEngine(sim_config(), ["SPY"]).run(events, ScriptedStrategy(script))
    m = run_metrics(result, 100_000)
    assert m.round_trips == 2 and m.losing_trades == 1
    win, loss = (101.00 - 100.02) * 10, (99.00 - 101.02) * 10
    assert m.profit_factor == pytest.approx(win / -loss)
    assert m.win_rate == 0.5 and m.fill_rate == 1.0
    assert m.symbol_pnl["SPY"] == pytest.approx(win + loss)


def test_sharpe_none_for_flat_equity_and_minute_sampling() -> None:
    assert sharpe_sortino([0.0, 0.0, 0.0]) == (None, None)
    curve = [(i * 1_000_000_000, 100.0 + i) for i in range(0, 181)]  # 3 minutes of 1 s samples
    assert len(minute_returns(curve, 100.0)) == 4


# --------------------------------------------------- repeated-validation overfit


def test_best_of_many_on_validation_overstates_fresh_performance() -> None:
    """Selecting the best of many random agents on few sessions fits noise: on fresh sessions the
    same agent does worse. This is why selection must never touch the final test set."""
    p = SyntheticProvider(data_config(), DAYS[0], 8, duration_s=120, symbols=["SPY"])
    refs = p.refs()
    select_on, fresh = refs[:2], refs[2:]
    rng = np.random.default_rng(0)
    space = FAMILY_CLASSES["mean_reversion"].param_space
    recs = [AgentRecord(f"r{i}", "mean_reversion", 0, sample_genome(space, rng)) for i in range(40)]
    sim = sim_config(stale_quote_ms=2_000.0)
    sel = evaluate_records(recs, select_on, sim, AgentLimits(), 500)
    traded = [e for e in sel if sum(m.fills for m in e.windows) >= 4]
    best = max(traded, key=lambda e: statistics.fmean(m.net_return for m in e.windows))
    best_in_sample = statistics.fmean(m.net_return for m in best.windows)
    best_rec = next(r for r in recs if r.agent_id == best.agent_id)
    out = evaluate_records([best_rec], fresh, sim, AgentLimits(), 500)[0]
    assert statistics.fmean(m.net_return for m in out.windows) < best_in_sample


# ---------------------------------------------------------------- end to end


def _wf_setup(tmp_path: Path) -> tuple[WalkForwardConfig, EvolutionConfig, Any, Any, SyntheticProvider]:
    evo = EvolutionConfig.model_validate({
        "experiment_name": "wf_test", "seed": 3, "generations": 1, "simulator_config": "x", "data_config": "x",
        "evaluation": {"windows": [{"session_date": "2026-06-01", "seed": 1}], "duration_s": 60,
                       "symbols": ["SPY"], "workers": 1},
        "fitness": {"gates": {"min_fills": 2, "min_windows_traded": 1}},
        "on_insufficient_parents": "explore",
    })
    wf = WalkForwardConfig.model_validate({
        "experiment_name": "wf_test", "evolution_config": "x",
        "data_source": {"kind": "synthetic", "data_config": "x", "start_date": "2026-06-01", "n_sessions": 8},
        "splits": {"holdout_sessions": 2, "train_sessions": 2, "validation_sessions": 1, "step_sessions": 2,
                   "embargo_sessions": 0},
        "generations_per_fold": 1, "candidates_per_fold": 2,
        "bootstrap": {"samples": 200},
        "reports_dir": str(tmp_path / "wf"), "holdout_ledger": str(tmp_path / "ledger.jsonl"),
    })
    provider = SyntheticProvider(data_config(), date(2026, 6, 1), 8, duration_s=60, symbols=["SPY"])
    return wf, evo, sim_config(stale_quote_ms=2_000.0), data_config(), provider


def test_walkforward_never_reads_holdout_then_holdout_evaluates_once(tmp_path: Path, monkeypatch) -> None:
    wf, evo, sim, base, provider = _wf_setup(tmp_path)
    loaded: list[date] = []
    real = providers_mod.load_raw

    def spy(ref):
        loaded.append(ref.session_date)
        return real(ref)

    providers_mod._load_validated.cache_clear()
    monkeypatch.setattr(providers_mod, "load_raw", spy)
    report, run_dir = run_walkforward(wf, evo, sim, base, provider)
    holdout = {date.fromisoformat(d) for d in report["holdout"]["sessions"]}
    assert loaded and not set(loaded) & holdout
    assert report["holdout"]["status"] == "untouched by this run"
    for f in report["folds"]:
        assert max(f["train"]) < min(f["validation"])
        for c in f["candidates"]:
            assert set(c["out_of_sample"]) == {"optimistic", "base", "pessimistic"}
        assert set(f["baselines"]) == {r.agent_id for r in baseline_records()}
    assert report["selection_trials"]["agent_session_evaluations"] > 0
    assert (run_dir / "report.md").read_text(encoding="utf-8").startswith("# Walk-forward report")
    assert any("synthetic" in x.lower() for x in report["limitations"])

    out = evaluate_holdout(run_dir, wf, evo, sim, base, provider)
    assert {date.fromisoformat(d) for d in out["holdout_sessions"]} == holdout
    assert out["holdout_reused"] is False
    assert "Outer holdout" in (run_dir / "report.md").read_text(encoding="utf-8")

    changed = evo.model_copy(update={"seed": 4})
    with pytest.raises(ValueError, match="frozen design"):
        evaluate_holdout(run_dir, wf, changed, sim, base, provider)
    ledger = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert ledger["design_fingerprint"] == report["design_fingerprint"]
