from __future__ import annotations

import dataclasses
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from helpers import ScriptedStrategy, data_config, sim_config  # noqa: E402

from atlas.agents.base import AgentRecord  # noqa: E402
from atlas.config import AgentLimits  # noqa: E402
from atlas.data.events import Quote  # noqa: E402
from atlas.data.providers import SessionRef, SyntheticProvider, load_raw  # noqa: E402
from atlas.environment.orders import OrderIntent, Side  # noqa: E402
from atlas.environment.replay import ReplayEngine  # noqa: E402
from atlas.evolution.evaluation import evaluate_records  # noqa: E402
from atlas.learning.env import AtlasTradingEnv  # noqa: E402
from atlas.learning.observation import SYMBOL_FEATURES, FeatureNormalizer, fit_normalizer  # noqa: E402
from atlas.learning.policy import PolicyAgent, load_policy  # noqa: E402
from atlas.learning.ppo import train_ppo  # noqa: E402
from atlas.learning.settings import PPOSettings  # noqa: E402
from atlas.timeutils import NS_PER_S  # noqa: E402

LIMITS = AgentLimits()
SIM = sim_config(stale_quote_ms=2_000.0)
TINY = PPOSettings(enabled=True, total_steps=512, rollout_steps=256, minibatch=64, epochs=2, hidden=16)


def refs(n: int = 3, duration: float = 60.0) -> list[SessionRef]:
    return SyntheticProvider(data_config(), date(2026, 6, 1), n, duration_s=duration, symbols=["SPY", "QQQ"]).refs()


@pytest.fixture(scope="module")
def normalizer() -> FeatureNormalizer:
    return fit_normalizer(refs(2), 500, 2_000)


def make_env(normalizer, sessions=None, **kw) -> AtlasTradingEnv:
    return AtlasTradingEnv(sessions or refs(2), SIM, LIMITS, 500, normalizer, size_frac=0.2, **kw)


def test_iter_run_matches_callback_run() -> None:
    from atlas.data.synthetic import generate_events
    from atlas.data.validation import validate_events

    events = validate_events(generate_events(data_config()), 2_000).valid_events  # helper's session day
    script =[(1_000, OrderIntent("SPY", Side.BUY, 50)), (30_000, OrderIntent("SPY", Side.SELL, 50))]
    a = ReplayEngine(SIM, ["SPY", "QQQ"]).run(events, ScriptedStrategy(script))
    strat = ScriptedStrategy(script)
    steps = ReplayEngine(SIM, ["SPY", "QQQ"]).iter_run(events, strat.on_order_update)
    market, state = next(steps)
    try:
        while True:
            market, state = steps.send(strat.on_market(market, state))
    except StopIteration as done:
        b = done.value
    assert a.fills and a.digest == b.digest


def test_normalizer_is_fit_on_given_sessions_only(normalizer) -> None:
    assert normalizer.fit_sessions == ("2026-06-01", "2026-06-02")
    assert len(normalizer.mean) == len(SYMBOL_FEATURES)
    d = normalizer.to_dict()
    assert FeatureNormalizer.from_dict(d) == normalizer
    with pytest.raises(ValueError, match="schema"):
        FeatureNormalizer.from_dict({**d, "features": d["features"][:-1]})
    raw = np.full(len(SYMBOL_FEATURES), 1e9)
    out = normalizer.transform(raw)
    assert out.max() <= 5.0 + 1e-9  # clipped (flags pass through unscaled but are 0/1 in practice)


def test_env_spaces_and_reward_telescopes_to_adjusted_log_return(normalizer) -> None:
    env = make_env(normalizer)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    rng = np.random.default_rng(0)
    total, done = 0.0, False
    while not done:
        obs, r, done, _, info = env.step(rng.integers(0, 3, size=2))
        total += r
    assert info["fills"] > 0
    assert total / env.reward_scale == pytest.approx(info["adjusted_log_return"], abs=1e-9)
    assert info["adjusted_log_return"] <= info["realized_log_return"] + 1e-12  # exit cost only ever subtracts


def test_env_without_exit_cost_rewards_realized_return(normalizer) -> None:
    env = make_env(normalizer, terminal_exit_cost=False)
    env.reset(seed=2)
    total, done = 0.0, False
    while not done:
        _, r, done, _, info = env.step(np.array([2, 2]))
        total += r
    assert total / env.reward_scale == pytest.approx(info["realized_log_return"], abs=1e-9)


def test_env_cycles_each_session_once_per_pass(normalizer) -> None:
    sessions = refs(3)
    env = make_env(normalizer, sessions=sessions)
    seen = []
    for k in range(3):
        _, info = env.reset(seed=5) if k == 0 else env.reset()
        seen.append(info["session"])
    assert sorted(seen) == [r.session_date.isoformat() for r in sessions]


def test_env_observations_do_not_depend_on_future_data(normalizer) -> None:
    ref = refs(1, duration=120)[0]
    events = load_raw(ref)
    cutoff = events[0].event_ts + 60 * NS_PER_S

    def shift(e):
        if e.event_ts < cutoff:
            return e
        if isinstance(e, Quote):
            return dataclasses.replace(e, bid=e.bid * 3, ask=e.ask * 3)
        return dataclasses.replace(e, price=e.price * 3)

    def observations(evts):
        env = make_env(normalizer, sessions=[ref])
        import atlas.learning.env as env_mod

        original = env_mod.load_session
        env_mod.load_session = lambda r: evts
        try:
            obs, _ = env.reset(seed=0)
            out, done = [], False
            while not done:
                now = env._market.now_ns
                if now < cutoff:
                    out.append((now, obs["symbols"].tobytes(), obs["global"].tobytes()))
                obs, _, done, _, _ = env.step(np.array([1, 0]))
            return out
        finally:
            env_mod.load_session = original

    a, b = observations(events), observations([shift(e) for e in events])
    assert a and a == b


def test_ppo_training_is_deterministic_and_logs_diagnostics(tmp_path: Path) -> None:
    train = refs(2)
    r1 = train_ppo(TINY, train, SIM, LIMITS, 500, tmp_path / "a")
    r2 = train_ppo(TINY, train, SIM, LIMITS, 500, tmp_path / "b")
    r3 = train_ppo(TINY.model_copy(update={"seed": 9}), train, SIM, LIMITS, 500, tmp_path / "c")
    sha = lambda r: r.artifact.rpartition("#")[2]  # noqa: E731
    assert sha(r1) == sha(r2) != sha(r3)
    assert len(r1.diagnostics) == 2
    for d in r1.diagnostics:
        assert {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_variance",
                "action_distribution"} <= set(d)
        assert sum(d["action_distribution"]) == pytest.approx(1.0, abs=1e-3)
    s = r1.summary()
    assert s["mean_abs_reward_vs_realized_gap_bps"] is not None
    meta = json.loads((tmp_path / "a" / "policy.json").read_text(encoding="utf-8"))
    assert meta["train_sessions"] == ["2026-06-01", "2026-06-02"]
    assert meta["normalizer"]["fit_sessions"] == ["2026-06-01", "2026-06-02"]


def test_policy_agent_is_frozen_pinned_and_parallel_safe(tmp_path: Path) -> None:
    res = train_ppo(TINY, refs(2), SIM, LIMITS, 500, tmp_path / "p")
    rec = AgentRecord("ppo-t", "ppo", 0, {}, origin="ppo", artifact=res.artifact)
    agent = PolicyAgent(rec)
    with pytest.raises(NotImplementedError, match="frozen"):
        agent.learn([])
    assert agent.metadata().trainable and agent.metadata().population_version == "ppo-v1"

    later = refs(4)[2:]  # sessions after the training ones
    serial = evaluate_records([rec], later, SIM, LIMITS, 500, workers=1)[0]
    parallel = evaluate_records([rec, rec], later, SIM, LIMITS, 500, workers=2)[1]
    assert serial.error is None and serial.violations == []
    assert serial.to_dict() == {**parallel.to_dict(), "agent_id": serial.agent_id}

    weights = Path(res.artifact.rpartition("#")[0])
    weights.write_bytes(weights.read_bytes() + b"tamper")
    load_policy.cache_clear()
    with pytest.raises(ValueError, match="sha256"):
        PolicyAgent(rec)
    with pytest.raises(ValueError, match="artifact"):
        PolicyAgent(AgentRecord("x", "ppo", 0, {}))


def test_walkforward_with_ppo_never_auto_promotes(tmp_path: Path) -> None:
    from test_walkforward import _wf_setup

    from atlas.evaluation.walkforward import WalkForwardConfig, evaluate_holdout, run_walkforward

    wf, evo, sim, base, provider = _wf_setup(tmp_path)
    ppo = TINY.model_dump()
    wf_ppo = WalkForwardConfig.model_validate({**wf.model_dump(), "ppo": ppo})
    report, run_dir = run_walkforward(wf_ppo, evo, sim, base, provider)
    assert "never promoted automatically" in report["ppo"]["promotion"]
    for f in report["folds"]:
        assert f["ppo"]["out_of_sample"].keys() == {"optimistic", "base", "pessimistic"}
        assert f["ppo_training"]["summary"]["final_action_distribution"]
    design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    assert design["final_candidate"] is None or design["final_candidate"]["family"] != "ppo"
    assert "PPO vs evolution" in (run_dir / "report.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="requires ppo.enabled"):
        WalkForwardConfig.model_validate({**wf.model_dump(), "final_candidate_source": "ppo"})

    explicit = WalkForwardConfig.model_validate({**wf.model_dump(), "ppo": ppo, "final_candidate_source": "ppo",
                                                 "experiment_name": "wf_ppo_explicit"})
    report2, run_dir2 = run_walkforward(explicit, evo, sim, base, provider)
    design2 = json.loads((run_dir2 / "design.json").read_text(encoding="utf-8"))
    assert design2["final_candidate"]["family"] == "ppo"
    assert report2["design_fingerprint"] != report["design_fingerprint"]
    out = evaluate_holdout(run_dir2, explicit, evo, sim, base, provider, allow_reuse=True)
    assert out["candidate"]["family"] == "ppo"
    assert math.isfinite(out["champion_ci"]["base"]["mean"])
