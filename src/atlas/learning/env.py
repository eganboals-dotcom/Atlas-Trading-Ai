"""Gymnasium environment over the deterministic replay engine.

One episode is one training session. Each step is one agent decision, made at the
same throttled decision interval the evolutionary agents use. The action is one
discrete target level per symbol (``ACTION_LEVELS`` times ``size_frac`` of equity).
Targets go through the *same* adapter, limits, latency, and fill model as every other
agent, so PPO gets no execution advantage.

Reward for step t: ``log(V_t / V_{t-1}) * reward_scale``. Here V is portfolio equity
(marked at the visible mid) just before each decision, after all fills and costs up to
that point. The final step uses the closing equity. With ``terminal_exit_cost``, the
estimated cost of crossing out of open positions is deducted from that closing equity.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from atlas.agents.adapter import AgentStrategy
from atlas.agents.baselines import CashBaseline, baseline_records
from atlas.agents.features import SymbolFeatures
from atlas.config import AgentLimits, SimulatorConfig
from atlas.data.providers import SessionRef, load_session
from atlas.environment.costs import CostModel
from atlas.environment.market_state import MarketView, PortfolioState
from atlas.environment.orders import Side
from atlas.environment.replay import ReplayEngine, RunResult
from atlas.learning.observation import N_GLOBAL_INPUTS, N_SYMBOL_INPUTS, FeatureNormalizer, build_observation
from atlas.learning.settings import ACTION_LEVELS
from atlas.timeutils import NS_PER_MS


def targets_from_action(action: np.ndarray, snapshot: dict[str, SymbolFeatures], portfolio: PortfolioState,
                        symbols: list[str], size_frac: float, limits: AgentLimits) -> dict[str, int]:
    out: dict[str, int] = {}
    for s, a in zip(symbols, action):
        f = snapshot[s]
        if not f.valid or not math.isfinite(f.ask) or f.ask <= 0:
            out[s] = portfolio.qty(s)  # hold: never act on stale data
            continue
        frac = ACTION_LEVELS[int(a)] * size_frac
        out[s] = max(0, min(int(frac * portfolio.equity / f.ask), limits.max_position_shares))
    return out


class AtlasTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sessions: list[SessionRef], sim: SimulatorConfig, limits: AgentLimits,
                 decision_interval_ms: float, normalizer: FeatureNormalizer, size_frac: float,
                 reward_scale: float = 1e4, terminal_exit_cost: bool = True, seed: int = 0) -> None:
        if not sessions:
            raise ValueError("environment needs at least one session")
        symbol_sets = {r.symbols for r in sessions}
        if len(symbol_sets) != 1:
            raise ValueError(f"all training sessions must share one symbol set, got {symbol_sets}")
        self.sessions = sessions
        self.symbols = list(sessions[0].symbols)
        self.sim, self.limits = sim, limits
        self.interval_ns = int(decision_interval_ms * NS_PER_MS)
        self.normalizer = normalizer
        self.size_frac, self.reward_scale, self.terminal_exit_cost = size_frac, reward_scale, terminal_exit_cost
        self.costs = CostModel(sim.fees, sim.slippage, sim.spread)
        n = len(self.symbols)
        self.observation_space = gym.spaces.Dict({
            "symbols": gym.spaces.Box(-np.inf, np.inf, (n, N_SYMBOL_INPUTS), np.float32),
            "global": gym.spaces.Box(0.0, 1.0, (N_GLOBAL_INPUTS,), np.float32),
        })
        self.action_space = gym.spaces.MultiDiscrete([len(ACTION_LEVELS)] * n)
        self._order = np.random.default_rng(seed)
        self._queue: list[int] = []
        self.result: RunResult | None = None

    # -- episode control -----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._order = np.random.default_rng(seed)
        if options and "session" in options:
            idx = int(options["session"])
        else:
            if not self._queue:  # each session once per pass, in a seeded shuffled order
                self._queue = list(self._order.permutation(len(self.sessions)))
            idx = int(self._queue.pop())
        self.ref = self.sessions[idx]
        self.adapter = AgentStrategy(CashBaseline(baseline_records()[0]), self.symbols, self.limits,
                                     self.interval_ns, int(self.sim.stale_quote_ms * NS_PER_MS), self.sim.seed)
        self._steps = ReplayEngine(self.sim, self.symbols).iter_run(load_session(self.ref),
                                                                    self.adapter.on_order_update)
        self.result = None
        self.episode_reward = 0.0
        self.action_counts = np.zeros((len(self.symbols), len(ACTION_LEVELS)), dtype=np.int64)
        try:
            market, state = next(self._steps)
            while not self.adapter.poll(market):
                market, state = self._steps.send(())
        except StopIteration as done:  # pragma: no cover - empty session
            raise ValueError(f"session {self.ref.session_date} has no decision points") from done
        self._set(market, state)
        self._equity0 = state.equity
        return self._obs(), {"session": self.ref.session_date.isoformat()}

    def _set(self, market: MarketView, state: PortfolioState) -> None:
        self._market, self._state = market, state
        self._snapshot = self.adapter.snapshot(market.now_ns)
        self._prev_equity = state.equity

    def _obs(self) -> dict[str, np.ndarray]:
        sym, glob = build_observation(self._snapshot, self._state, self.symbols, self.normalizer,
                                      self._market.now_ns)
        return {"symbols": sym, "global": glob}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        for i, a in enumerate(action):
            self.action_counts[i, a] += 1
        targets = targets_from_action(action, self._snapshot, self._state, self.symbols, self.size_frac,
                                      self.limits)
        intents = self.adapter.intents_for(targets, self._snapshot, self._state)
        prev = self._prev_equity
        try:
            market, state = self._steps.send(intents)
            while not self.adapter.poll(market):
                market, state = self._steps.send(())
        except StopIteration as done:
            self.result = done.value
            final = self.result.final["equity"]
            adjusted = final - (self._exit_cost(self.result) if self.terminal_exit_cost else 0.0)
            reward = math.log(max(adjusted, 1e-9) / prev) * self.reward_scale
            self.episode_reward += reward
            info = {
                "session": self.ref.session_date.isoformat(),
                "realized_log_return": math.log(final / self._equity0),
                "adjusted_log_return": math.log(max(adjusted, 1e-9) / self._equity0),
                "fills": len(self.result.fills),
                "action_counts": self.action_counts.tolist(),
            }
            return self._obs(), reward, True, False, info
        self._set(market, state)
        reward = math.log(state.equity / prev) * self.reward_scale
        self.episode_reward += reward
        return self._obs(), reward, False, False, {}

    def _exit_cost(self, result: RunResult) -> float:
        """Estimated cost to flatten at the close: mid -> slipped bid, plus sell fees."""
        cost = 0.0
        for s, pos in result.final["positions"].items():
            qty = pos["quantity"]
            f = self._snapshot.get(s)
            if not qty or f is None or not math.isfinite(f.bid):
                continue
            exit_px = self.costs.execution_price(Side.SELL, f.bid, f.ask)
            cost += qty * (result.final["marks"].get(s, f.mid) - exit_px) + self.costs.fee(Side.SELL, qty, exit_px)
        return max(0.0, cost)
