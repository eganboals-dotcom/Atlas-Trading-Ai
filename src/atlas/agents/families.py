"""The five initial non-RL strategy families.

Each family is a parameterized entry/exit rule applied independently per symbol.
These are strategy *hypotheses* to be tested, not claims that any of them works.
Every family shares three parameters:

* ``size_frac`` — target position value as a fraction of equity (clipped by limits);
* ``max_hold_s`` — positions are closed after this long regardless of signal;
* ``max_spread_bps`` — no entries when the quoted spread is wider than this.
"""

from __future__ import annotations

import math
from typing import ClassVar

from atlas.agents.base import Agent, Observation, TargetPositions
from atlas.agents.features import FLOW_WINDOWS_S, RETURN_HORIZONS_S, SymbolFeatures
from atlas.agents.genome import ParamSpec
from atlas.timeutils import NS_PER_S

COMMON_SPACE: dict[str, ParamSpec] = {
    "size_frac": ParamSpec("float", 0.05, 0.45),
    "max_hold_s": ParamSpec("float", 5.0, 600.0, log=True),
    "max_spread_bps": ParamSpec("float", 0.3, 10.0, log=True),
}
COMMON_DEFAULT = {"size_frac": 0.2, "max_hold_s": 60.0, "max_spread_bps": 3.0}


def zscore(f: SymbolFeatures, horizon: float) -> float:
    """Return over ``horizon`` in units of its expected std under current volatility."""
    r = f.returns_bps[horizon]
    if math.isnan(r) or not (f.vol_bps > 0):
        return math.nan
    return r / (f.vol_bps * math.sqrt(horizon))


class PositionRuleAgent(Agent):
    """Shared per-symbol state machine: flat -> (entry signal) -> long -> (exit or timeout) -> flat."""

    def reset(self) -> None:
        self._entry_ts: dict[str, int] = {}

    def should_enter(self, f: SymbolFeatures, symbol: str) -> bool:
        raise NotImplementedError

    def should_exit(self, f: SymbolFeatures, symbol: str) -> bool:
        raise NotImplementedError

    def act(self, obs: Observation) -> TargetPositions:
        g = self.genome
        targets: TargetPositions = {}
        for symbol, f in obs.features.items():
            held = obs.qty(symbol)
            if held == 0:
                targets[symbol] = 0
                if f.valid and f.spread_bps <= g["max_spread_bps"] and self.should_enter(f, symbol):
                    size = self.size_for(f, obs.equity, g["size_frac"])
                    if size > 0:
                        targets[symbol] = size
                        self._entry_ts.setdefault(symbol, obs.now_ns)
                        continue
                self._entry_ts.pop(symbol, None)
            else:
                entry = self._entry_ts.setdefault(symbol, obs.now_ns)
                timed_out = obs.now_ns - entry >= g["max_hold_s"] * NS_PER_S
                exit_now = timed_out or (f.valid and self.should_exit(f, symbol))
                targets[symbol] = 0 if exit_now else held
        return targets


class MomentumAgent(PositionRuleAgent):
    """Short-horizon continuation: buy after a sufficiently strong up-move."""

    family: ClassVar[str] = "momentum"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **COMMON_SPACE,
        "horizon_s": ParamSpec("choice", choices=RETURN_HORIZONS_S),
        "entry_bps": ParamSpec("float", 0.2, 20.0, log=True),
        "exit_bps": ParamSpec("float", 0.1, 20.0, log=True),
    }
    default_genome: ClassVar[dict[str, float]] = {**COMMON_DEFAULT, "horizon_s": 5.0, "entry_bps": 2.0, "exit_bps": 1.0}

    def should_enter(self, f: SymbolFeatures, symbol: str) -> bool:
        r = f.returns_bps[self.genome["horizon_s"]]
        return not math.isnan(r) and r >= self.genome["entry_bps"]

    def should_exit(self, f: SymbolFeatures, symbol: str) -> bool:
        r = f.returns_bps[self.genome["horizon_s"]]
        return not math.isnan(r) and r <= -self.genome["exit_bps"]


class MeanReversionAgent(PositionRuleAgent):
    """Reversion after a brief dislocation, measured as a volatility-scaled z-score.

    ``imbalance_guard`` guards against adverse selection: skip entries when the book is
    leaning heavily to the offer (l1_imbalance below -guard), a sign the drop may continue.
    """

    family: ClassVar[str] = "mean_reversion"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **COMMON_SPACE,
        "horizon_s": ParamSpec("choice", choices=RETURN_HORIZONS_S),
        "entry_z": ParamSpec("float", 0.5, 4.0),
        "exit_z": ParamSpec("float", -0.5, 1.5),
        "imbalance_guard": ParamSpec("float", 0.0, 1.0),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **COMMON_DEFAULT, "horizon_s": 5.0, "entry_z": 2.0, "exit_z": 0.0, "imbalance_guard": 0.5,
    }

    def should_enter(self, f: SymbolFeatures, symbol: str) -> bool:
        z = zscore(f, self.genome["horizon_s"])
        return (not math.isnan(z) and z <= -self.genome["entry_z"]
                and f.l1_imbalance >= -self.genome["imbalance_guard"])

    def should_exit(self, f: SymbolFeatures, symbol: str) -> bool:
        z = zscore(f, self.genome["horizon_s"])
        return not math.isnan(z) and z >= self.genome["exit_z"]


class OrderFlowAgent(PositionRuleAgent):
    """Buy/sell pressure: blend of signed trade-flow imbalance and top-of-book size imbalance."""

    family: ClassVar[str] = "order_flow"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **COMMON_SPACE,
        "flow_window_s": ParamSpec("choice", choices=FLOW_WINDOWS_S),
        "l1_weight": ParamSpec("float", 0.0, 1.0),
        "entry_signal": ParamSpec("float", 0.05, 0.9),
        "exit_signal": ParamSpec("float", -0.9, 0.3),
        "min_volume": ParamSpec("int", 1, 2_000, log=True),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **COMMON_DEFAULT, "flow_window_s": 5.0, "l1_weight": 0.3, "entry_signal": 0.4, "exit_signal": -0.2,
        "min_volume": 50,
    }

    def _signal(self, f: SymbolFeatures) -> float:
        w = self.genome["l1_weight"]
        return (1 - w) * f.flow_imbalance[self.genome["flow_window_s"]] + w * f.l1_imbalance

    def should_enter(self, f: SymbolFeatures, symbol: str) -> bool:
        return (f.trade_volume[self.genome["flow_window_s"]] >= self.genome["min_volume"]
                and self._signal(f) >= self.genome["entry_signal"])

    def should_exit(self, f: SymbolFeatures, symbol: str) -> bool:
        return self._signal(f) <= self.genome["exit_signal"]


class LiquidityAgent(PositionRuleAgent):
    """Enter only into tight, deep markets with bid-side support; exit when liquidity degrades.

    No queue priority is assumed: entries cross the spread like every other family.
    """

    family: ClassVar[str] = "liquidity"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **COMMON_SPACE,
        "min_depth": ParamSpec("int", 100, 3_000, log=True),
        "imbalance_entry": ParamSpec("float", 0.0, 0.9),
        "imbalance_exit": ParamSpec("float", -0.9, 0.0),
        "exit_spread_mult": ParamSpec("float", 1.0, 5.0),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **COMMON_DEFAULT, "max_spread_bps": 1.0, "min_depth": 800, "imbalance_entry": 0.3,
        "imbalance_exit": -0.3, "exit_spread_mult": 2.0,
    }

    def should_enter(self, f: SymbolFeatures, symbol: str) -> bool:
        return f.ask_size >= self.genome["min_depth"] and f.l1_imbalance >= self.genome["imbalance_entry"]

    def should_exit(self, f: SymbolFeatures, symbol: str) -> bool:
        too_wide = f.spread_bps > self.genome["exit_spread_mult"] * self.genome["max_spread_bps"]
        return too_wide or f.l1_imbalance <= self.genome["imbalance_exit"]


class RegimeAdaptiveAgent(PositionRuleAgent):
    """Momentum in calm regimes, mean reversion (or stand aside) in volatile ones.

    The regime is high-volatility when short-horizon vol exceeds long-horizon vol by
    ``vol_ratio_threshold``. Exit logic follows the regime the position was opened in.
    """

    family: ClassVar[str] = "regime_adaptive"
    param_space: ClassVar[dict[str, ParamSpec]] = {
        **COMMON_SPACE,
        "horizon_s": ParamSpec("choice", choices=RETURN_HORIZONS_S),
        "vol_ratio_threshold": ParamSpec("float", 0.5, 2.0),
        "mom_entry_bps": ParamSpec("float", 0.2, 20.0, log=True),
        "mr_entry_z": ParamSpec("float", 0.5, 4.0),
        "exit_bps": ParamSpec("float", 0.1, 20.0, log=True),
        "high_vol_flat": ParamSpec("choice", choices=(0, 1)),
    }
    default_genome: ClassVar[dict[str, float]] = {
        **COMMON_DEFAULT, "horizon_s": 5.0, "vol_ratio_threshold": 1.2, "mom_entry_bps": 2.0,
        "mr_entry_z": 2.0, "exit_bps": 1.0, "high_vol_flat": 0,
    }

    def reset(self) -> None:
        super().reset()
        self._entry_regime: dict[str, str] = {}

    def should_enter(self, f: SymbolFeatures, symbol: str) -> bool:
        g = self.genome
        if f.vol_ratio > g["vol_ratio_threshold"]:
            z = zscore(f, g["horizon_s"])
            ok = not g["high_vol_flat"] and not math.isnan(z) and z <= -g["mr_entry_z"]
            regime = "high"
        else:
            r = f.returns_bps[g["horizon_s"]]
            ok = not math.isnan(r) and r >= g["mom_entry_bps"]
            regime = "low"
        if ok:
            self._entry_regime[symbol] = regime
        return ok

    def should_exit(self, f: SymbolFeatures, symbol: str) -> bool:
        g = self.genome
        if self._entry_regime.get(symbol) == "high":
            z = zscore(f, g["horizon_s"])
            return not math.isnan(z) and z >= 0.0
        r = f.returns_bps[g["horizon_s"]]
        return not math.isnan(r) and r <= -g["exit_bps"]
