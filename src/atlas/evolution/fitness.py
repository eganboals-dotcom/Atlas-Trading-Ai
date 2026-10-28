"""Eligibility gates and the research score.

    J = w_R * R/a_R - w_D * D/a_D - w_V * V/a_V - w_T * T/a_T - w_S * S/a_S

R = mean net return per window (optionally in excess of Hold's return on the same window),
D = worst (or mean) window drawdown, V = mean downside
deviation, T = mean turnover, S = std of per-window net returns (instability).
Each metric is divided by a fixed, predeclared anchor (see ``FitnessAnchors``), so the
terms are dimensionless and comparable. Weights and anchors are frozen in config and
logged; gates are applied first, and a gated-out agent has no score at all.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from atlas.config import FitnessConfig
from atlas.evolution.evaluation import AgentEvaluation


@dataclass(frozen=True)
class Aggregate:
    mean_net_return: float
    worst_drawdown: float
    mean_downside: float
    mean_turnover: float
    instability: float
    total_fills: int
    windows_traded: int
    window_returns: tuple[float, ...]


@dataclass
class FitnessResult:
    agent_id: str
    eligible: bool
    score: float | None
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    aggregate: Aggregate | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.aggregate is not None:
            d["aggregate"]["window_returns"] = list(self.aggregate.window_returns)
        return d


def aggregate(ev: AgentEvaluation, benchmark: tuple[float, ...] | None = None,
              drawdown_stat: str = "worst") -> Aggregate:
    """Window metrics. With ``benchmark`` (one return per window), returns are excess over it."""
    w = ev.windows
    returns = tuple(m.net_return for m in w)
    if benchmark is not None:
        returns = tuple(r - b for r, b in zip(returns, benchmark, strict=True))
    dds = [m.max_drawdown for m in w]
    return Aggregate(
        mean_net_return=statistics.fmean(returns),
        worst_drawdown=max(dds) if drawdown_stat == "worst" else statistics.fmean(dds),
        mean_downside=statistics.fmean(m.downside for m in w),
        mean_turnover=statistics.fmean(m.turnover for m in w),
        instability=statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        total_fills=sum(m.fills for m in w),
        windows_traded=sum(m.fills > 0 for m in w),
        window_returns=returns,
    )


def score(ev: AgentEvaluation, cfg: FitnessConfig, expected_windows: int,
          benchmark: tuple[float, ...] | None = None) -> FitnessResult:
    """``benchmark``: Hold's return per window, required when ``cfg.benchmark == "buy_hold"``."""
    if ev.error:
        return FitnessResult(ev.agent_id, False, None, [f"invalid: {ev.error}"])
    if len(ev.windows) != expected_windows:
        return FitnessResult(ev.agent_id, False, None, ["invalid: incomplete evaluation"])

    if cfg.benchmark == "buy_hold" and benchmark is None:
        raise ValueError("fitness benchmark is buy_hold but no benchmark returns were given")
    agg = aggregate(ev, benchmark if cfg.benchmark == "buy_hold" else None, cfg.drawdown_stat)
    g, w, a = cfg.gates, cfg.weights, cfg.anchors
    reasons: list[str] = []
    if len(ev.violations) > g.max_violations:
        reasons.append(f"illegal actions: {len(ev.violations)}")
    if agg.worst_drawdown > g.max_drawdown:
        reasons.append(f"drawdown {agg.worst_drawdown:.4f} > {g.max_drawdown}")
    if agg.mean_turnover > g.max_turnover:
        reasons.append(f"turnover {agg.mean_turnover:.1f} > {g.max_turnover}")
    if agg.total_fills < g.min_fills or agg.windows_traded < g.min_windows_traded:
        reasons.append(
            f"insufficient evidence: {agg.total_fills} fills in {agg.windows_traded} windows "
            f"(need {g.min_fills} fills, {g.min_windows_traded} windows)"
        )

    components = {
        "net_return": w.net_return * agg.mean_net_return / a.net_return,
        "max_drawdown": -w.max_drawdown * agg.worst_drawdown / a.max_drawdown,
        "downside": -w.downside * agg.mean_downside / a.downside,
        "turnover": -w.turnover * agg.mean_turnover / a.turnover,
        "instability": -w.instability * agg.instability / a.instability,
    }
    eligible = not reasons
    return FitnessResult(ev.agent_id, eligible, sum(components.values()) if eligible else None,
                         reasons, components, agg)


def fitness_fingerprint(cfg: FitnessConfig) -> str:
    """Hash of the frozen weights/anchors/gates, logged with every generation."""
    return hashlib.sha256(json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]
