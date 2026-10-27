"""Execution-cost scenarios for sensitivity analysis.

A scenario overrides parts of the simulator config: slippage and spread widening (in
bps), fee and latency multipliers, and the participation cap. Reports emphasize the
base and pessimistic scenarios. Optimistic results are shown for contrast only, never
as the main evidence.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from atlas.config import LatencyDistribution, SimulatorConfig

EMPHASIZED = ("base", "pessimistic")


class CostScenario(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slippage_bps: Annotated[float, Field(ge=0)] | None = None
    widen_bps: Annotated[float, Field(ge=0)] | None = None
    fee_multiplier: Annotated[float, Field(ge=0)] = 1.0
    latency_multiplier: Annotated[float, Field(gt=0)] = 1.0
    max_participation: Annotated[float, Field(gt=0, le=1)] | None = None


DEFAULT_SCENARIOS: dict[str, CostScenario] = {
    "optimistic": CostScenario(slippage_bps=0.0, widen_bps=0.0, latency_multiplier=0.5),
    "base": CostScenario(),
    "pessimistic": CostScenario(slippage_bps=2.0, widen_bps=1.0, fee_multiplier=1.5, latency_multiplier=2.0,
                                max_participation=0.25),
}


def _scale(d: LatencyDistribution, k: float) -> LatencyDistribution:
    return d.model_copy(update={"value_ms": d.value_ms * k, "low_ms": d.low_ms * k, "high_ms": d.high_ms * k})


def apply_scenario(sim: SimulatorConfig, sc: CostScenario) -> SimulatorConfig:
    fees = sim.fees.model_copy(update={k: getattr(sim.fees, k) * sc.fee_multiplier
                                       for k in ("per_share", "bps", "min_per_order", "sell_regulatory_bps")})
    lat = sim.latency
    latency = lat.model_copy(update={"feed": _scale(lat.feed, sc.latency_multiplier),
                                     "decision": _scale(lat.decision, sc.latency_multiplier),
                                     "order": _scale(lat.order, sc.latency_multiplier)})
    update = {"fees": fees, "latency": latency}
    if sc.slippage_bps is not None:
        update["slippage"] = sim.slippage.model_copy(update={"bps": sc.slippage_bps})
    if sc.widen_bps is not None:
        update["spread"] = sim.spread.model_copy(update={"widen_bps": sc.widen_bps})
    if sc.max_participation is not None:
        update["fill"] = sim.fill.model_copy(update={"max_participation": sc.max_participation})
    return SimulatorConfig.model_validate({**sim.model_dump(), **{k: v.model_dump() for k, v in update.items()}})
