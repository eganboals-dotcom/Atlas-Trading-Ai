"""Family registry and agent factory."""

from __future__ import annotations

from atlas.agents.base import Agent, AgentRecord
from atlas.agents.baselines import BASELINE_CLASSES
from atlas.agents.families import (
    LiquidityAgent,
    MeanReversionAgent,
    MomentumAgent,
    OrderFlowAgent,
    RegimeAdaptiveAgent,
)
from atlas.agents.genome import ParamSpace

#: Families that may appear in the evolving population. Baselines are deliberately excluded.
FAMILY_CLASSES: dict[str, type[Agent]] = {
    cls.family: cls
    for cls in (MomentumAgent, MeanReversionAgent, OrderFlowAgent, LiquidityAgent, RegimeAdaptiveAgent)
}


def family_space(family: str) -> ParamSpace:
    if family not in FAMILY_CLASSES:
        raise KeyError(f"{family!r} is not an evolvable family")
    return FAMILY_CLASSES[family].param_space


def create_agent(record: AgentRecord) -> Agent:
    cls = FAMILY_CLASSES.get(record.family) or BASELINE_CLASSES.get(record.family)
    if cls is None:
        raise KeyError(f"unknown agent family {record.family!r}")
    return cls(record)
