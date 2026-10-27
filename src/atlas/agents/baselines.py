"""Fixed evaluation baselines. They sit outside the population and are never bred or selected.

Every evolved agent is compared against these under identical data and costs.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from atlas.agents.base import Agent, AgentRecord, Observation, TargetPositions
from atlas.agents.families import MeanReversionAgent, MomentumAgent
from atlas.agents.genome import ParamSpec


class CashBaseline(Agent):
    """Never trades. Any agent that cannot beat doing nothing after costs has no edge here."""

    family: ClassVar[str] = "baseline_cash"
    param_space: ClassVar[dict[str, ParamSpec]] = {}
    default_genome: ClassVar[dict[str, float]] = {}

    def act(self, obs: Observation) -> TargetPositions:
        return {s: 0 for s in obs.features}


class BuyAndHoldBaseline(Agent):
    """Buys an equal-weight basket at the first valid quotes and holds to the end."""

    family: ClassVar[str] = "baseline_buy_hold"
    param_space: ClassVar[dict[str, ParamSpec]] = {}
    default_genome: ClassVar[dict[str, float]] = {}
    INVESTED_FRAC = 0.9

    def act(self, obs: Observation) -> TargetPositions:
        targets: TargetPositions = {}
        frac = self.INVESTED_FRAC / max(1, len(obs.features))
        for symbol, f in obs.features.items():
            held = obs.qty(symbol)
            targets[symbol] = held if held else self.size_for(f, obs.equity, frac)
        return targets


class SimpleMomentumBaseline(MomentumAgent):
    family: ClassVar[str] = "baseline_momentum"


class SimpleMeanReversionBaseline(MeanReversionAgent):
    family: ClassVar[str] = "baseline_mean_reversion"


class RandomControl(Agent):
    """Randomly toggles between flat and a fixed position. Seeded, so reproducible."""

    family: ClassVar[str] = "baseline_random"
    param_space: ClassVar[dict[str, ParamSpec]] = {}
    default_genome: ClassVar[dict[str, float]] = {}
    TOGGLE_PROB = 0.02
    SIZE_FRAC = 0.2

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs: Observation) -> TargetPositions:
        targets: TargetPositions = {}
        for symbol, f in obs.features.items():
            held = obs.qty(symbol)
            if self._rng.random() < self.TOGGLE_PROB:
                targets[symbol] = 0 if held else self.size_for(f, obs.equity, self.SIZE_FRAC)
            else:
                targets[symbol] = held
        return targets


BASELINE_CLASSES: dict[str, type[Agent]] = {
    cls.family: cls
    for cls in (CashBaseline, BuyAndHoldBaseline, SimpleMomentumBaseline, SimpleMeanReversionBaseline, RandomControl)
}


def baseline_records() -> list[AgentRecord]:
    return [
        AgentRecord(agent_id=f"baseline-{family.removeprefix('baseline_')}", family=family, generation=-1,
                    genome=dict(cls.default_genome), origin="baseline")
        for family, cls in BASELINE_CLASSES.items()
    ]
