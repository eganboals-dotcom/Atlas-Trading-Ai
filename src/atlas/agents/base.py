"""The common agent contract.

An agent turns an observation into a *target position* per symbol. It never places
orders: the strategy adapter converts targets into order proposals, limits clip them,
and the simulator decides what executes.

Lifecycle: ``initialize`` -> repeated ``observe``/``act`` -> optional ``learn``
(offline only; rule-based agents do not learn, evolution changes their genomes).
A fresh agent instance is created for every evaluation run, so no mutable state can
leak between agents or windows.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from atlas import __version__
from atlas.agents.features import SymbolFeatures
from atlas.agents.genome import Genome, ParamSpace, genome_hash, validate_genome
from atlas.config import AgentLimits
from atlas.environment.market_state import PortfolioState

POPULATION_VERSION = "v1-non-rl"  # bump when the population's composition changes (e.g. PPO joins)

TargetPositions = dict[str, int]


@dataclass(frozen=True, slots=True)
class AgentMetadata:
    agent_id: str
    family: str
    generation: int
    parents: tuple[str, ...]
    origin: str
    genome_hash: str
    code_version: str = __version__
    population_version: str = POPULATION_VERSION
    trainable: bool = False


@dataclass(frozen=True, slots=True)
class AgentRuntime:
    symbols: tuple[str, ...]
    limits: AgentLimits


@dataclass(frozen=True, slots=True)
class Observation:
    now_ns: int
    features: Mapping[str, SymbolFeatures]
    cash: float
    equity: float
    positions: Mapping[str, int]

    def qty(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)


@dataclass
class AgentRecord:
    """Serializable identity + lineage of one agent. The genome fully determines behavior."""

    agent_id: str
    family: str
    generation: int
    genome: Genome
    parents: tuple[str, ...] = ()
    origin: str = "initial"  # initial | elite | small_mutation | large_mutation | crossover | immigrant | ppo
    mutations: list[str] = field(default_factory=list)
    # Learned policies only: "<path to policy file>#<sha256>". The hash pins the exact weights.
    artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["parents"] = list(self.parents)
        return out

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> AgentRecord:
        return AgentRecord(
            agent_id=d["agent_id"],
            family=d["family"],
            generation=int(d["generation"]),
            genome=dict(d["genome"]),
            parents=tuple(d.get("parents", ())),
            origin=d.get("origin", "initial"),
            mutations=list(d.get("mutations", [])),
            artifact=d.get("artifact"),
        )


class Agent(ABC):
    family: ClassVar[str]
    param_space: ClassVar[ParamSpace]
    default_genome: ClassVar[Genome]

    def __init__(self, record: AgentRecord) -> None:
        if record.family != self.family:
            raise ValueError(f"record family {record.family!r} does not match agent family {self.family!r}")
        validate_genome(record.genome, self.param_space)
        self.record = record
        self.genome: Mapping[str, float] = MappingProxyType(dict(record.genome))
        self.runtime: AgentRuntime | None = None
        self.seed = 0

    # -- contract -------------------------------------------------------------

    def initialize(self, runtime: AgentRuntime, seed: int) -> None:
        """Deterministic setup; clears all per-run state."""
        self.runtime = runtime
        self.seed = seed
        self.reset()

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Clear per-run state. Subclasses with state override this."""

    def observe(self, features: Mapping[str, SymbolFeatures], portfolio: PortfolioState, now_ns: int) -> Observation:
        return Observation(now_ns, MappingProxyType(dict(features)), portfolio.cash, portfolio.equity,
                           portfolio.positions)

    @abstractmethod
    def act(self, observation: Observation) -> TargetPositions:
        """Return the desired share count per symbol (long-only: >= 0)."""

    def learn(self, experience_batch: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is rule-based and does not learn online; evolution mutates its genome"
        )

    def metadata(self) -> AgentMetadata:
        r = self.record
        return AgentMetadata(r.agent_id, r.family, r.generation, tuple(r.parents), r.origin,
                             genome_hash(r.genome))

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {"record": self.record.to_dict(), "metadata": asdict(self.metadata())}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> Agent:
        from atlas.agents.registry import create_agent

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = create_agent(AgentRecord.from_dict(payload["record"]))
        if payload["metadata"]["genome_hash"] != agent.metadata().genome_hash:
            raise ValueError("checkpoint genome hash mismatch")
        return agent

    # -- helpers ----------------------------------------------------------------

    def size_for(self, f: SymbolFeatures, equity: float, frac: float) -> int:
        """Shares worth ``frac`` of equity at the ask, within hard limits."""
        assert self.runtime is not None, "initialize() must be called before act()"
        if not f.valid or not math.isfinite(f.ask) or f.ask <= 0:
            return 0
        limits = self.runtime.limits
        shares = int(min(frac, limits.max_position_frac) * equity / f.ask)
        return max(0, min(shares, limits.max_position_shares))
