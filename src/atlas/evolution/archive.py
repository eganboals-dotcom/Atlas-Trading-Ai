"""Champion and novelty archives, kept separate from the active population.

* Champion archive: the best eligible agents ever seen, by score.
* Novelty archive: eligible agents that were behaviorally/genetically far from
  everything already archived — preserved even if they never ranked first.

Full lineage of every agent ever created is kept separately, so nothing is lost when an
archive is capped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas.agents.base import AgentRecord
from atlas.evolution.fitness import FitnessResult
from atlas.evolution.selection import record_distance


@dataclass
class ArchiveEntry:
    record: AgentRecord
    score: float
    generation_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return {"record": self.record.to_dict(), "score": self.score, "generation_evaluated": self.generation_evaluated}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ArchiveEntry:
        return ArchiveEntry(AgentRecord.from_dict(d["record"]), d["score"], d["generation_evaluated"])


@dataclass
class Archives:
    champion_size: int
    novelty_size: int
    novelty_threshold: float
    champions: list[ArchiveEntry] = field(default_factory=list)
    novelty: list[ArchiveEntry] = field(default_factory=list)

    def update(self, records: list[AgentRecord], fitness: dict[str, FitnessResult], generation: int) -> None:
        eligible = sorted(
            (r for r in records if fitness[r.agent_id].eligible),
            key=lambda r: (-fitness[r.agent_id].score, r.agent_id),
        )
        known = {_key(e.record) for e in self.champions}
        for r in eligible:
            if _key(r) not in known:
                self.champions.append(ArchiveEntry(r, fitness[r.agent_id].score, generation))
                known.add(_key(r))
        self.champions.sort(key=lambda e: (-e.score, e.record.agent_id))
        del self.champions[self.champion_size:]

        for r in eligible:
            if len(self.novelty) >= self.novelty_size:
                break
            nearest = min((record_distance(r, e.record) for e in self.novelty), default=1.0)
            if nearest >= self.novelty_threshold:
                self.novelty.append(ArchiveEntry(r, fitness[r.agent_id].score, generation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_size": self.champion_size,
            "novelty_size": self.novelty_size,
            "novelty_threshold": self.novelty_threshold,
            "champions": [e.to_dict() for e in self.champions],
            "novelty": [e.to_dict() for e in self.novelty],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Archives:
        return Archives(
            d["champion_size"], d["novelty_size"], d["novelty_threshold"],
            [ArchiveEntry.from_dict(e) for e in d["champions"]],
            [ArchiveEntry.from_dict(e) for e in d["novelty"]],
        )


def _key(r: AgentRecord) -> tuple[str, tuple[tuple[str, float], ...]]:
    """Identity of a strategy: family + genome. Elite copies of a champion are not re-archived."""
    return r.family, tuple(sorted(r.genome.items()))
