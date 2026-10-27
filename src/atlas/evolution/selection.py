"""Diversity-aware parent selection.

Greedy by score, with two diversity constraints so the parents are not near-clones and
one family cannot take over the population in a single generation:

1. skip candidates within ``min_distance`` of an already chosen parent (agents from
   different families are at maximal distance, 1.0);
2. take at most ``max_per_family`` parents from any one family.

If those constraints leave slots open, they are relaxed in that order among *eligible*
candidates only, and every relaxation is logged. If fewer candidates are eligible than
requested, fewer parents are returned — slots are never padded with weak candidates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from atlas.agents.base import AgentRecord
from atlas.agents.genome import genome_distance
from atlas.agents.registry import family_space
from atlas.evolution.fitness import FitnessResult


def record_distance(a: AgentRecord, b: AgentRecord) -> float:
    if a.family != b.family:
        return 1.0
    return genome_distance(a.genome, b.genome, family_space(a.family))


@dataclass
class Selection:
    parents: list[AgentRecord]
    notes: list[str]


def select_parents(
    records: list[AgentRecord],
    fitness: dict[str, FitnessResult],
    count: int,
    min_distance: float,
    max_per_family: int | None = None,
) -> Selection:
    eligible = [r for r in records if fitness[r.agent_id].eligible]
    # Stable ordering: score descending, then agent_id for deterministic tie-breaks.
    ranked = sorted(eligible, key=lambda r: (-fitness[r.agent_id].score, r.agent_id))
    chosen: list[AgentRecord] = []
    notes: list[str] = []
    per_family: Counter[str] = Counter()

    def take(check_distance: bool, check_family: bool, label: str | None) -> None:
        for r in ranked:
            if len(chosen) == count:
                return
            if r in chosen:
                continue
            if check_family and max_per_family is not None and per_family[r.family] >= max_per_family:
                if label is None:
                    notes.append(f"skipped {r.agent_id}: family {r.family} already has {max_per_family} parents")
                continue
            if check_distance:
                nearest = min((record_distance(r, p) for p in chosen), default=1.0)
                if nearest < min_distance:
                    if label is None:
                        notes.append(f"skipped {r.agent_id}: distance {nearest:.3f} < {min_distance} to a parent")
                    continue
            chosen.append(r)
            per_family[r.family] += 1
            if label:
                notes.append(f"{label}: added {r.agent_id}")

    take(check_distance=True, check_family=True, label=None)
    take(check_distance=False, check_family=True, label="distance constraint relaxed")
    take(check_distance=False, check_family=False, label="family cap relaxed")
    if len(chosen) < count:
        notes.append(f"only {len(chosen)} eligible parents (need {count})")
    return Selection(chosen, notes)
