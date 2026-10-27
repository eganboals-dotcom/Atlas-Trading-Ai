"""Create the next generation from selected parents.

Each parent fills ``offspring_per_parent`` slots with a configurable mix of an exact
elite copy, small mutations (optionally after same-family crossover), large mutations,
and fresh immigrants of a random family. Every child records its origin, parents,
and the exact parameter changes.
"""

from __future__ import annotations

import numpy as np

from atlas.agents.base import AgentRecord
from atlas.agents.genome import crossover, mutate_genome, sample_genome
from atlas.agents.registry import FAMILY_CLASSES, family_space
from atlas.config import OffspringConfig


def agent_id(generation: int, index: int) -> str:
    return f"g{generation:03d}-a{index:02d}"


def initial_population(families: list[str], size: int, rng: np.random.Generator) -> list[AgentRecord]:
    """Equal count per family. The first member of each family uses the default genome."""
    per_family = size // len(families)
    records: list[AgentRecord] = []
    for family in families:
        cls = FAMILY_CLASSES[family]
        for k in range(per_family):
            genome = dict(cls.default_genome) if k == 0 else sample_genome(cls.param_space, rng)
            records.append(AgentRecord(agent_id(0, len(records)), family, 0, genome, (), "initial"))
    return records


def immigrant(families: list[str], generation: int, index: int, rng: np.random.Generator,
              slot_parent: str | None = None) -> AgentRecord:
    family = families[int(rng.integers(len(families)))]
    note = [f"slot of {slot_parent}"] if slot_parent else []
    return AgentRecord(agent_id(generation, index), family, generation,
                       sample_genome(family_space(family), rng), (), "immigrant", note)


def make_offspring(
    parents: list[AgentRecord],
    generation: int,
    cfg: OffspringConfig,
    families: list[str],
    rng: np.random.Generator,
) -> list[AgentRecord]:
    children: list[AgentRecord] = []

    def nid() -> str:
        return agent_id(generation, len(children))

    for p in parents:
        space = family_space(p.family)
        mates = [q for q in parents if q.family == p.family and q.agent_id != p.agent_id]
        for _ in range(cfg.elite_copies):
            children.append(AgentRecord(nid(), p.family, generation, dict(p.genome), (p.agent_id,), "elite"))
        for _ in range(cfg.small_mutations):
            base, lineage, origin = p.genome, (p.agent_id,), "small_mutation"
            if mates and rng.random() < cfg.crossover_rate:
                mate = mates[int(rng.integers(len(mates)))]
                base, lineage, origin = crossover(p.genome, mate.genome, rng), (p.agent_id, mate.agent_id), "crossover"
            genome, changes = mutate_genome(base, space, rng, cfg.small_scale, cfg.mutation_rate)
            children.append(AgentRecord(nid(), p.family, generation, genome, lineage, origin, changes))
        for _ in range(cfg.large_mutations):
            genome, changes = mutate_genome(p.genome, space, rng, cfg.large_scale, cfg.mutation_rate)
            children.append(AgentRecord(nid(), p.family, generation, genome, (p.agent_id,), "large_mutation", changes))
        for _ in range(cfg.immigrants):
            children.append(immigrant(families, generation, len(children), rng, slot_parent=p.agent_id))
    return children
