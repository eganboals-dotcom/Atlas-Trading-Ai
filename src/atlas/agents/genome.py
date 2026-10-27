"""Parameter spaces and genome operators: sampling, mutation, crossover, distance.

A genome is a plain ``dict[str, float | int]`` so it serializes losslessly to JSON.
Mutation works in a normalized [0, 1] space (log-scaled where declared), so one
mutation scale means the same thing for every parameter.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

Genome = dict[str, float | int]
ParamSpace = Mapping[str, "ParamSpec"]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    kind: Literal["float", "int", "choice"]
    low: float = 0.0
    high: float = 1.0
    choices: tuple[float, ...] = ()
    log: bool = False

    def __post_init__(self) -> None:
        if self.kind == "choice":
            if len(self.choices) < 2:
                raise ValueError("choice params need at least two choices")
        elif self.high <= self.low or (self.log and self.low <= 0):
            raise ValueError(f"invalid bounds [{self.low}, {self.high}] (log={self.log})")

    def to_unit(self, value: float) -> float:
        if self.kind == "choice":
            return self.choices.index(value) / (len(self.choices) - 1)
        if self.log:
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float | int:
        u = min(1.0, max(0.0, u))
        if self.kind == "choice":
            return self.choices[int(round(u * (len(self.choices) - 1)))]
        if self.log:
            value = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            value = self.low + u * (self.high - self.low)
        if self.kind == "int":
            return int(min(self.high, max(self.low, round(value))))
        return float(value)

    def sample(self, rng: np.random.Generator) -> float | int:
        if self.kind == "choice":
            return self.choices[int(rng.integers(len(self.choices)))]
        return self.from_unit(float(rng.random()))

    def mutate(self, value: float, rng: np.random.Generator, scale: float) -> float | int:
        if self.kind == "choice":
            others = [c for c in self.choices if c != value]
            return others[int(rng.integers(len(others)))]
        return self.from_unit(self.to_unit(value) + float(rng.normal(0.0, scale)))

    def contains(self, value: float) -> bool:
        if self.kind == "choice":
            return value in self.choices
        return self.low - 1e-12 <= value <= self.high + 1e-12


def sample_genome(space: ParamSpace, rng: np.random.Generator) -> Genome:
    return {name: space[name].sample(rng) for name in sorted(space)}


def validate_genome(genome: Mapping[str, float], space: ParamSpace) -> None:
    if set(genome) != set(space):
        raise ValueError(f"genome keys {sorted(genome)} do not match space {sorted(space)}")
    bad = [k for k, v in genome.items() if not space[k].contains(v)]
    if bad:
        raise ValueError(f"genome values out of bounds: {bad}")


def mutate_genome(
    genome: Genome, space: ParamSpace, rng: np.random.Generator, scale: float, rate: float
) -> tuple[Genome, list[str]]:
    """Perturb each parameter with probability ``rate``; at least one is always perturbed."""
    names = sorted(space)
    chosen = [n for n in names if rng.random() < rate]
    if not chosen:
        chosen = [names[int(rng.integers(len(names)))]]
    child = dict(genome)
    changes: list[str] = []
    for name in chosen:
        before = child[name]
        child[name] = space[name].mutate(before, rng, scale)
        changes.append(f"{name}: {before:.6g} -> {child[name]:.6g}")
    return child, changes


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """Uniform crossover. Callers must only cross genomes from the same family."""
    if set(a) != set(b):
        raise ValueError("crossover requires genomes from the same parameter space")
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in sorted(a)}


def genome_distance(a: Genome, b: Genome, space: ParamSpace) -> float:
    """Mean per-parameter distance in normalized space, in [0, 1]."""
    total = 0.0
    for name, spec in space.items():
        if spec.kind == "choice":
            total += 0.0 if a[name] == b[name] else 1.0
        else:
            total += abs(spec.to_unit(a[name]) - spec.to_unit(b[name]))
    return total / len(space)


def genome_hash(genome: Mapping[str, float]) -> str:
    return hashlib.sha256(json.dumps(dict(genome), sort_keys=True).encode()).hexdigest()[:16]
