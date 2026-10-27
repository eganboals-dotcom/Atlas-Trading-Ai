"""Chronological splits: nested walk-forward folds plus an untouched outer holdout.

Sessions are never shuffled. The last ``holdout_sessions`` sessions form the outer
holdout. The remaining development sessions are cut into folds:

    rolling:   [ train (n) ][embargo][ validation (v) ]  -> slide forward by `step`
    expanding: [ train (grows from the start) ][embargo][ validation (v) ]

Within a fold, evolution selects on the training sessions only; the fold's champions
are then scored on its validation sessions. An embargo gap of whole sessions sits
between them. Validation windows rotate across folds, so no single period decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    holdout_sessions: int = Field(ge=1)
    train_sessions: int = Field(ge=1)
    validation_sessions: int = Field(ge=1)
    step_sessions: int = Field(ge=1)
    embargo_sessions: int = Field(default=0, ge=0)
    scheme: Literal["rolling", "expanding"] = "rolling"
    max_folds: int | None = Field(default=None, ge=1)


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[date, ...]
    validation: tuple[date, ...]


@dataclass(frozen=True)
class SplitPlan:
    development: tuple[date, ...]
    holdout: tuple[date, ...]
    folds: tuple[Fold, ...]

    def to_dict(self) -> dict:
        return {
            "development": [d.isoformat() for d in self.development],
            "holdout": [d.isoformat() for d in self.holdout],
            "folds": [{"index": f.index, "train": [d.isoformat() for d in f.train],
                       "validation": [d.isoformat() for d in f.validation]} for f in self.folds],
        }


class SplitError(ValueError):
    pass


def plan_splits(sessions: list[date], cfg: SplitConfig) -> SplitPlan:
    ordered = sorted(sessions)
    if len(set(ordered)) != len(ordered):
        raise SplitError("duplicate session dates")
    if len(ordered) <= cfg.holdout_sessions:
        raise SplitError(f"{len(ordered)} sessions cannot leave a {cfg.holdout_sessions}-session holdout")
    dev, holdout = ordered[: -cfg.holdout_sessions], ordered[-cfg.holdout_sessions:]

    folds: list[Fold] = []
    start = 0
    while True:
        train_start = 0 if cfg.scheme == "expanding" else start
        train_end = start + cfg.train_sessions
        val_start = train_end + cfg.embargo_sessions
        val_end = val_start + cfg.validation_sessions
        if val_end > len(dev) or (cfg.max_folds and len(folds) >= cfg.max_folds):
            break
        folds.append(Fold(len(folds), tuple(dev[train_start:train_end]), tuple(dev[val_start:val_end])))
        start += cfg.step_sessions
    if not folds:
        raise SplitError(
            f"{len(dev)} development sessions are too few for train={cfg.train_sessions}, "
            f"embargo={cfg.embargo_sessions}, validation={cfg.validation_sessions}"
        )
    plan = SplitPlan(tuple(dev), tuple(holdout), tuple(folds))
    check_plan(plan)
    return plan


def check_plan(plan: SplitPlan) -> None:
    """Invariants every plan must satisfy; raises SplitError on violation."""
    hold = set(plan.holdout)
    if plan.development and plan.holdout and max(plan.development) >= min(plan.holdout):
        raise SplitError("holdout must come strictly after all development sessions")
    for f in plan.folds:
        if set(f.train) & set(f.validation):
            raise SplitError(f"fold {f.index}: train and validation overlap")
        if max(f.train) >= min(f.validation):
            raise SplitError(f"fold {f.index}: validation must follow training chronologically")
        if (set(f.train) | set(f.validation)) & hold:
            raise SplitError(f"fold {f.index}: uses holdout sessions")
