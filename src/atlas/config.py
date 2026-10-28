"""Typed configuration models, validated on load.

Every model forbids unknown keys and is immutable, so a typo in a YAML file fails
loudly at startup instead of silently falling back to a default (fail closed).
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Annotated, Literal, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_serializer, model_validator

NonNeg = Annotated[float, Field(ge=0)]
Pos = Annotated[float, Field(gt=0)]
Rate = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- data


class AnomalyConfig(StrictModel):
    """Deliberate defects injected into synthetic data to exercise validation."""

    duplicate_rate: Rate = 0.0
    out_of_order_rate: Rate = 0.0
    crossed_quote_rate: Rate = 0.0
    missing_field_rate: Rate = 0.0
    stale_gaps: Annotated[int, Field(ge=0)] = 0
    stale_gap_s: Pos = 5.0


class SyntheticConfig(StrictModel):
    seed: int = 7
    initial_prices: dict[str, Pos] = Field(default_factory=dict)
    default_initial_price: Pos = 100.0
    tick_size: Pos = 0.01
    volatility_bps_per_sqrt_s: NonNeg = 1.5
    mean_quote_interval_ms: Pos = 200.0
    trade_probability: Rate = 0.3
    min_spread_ticks: Annotated[int, Field(ge=1)] = 1
    max_spread_ticks: Annotated[int, Field(ge=1)] = 3
    min_size: Annotated[int, Field(ge=1)] = 100
    max_size: Annotated[int, Field(ge=1)] = 2_000
    feed_latency_ms: NonNeg = 2.0
    feed_jitter_ms: NonNeg = 1.0
    anomalies: AnomalyConfig = AnomalyConfig()

    @model_validator(mode="after")
    def _ranges(self) -> SyntheticConfig:
        if self.max_spread_ticks < self.min_spread_ticks:
            raise ValueError("max_spread_ticks must be >= min_spread_ticks")
        if self.max_size < self.min_size:
            raise ValueError("max_size must be >= min_size")
        return self


class DataConfig(StrictModel):
    provider: Literal["synthetic"]
    symbols: list[str] = Field(min_length=1)
    timezone: str = "America/New_York"
    feed: str = "synthetic-l1"
    session_date: date
    start_time: time = time(9, 30)
    duration_s: Annotated[float, Field(gt=0, le=23_400)]
    allowed_data_age_ms: Pos = 2_000.0
    license_notes: str = "Synthetic data generated locally; no license restrictions."
    synthetic: SyntheticConfig = SyntheticConfig()

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip().upper() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("symbols must be non-empty strings")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("symbols must be unique")
        return cleaned

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone {v!r}") from exc
        return v

    @field_validator("session_date")
    @classmethod
    def _weekday(cls, v: date) -> date:
        if v.weekday() >= 5:
            raise ValueError(f"session_date {v} is a weekend")
        return v


# ---------------------------------------------------------------------- simulator


class LatencyDistribution(StrictModel):
    """A latency distribution in milliseconds.

    fixed:     always ``value_ms``
    uniform:   uniform on [low_ms, high_ms]
    lognormal: median ``value_ms``, log-space std ``sigma`` (heavy right tail)
    """

    kind: Literal["fixed", "uniform", "lognormal"] = "fixed"
    value_ms: NonNeg = 0.0
    low_ms: NonNeg = 0.0
    high_ms: NonNeg = 0.0
    sigma: NonNeg = 0.5

    @model_validator(mode="after")
    def _check(self) -> LatencyDistribution:
        if self.kind == "uniform" and self.high_ms < self.low_ms:
            raise ValueError("uniform latency requires high_ms >= low_ms")
        if self.kind == "lognormal" and self.value_ms <= 0:
            raise ValueError("lognormal latency requires value_ms > 0 (the median)")
        return self


class LatencyConfig(StrictModel):
    feed: LatencyDistribution = LatencyDistribution(kind="fixed", value_ms=2.0)
    decision: LatencyDistribution = LatencyDistribution(kind="fixed", value_ms=1.0)
    order: LatencyDistribution = LatencyDistribution(kind="lognormal", value_ms=20.0, sigma=0.4)


class FeeConfig(StrictModel):
    per_share: NonNeg = 0.0
    bps: NonNeg = 0.0
    min_per_order: NonNeg = 0.0
    sell_regulatory_bps: NonNeg = 0.0


class SlippageConfig(StrictModel):
    """Extra adverse price impact applied to every fill, in basis points."""

    bps: NonNeg = 0.5


class SpreadConfig(StrictModel):
    """Fills use the quoted bid/ask; ``widen_bps`` pessimistically widens each side."""

    model: Literal["quoted"] = "quoted"
    widen_bps: NonNeg = 0.0


class FillConfig(StrictModel):
    """Phase 1 fill model: immediate-or-cancel against top of book at order arrival.

    ``max_participation`` caps the fill at a fraction of displayed size.
    Resting limit orders and queue modeling are not implemented yet.
    """

    model: Literal["top_of_book_ioc"] = "top_of_book_ioc"
    max_participation: Annotated[float, Field(gt=0, le=1)] = 1.0


class DemoStrategyConfig(StrictModel):
    """Plumbing-only demo: buy, hold, sell. Not a research strategy."""

    symbol: str | None = None
    order_qty: Annotated[int, Field(gt=0)] = 100
    hold_s: Pos = 30.0
    round_trips: Annotated[int, Field(ge=1)] = 3


class SimulatorConfig(StrictModel):
    data_config: Path
    seed: int = 42
    initial_cash: Pos = 100_000.0
    fees: FeeConfig = FeeConfig()
    slippage: SlippageConfig = SlippageConfig()
    spread: SpreadConfig = SpreadConfig()
    latency: LatencyConfig = LatencyConfig()
    fill: FillConfig = FillConfig()
    stale_quote_ms: Pos = 2_000.0
    invalid_event_policy: Literal["drop", "fail"] = "drop"
    reports_dir: Path = Path("artifacts/reports")
    demo_strategy: DemoStrategyConfig = DemoStrategyConfig()


# ---------------------------------------------------------------------- evolution


class AgentLimits(StrictModel):
    """Hard constraints on what any agent's target position can become.

    These clip proposals before they reach the simulator. A full independent risk
    manager (loss limits, kill switch, order-rate limits) arrives in Phase 5.
    """

    max_position_shares: Annotated[int, Field(gt=0)] = 2_000
    max_order_shares: Annotated[int, Field(gt=0)] = 500
    max_position_frac: Annotated[float, Field(gt=0, le=1)] = 0.5


class WindowConfig(StrictModel):
    """One evaluation window: a synthetic session identified by date and data seed."""

    session_date: date
    seed: int

    @field_validator("session_date")
    @classmethod
    def _weekday(cls, v: date) -> date:
        if v.weekday() >= 5:
            raise ValueError(f"window session_date {v} is a weekend")
        return v


class EvaluationConfig(StrictModel):
    windows: list[WindowConfig] = Field(min_length=1)
    duration_s: Annotated[float, Field(gt=0, le=23_400)] = 600.0
    symbols: list[str] | None = None  # overrides the data config's symbols when set
    decision_interval_ms: Pos = 500.0
    workers: Annotated[int, Field(ge=0)] = 1  # 0 = one per CPU core minus one


class FitnessWeights(StrictModel):
    net_return: NonNeg = 1.0
    max_drawdown: NonNeg = 0.5
    downside: NonNeg = 0.25
    turnover: NonNeg = 0.1
    instability: NonNeg = 0.25


class FitnessAnchors(StrictModel):
    """Fixed, predeclared scales that make each metric dimensionless.

    A metric divided by its anchor reads as "how many anchor units". Anchors and
    weights are frozen before candidates are compared and logged with every run.
    """

    net_return: Pos = 0.001  # 10 bps per window
    max_drawdown: Pos = 0.005  # 50 bps
    downside: Pos = 0.001
    turnover: Pos = 5.0  # traded notional / initial cash, per window
    instability: Pos = 0.001  # std of per-window net returns


class FitnessGates(StrictModel):
    """Hard eligibility gates applied before ranking. High return never offsets a breach."""

    max_drawdown: Annotated[float, Field(gt=0, le=1)] = 0.03
    min_fills: Annotated[int, Field(ge=0)] = 30
    min_windows_traded: Annotated[int, Field(ge=0)] = 3
    max_turnover: Pos = 200.0
    max_violations: Annotated[int, Field(ge=0)] = 0


class FitnessConfig(StrictModel):
    weights: FitnessWeights = FitnessWeights()
    anchors: FitnessAnchors = FitnessAnchors()
    gates: FitnessGates = FitnessGates()
    # "buy_hold": score each window's return minus Hold's return on the same window, so an agent
    # must beat simply holding (and doing nothing is not free in a rising market).
    benchmark: Literal["none", "buy_hold"] = "none"
    # "mean": the drawdown term and gate use the average window's drawdown instead of the single
    # worst one, so one crash day in hundreds does not decide everything.
    drawdown_stat: Literal["worst", "mean"] = "worst"

    @model_serializer(mode="wrap")
    def _omit_defaults(self, handler):  # keeps fingerprints of configs that predate these options
        d = handler(self)
        for k, default in (("benchmark", "none"), ("drawdown_stat", "worst")):
            if d.get(k) == default:
                d.pop(k)
        return d


class OffspringConfig(StrictModel):
    """How each parent's descendants are produced. Counts must sum to offspring_per_parent."""

    elite_copies: Annotated[int, Field(ge=0)] = 1
    small_mutations: Annotated[int, Field(ge=0)] = 6
    large_mutations: Annotated[int, Field(ge=0)] = 2
    immigrants: Annotated[int, Field(ge=0)] = 1
    small_scale: Annotated[float, Field(gt=0, le=1)] = 0.05
    large_scale: Annotated[float, Field(gt=0, le=1)] = 0.25
    mutation_rate: Annotated[float, Field(gt=0, le=1)] = 0.3
    crossover_rate: Rate = 0.2

    @property
    def total(self) -> int:
        return self.elite_copies + self.small_mutations + self.large_mutations + self.immigrants


class DiversityConfig(StrictModel):
    min_parent_distance: Rate = 0.15
    max_parents_per_family: Annotated[int, Field(ge=1)] = 2
    novelty_threshold: Rate = 0.25
    novelty_archive_size: Annotated[int, Field(ge=0)] = 50
    champion_archive_size: Annotated[int, Field(ge=1)] = 20


FAMILY_NAMES = ("momentum", "mean_reversion", "order_flow", "liquidity", "regime_adaptive")
#: Minutes-to-hours families (src/atlas/agents/slow_families.py); evolved as their own population.
SLOW_FAMILY_NAMES = ("intraday_momentum", "opening_range", "trend_follow", "pullback", "smart_hold")
FamilyName = Literal["momentum", "mean_reversion", "order_flow", "liquidity", "regime_adaptive",
                     "intraday_momentum", "opening_range", "trend_follow", "pullback", "smart_hold"]


class EvolutionConfig(StrictModel):
    experiment_name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]
    seed: int = 2026
    population_size: Literal[50] = 50
    parent_count: Annotated[int, Field(ge=1)] = 5
    offspring_per_parent: Annotated[int, Field(ge=1)] = 10
    generations: Annotated[int, Field(ge=1)] = 5
    families: list[FamilyName] = Field(
        default_factory=lambda: list(FAMILY_NAMES)
    )
    simulator_config: Path
    data_config: Path
    evaluation: EvaluationConfig
    limits: AgentLimits = AgentLimits()
    fitness: FitnessConfig = FitnessConfig()
    offspring: OffspringConfig = OffspringConfig()
    diversity: DiversityConfig = DiversityConfig()
    on_insufficient_parents: Literal["hold", "explore"] = "hold"
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    reports_dir: Path = Path("artifacts/reports/evolution")

    @model_validator(mode="after")
    def _shape(self) -> EvolutionConfig:
        if self.parent_count * self.offspring_per_parent != self.population_size:
            raise ValueError(
                f"parent_count ({self.parent_count}) x offspring_per_parent ({self.offspring_per_parent}) "
                f"must equal population_size ({self.population_size})"
            )
        if self.offspring.total != self.offspring_per_parent:
            raise ValueError(
                f"offspring counts sum to {self.offspring.total}, expected {self.offspring_per_parent}"
            )
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("families must be a non-empty list without duplicates")
        if self.population_size % len(self.families):
            raise ValueError("population_size must divide evenly across families")
        return self


# ------------------------------------------------------------------------ loading


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


M = TypeVar("M", bound=BaseModel)


def load_config(path: str | Path, model: type[M]) -> M:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: invalid {model.__name__}:\n{exc}") from exc
