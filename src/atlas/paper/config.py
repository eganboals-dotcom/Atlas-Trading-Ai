"""Paper-trading and risk configuration. Fail closed.

* ``mode`` must be exactly ``paper`` and ``live_orders_enabled`` exactly ``false``. Any
  other value fails validation, so the paper command refuses to start.
* Every risk limit is required (no defaults). A missing or invalid safety setting stops
  startup.
* Unknown keys are rejected everywhere, so credential fields cannot be smuggled in.
* All models are immutable after loading, so neither agents nor any other code can
  change a limit at runtime.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas.config import AgentLimits
from atlas.data.providers import DataSourceConfig

PosF = Annotated[float, Field(gt=0)]
PosI = Annotated[int, Field(gt=0)]
Frac = Annotated[float, Field(gt=0, le=1)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RiskConfig(Strict):
    allowed_symbols: list[str] = Field(min_length=1)
    max_position_shares: PosI
    max_position_value: PosF
    max_order_shares: PosI
    max_order_value: PosF
    max_gross_exposure: Frac  # market value / equity
    max_net_exposure: Frac  # long-only: equals gross; kept explicit for when shorting is ever specified
    max_daily_loss: Annotated[float, Field(gt=0, lt=1)]  # fraction of start-of-day equity
    max_drawdown: Annotated[float, Field(gt=0, lt=1)]  # fraction of peak equity
    max_orders_per_minute: PosI
    stale_data_ms: PosF
    max_clock_skew_ms: PosF
    clock_recovery_events: PosI
    reconnect_max_retries: Annotated[int, Field(ge=0)]
    reconnect_backoff_ms: Annotated[float, Field(ge=0)]

    @field_validator("allowed_symbols")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        out = [s.strip().upper() for s in v]
        if len(set(out)) != len(out) or any(not s for s in out):
            raise ValueError("allowed_symbols must be unique, non-empty")
        return out


class FaultConfig(Strict):
    """Deliberate failures injected by the replay feed, used for drills and tests."""

    disconnect_at_events: list[int] = Field(default_factory=list)
    disconnect_failures: Annotated[int, Field(ge=0)] = 1  # failed reconnect attempts per disconnect
    stale_gaps: list[tuple[int, float]] = Field(default_factory=list)  # (at event index, silence ms)
    clock_backwards_at: list[int] = Field(default_factory=list)
    future_timestamp_at: list[int] = Field(default_factory=list)
    crash_at_event: int | None = None


class FeedConfig(Strict):
    kind: Literal["replay"]  # the only feed implemented; a vendor streaming adapter is unresolved
    data_source: DataSourceConfig
    session_date: date
    speed: Annotated[float, Field(ge=0)] = 0.0  # 0 = virtual clock (as fast as possible); 1 = real time
    first_quote_timeout_ms: PosF = 5_000.0
    faults: FaultConfig = FaultConfig()


class CandidateSource(Strict):
    """Where a frozen agent comes from. Paper sessions never train or mutate it."""

    source: Literal["design", "checkpoint", "baseline"]
    path: Path | None = None  # design.json or evolution checkpoint
    rank: Annotated[int, Field(ge=0)] = 0  # champion rank within a checkpoint archive
    baseline: str | None = None  # e.g. baseline_buy_hold

    @model_validator(mode="after")
    def _check(self) -> CandidateSource:
        if self.source in ("design", "checkpoint") and self.path is None:
            raise ValueError(f"{self.source} candidate needs a path")
        if self.source == "baseline" and not self.baseline:
            raise ValueError("baseline candidate needs a baseline family name")
        return self


class AlertsConfig(Strict):
    console: bool = True
    file: bool = True


class PaperConfig(Strict):
    mode: Literal["paper"]
    live_orders_enabled: Literal[False]
    session_name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]
    simulator_config: Path
    risk_config: Path
    feed: FeedConfig
    candidate: CandidateSource
    shadows: list[CandidateSource] = Field(default_factory=list, max_length=5)
    agent_limits: AgentLimits
    decision_interval_ms: PosF
    alerts: AlertsConfig = AlertsConfig()
    output_dir: Path = Path("artifacts/paper")
    state_flush_events: PosI = 500
