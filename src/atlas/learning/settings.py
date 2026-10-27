"""PPO settings. Importing this module does not require PyTorch."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

ACTION_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0)  # fraction of ``size_frac`` targeted per symbol


class PPOSettings(BaseModel):
    """Hyperparameters for the PPO policy family.

    Rewards are log portfolio-value changes after all simulated fills and costs,
    multiplied by ``reward_scale`` (bps) for numerical stability. Scaling rewards does
    not change the optimal policy. With ``terminal_exit_cost`` the final reward also
    deducts the estimated cost of closing open positions at the far side of the
    quote. Without it, holding into the close would avoid the exit spread in training
    only, a known reward-hacking route. Evaluation is unaffected and identical for all
    agents.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    seed: int = 0
    total_steps: Annotated[int, Field(ge=256)] = 24_576
    rollout_steps: Annotated[int, Field(ge=64)] = 2_048
    epochs: Annotated[int, Field(ge=1)] = 4
    minibatch: Annotated[int, Field(ge=16)] = 256
    learning_rate: Annotated[float, Field(gt=0)] = 3e-4
    gamma: Annotated[float, Field(gt=0, le=1)] = 0.99
    gae_lambda: Annotated[float, Field(gt=0, le=1)] = 0.95
    clip: Annotated[float, Field(gt=0, lt=1)] = 0.2
    entropy_coef: Annotated[float, Field(ge=0)] = 0.01
    value_coef: Annotated[float, Field(ge=0)] = 0.5
    max_grad_norm: Annotated[float, Field(gt=0)] = 0.5
    hidden: Annotated[int, Field(ge=4)] = 64
    size_frac: Annotated[float, Field(gt=0, le=1)] = 0.2
    reward_scale: Annotated[float, Field(gt=0)] = 1e4
    terminal_exit_cost: bool = True
    kl_warning: Annotated[float, Field(gt=0)] = 0.05
