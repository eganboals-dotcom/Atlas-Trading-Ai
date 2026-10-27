"""PPO policy network, checkpoint format, and the frozen ``PolicyAgent``.

The network applies one shared encoder to every symbol (symbol features + global
context). Per-symbol heads output action logits over ``ACTION_LEVELS``. A value head
reads the mean of the symbol embeddings.

Checkpoint = ``policy.pt`` (state dict) + ``policy.json`` (settings, normalizer, training
sessions, diagnostics). Agents reference a checkpoint as ``path#sha256``. Loading
verifies the hash, so a record always means exactly one set of weights.

``PolicyAgent`` is frozen: it acts greedily (argmax) and never updates its weights during
evaluation or paper trading.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from atlas.agents.base import Agent, AgentMetadata, AgentRecord, Observation, TargetPositions
from atlas.agents.genome import ParamSpec
from atlas.environment.market_state import PortfolioState
from atlas.learning.env import targets_from_action
from atlas.learning.observation import N_GLOBAL_INPUTS, N_SYMBOL_INPUTS, FeatureNormalizer, build_observation
from atlas.learning.settings import ACTION_LEVELS, PPOSettings

# The network is tiny: one thread is fastest, avoids oversubscribing parallel evaluation
# workers, and keeps CPU results bit-for-bit reproducible.
torch.set_num_threads(1)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_SYMBOL_INPUTS + N_GLOBAL_INPUTS, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, len(ACTION_LEVELS))
        self.v = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, sym: torch.Tensor, glob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """sym [B, S, F], glob [B, G] -> logits [B, S, A], value [B]."""
        x = torch.cat([sym, glob.unsqueeze(1).expand(-1, sym.shape[1], -1)], dim=-1)
        h = self.encoder(x)
        return self.pi(h), self.v(h.mean(dim=1)).squeeze(-1)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_policy(net: PolicyNet, settings: PPOSettings, normalizer: FeatureNormalizer, out_dir: Path,
                extra: dict[str, Any]) -> str:
    """Write the checkpoint; returns the artifact reference ``path#sha256``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / "policy.pt"
    torch.save(net.state_dict(), weights)
    sha = file_sha256(weights)
    meta = {"settings": settings.model_dump(), "normalizer": normalizer.to_dict(), "weights_sha256": sha, **extra}
    (out_dir / "policy.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return f"{weights.resolve().as_posix()}#{sha}"


@lru_cache(maxsize=32)
def load_policy(artifact: str) -> tuple[PolicyNet, PPOSettings, FeatureNormalizer]:
    path_str, _, sha = artifact.rpartition("#")
    weights = Path(path_str)
    if not sha or file_sha256(weights) != sha:
        raise ValueError(f"policy weights {weights} do not match pinned sha256 {sha!r}")
    meta = json.loads((weights.parent / "policy.json").read_text(encoding="utf-8"))
    settings = PPOSettings.model_validate(meta["settings"])
    net = PolicyNet(settings.hidden)
    net.load_state_dict(torch.load(weights, weights_only=True))
    net.eval()
    return net, settings, FeatureNormalizer.from_dict(meta["normalizer"])


class PolicyAgent(Agent):
    """A frozen PPO policy behind the common agent contract."""

    family: ClassVar[str] = "ppo"
    param_space: ClassVar[dict[str, ParamSpec]] = {}
    default_genome: ClassVar[dict[str, float]] = {}

    def __init__(self, record: AgentRecord) -> None:
        super().__init__(record)
        if not record.artifact:
            raise ValueError("ppo agents need an artifact reference (path#sha256)")
        self.net, self.settings, self.normalizer = load_policy(record.artifact)

    def act(self, obs: Observation) -> TargetPositions:
        assert self.runtime is not None
        symbols = list(self.runtime.symbols)
        state = PortfolioState.of(obs.cash, obs.equity, dict(obs.positions))
        sym, glob = build_observation(obs.features, state, symbols, self.normalizer, obs.now_ns)
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(sym).unsqueeze(0), torch.from_numpy(glob).unsqueeze(0))
        action = logits[0].argmax(dim=-1).numpy()
        return targets_from_action(action, dict(obs.features), state, symbols, self.settings.size_frac,
                                   self.runtime.limits)

    def learn(self, experience_batch: Any) -> None:
        raise NotImplementedError("PolicyAgent is frozen; train offline with `atlas.cli train-ppo`")

    def metadata(self) -> AgentMetadata:
        r = self.record
        sha = (r.artifact or "").rpartition("#")[2][:16]
        return AgentMetadata(r.agent_id, r.family, r.generation, tuple(r.parents), r.origin, sha,
                             population_version="ppo-v1", trainable=True)


def greedy_actions(net: PolicyNet, sym: np.ndarray, glob: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(sym).unsqueeze(0), torch.from_numpy(glob).unsqueeze(0))
    return logits[0].argmax(dim=-1).numpy()
