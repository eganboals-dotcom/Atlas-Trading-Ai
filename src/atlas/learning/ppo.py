"""Proximal Policy Optimization (clipped objective, GAE) for the Atlas environment.

The implementation is deliberately small and explicit so every quantity it logs can be
audited. It trains only on the sessions it is given, which are training sessions
chosen by the caller. It records, per update:

* training reward, and the realized net log return of finished episodes. The gap between
  them flags reward hacking: training reward should track realized P&L;
* the action distribution per level, plus fills per episode (to detect collapse or churning);
* policy/value loss, entropy, approximate KL, clip fraction, and explained variance.
  KL spikes and non-finite losses are flagged.

Determinism: seeded PyTorch and NumPy, single-threaded CPU. The same seed and sessions
give the same weights.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from atlas.config import AgentLimits, SimulatorConfig
from atlas.data.providers import SessionRef
from atlas.learning.env import AtlasTradingEnv
from atlas.learning.observation import fit_normalizer
from atlas.learning.policy import PolicyNet, save_policy
from atlas.learning.settings import ACTION_LEVELS, PPOSettings
from atlas.logging_utils import get_logger

log = get_logger(__name__)


class TrainingError(RuntimeError):
    pass


@dataclass
class PPOResult:
    artifact: str
    out_dir: Path
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        eps = self.episodes
        last = eps[-max(1, len(eps) // 4):] if eps else []
        gaps = [abs(e["reward_log_return"] - e["realized_log_return"]) for e in eps]
        return {
            "artifact": self.artifact,
            "updates": len(self.diagnostics),
            "episodes": len(eps),
            "final_quarter_mean_realized_bps": statistics.fmean(e["realized_log_return"] for e in last) * 1e4
            if last else None,
            "mean_abs_reward_vs_realized_gap_bps": statistics.fmean(gaps) * 1e4 if gaps else None,
            "final_action_distribution": self.diagnostics[-1]["action_distribution"] if self.diagnostics else None,
            "final_entropy": self.diagnostics[-1]["entropy"] if self.diagnostics else None,
            "max_approx_kl": max((d["approx_kl"] for d in self.diagnostics), default=None),
            "warnings": self.warnings,
        }


def train_ppo(settings: PPOSettings, train_refs: list[SessionRef], sim: SimulatorConfig, limits: AgentLimits,
              decision_interval_ms: float, out_dir: Path) -> PPOResult:
    torch.manual_seed(settings.seed)
    torch.use_deterministic_algorithms(True)
    normalizer = fit_normalizer(train_refs, decision_interval_ms, sim.stale_quote_ms)
    env = AtlasTradingEnv(train_refs, sim, limits, decision_interval_ms, normalizer, settings.size_frac,
                          settings.reward_scale, settings.terminal_exit_cost, seed=settings.seed)
    net = PolicyNet(settings.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=settings.learning_rate, eps=1e-5)
    n_sym = len(env.symbols)
    result = PPOResult("", out_dir)

    obs, _ = env.reset(seed=settings.seed)
    n_updates = settings.total_steps // settings.rollout_steps
    for update in range(n_updates):
        T = settings.rollout_steps
        sym_buf = np.zeros((T, n_sym, obs["symbols"].shape[1]), np.float32)
        glob_buf = np.zeros((T, obs["global"].shape[0]), np.float32)
        act_buf = np.zeros((T, n_sym), np.int64)
        logp_buf, val_buf, rew_buf, done_buf = (np.zeros(T, np.float32) for _ in range(4))
        for t in range(T):
            sym_buf[t], glob_buf[t] = obs["symbols"], obs["global"]
            with torch.no_grad():
                logits, value = net(torch.from_numpy(obs["symbols"]).unsqueeze(0),
                                    torch.from_numpy(obs["global"]).unsqueeze(0))
                dist = Categorical(logits=logits[0])
                action = dist.sample()
            act_buf[t] = action.numpy()
            logp_buf[t] = dist.log_prob(action).sum().item()
            val_buf[t] = value.item()
            obs, reward, done, _, info = env.step(act_buf[t])
            rew_buf[t], done_buf[t] = reward, float(done)
            if done:
                result.episodes.append({
                    "update": update, "session": info["session"], "fills": info["fills"],
                    "reward_log_return": env.episode_reward / settings.reward_scale,
                    "realized_log_return": info["realized_log_return"],
                    "adjusted_log_return": info["adjusted_log_return"],
                })
                obs, _ = env.reset()
        with torch.no_grad():
            _, last_value = net(torch.from_numpy(obs["symbols"]).unsqueeze(0),
                                torch.from_numpy(obs["global"]).unsqueeze(0))
        adv = np.zeros(T, np.float32)
        gae, next_value = 0.0, last_value.item()
        for t in reversed(range(T)):
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + settings.gamma * next_value * nonterminal - val_buf[t]
            gae = delta + settings.gamma * settings.gae_lambda * nonterminal * gae
            adv[t], next_value = gae, val_buf[t]
        returns = adv + val_buf

        diag = _update(net, opt, settings, sym_buf, glob_buf, act_buf, logp_buf, adv, returns, val_buf)
        counts = np.bincount(act_buf.ravel(), minlength=len(ACTION_LEVELS))
        diag.update(update=update, mean_step_reward=float(rew_buf.mean()),
                    action_distribution=[round(float(c) / float(counts.sum()), 4) for c in counts])
        result.diagnostics.append(diag)
        if not all(math.isfinite(diag[k]) for k in ("policy_loss", "value_loss", "entropy")):
            raise TrainingError(f"non-finite loss at update {update}: {diag}")
        if diag["approx_kl"] > settings.kl_warning:
            result.warnings.append(f"update {update}: approx KL {diag['approx_kl']:.3f} > {settings.kl_warning}")
        log.info("ppo update", extra={k: diag[k] for k in ("update", "approx_kl", "entropy", "mean_step_reward")})

    dist = result.diagnostics[-1]["action_distribution"] if result.diagnostics else []
    if dist and max(dist) > 0.99:
        result.warnings.append(f"action distribution collapsed to level {int(np.argmax(dist))} ({max(dist):.1%})")
    fills = [e["fills"] for e in result.episodes]
    if fills and statistics.fmean(fills) == 0:
        result.warnings.append("policy never trades in training episodes")

    result.artifact = save_policy(net, settings, normalizer, out_dir, {
        "train_sessions": [r.session_date.isoformat() for r in train_refs],
        "symbols": env.symbols,
        "decision_interval_ms": decision_interval_ms,
        "summary": {k: v for k, v in result.summary().items() if k != "artifact"},
    })
    return result


def _update(net, opt, s: PPOSettings, sym, glob, act, old_logp, adv, returns, old_values) -> dict[str, float]:
    T = len(act)
    sym_t, glob_t = torch.from_numpy(sym), torch.from_numpy(glob)
    act_t, old_logp_t = torch.from_numpy(act), torch.from_numpy(old_logp)
    adv_t = torch.from_numpy(adv)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.from_numpy(returns)
    perm_rng = np.random.default_rng(s.seed)
    stats: dict[str, list[float]] = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")}
    for _ in range(s.epochs):
        for start in range(0, T, s.minibatch):
            idx = torch.from_numpy(perm_rng.permutation(T)[start:start + s.minibatch])
            logits, value = net(sym_t[idx], glob_t[idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_t[idx]).sum(-1)
            ratio = torch.exp(logp - old_logp_t[idx])
            a = adv_t[idx]
            policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - s.clip, 1 + s.clip) * a).mean()
            value_loss = 0.5 * (ret_t[idx] - value).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + s.value_coef * value_loss - s.entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), s.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > s.clip).float().mean().item())
    var = float(np.var(returns))
    explained = 1.0 - float(np.var(returns - old_values)) / var if var > 0 else 0.0
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["explained_variance"] = explained
    return out
