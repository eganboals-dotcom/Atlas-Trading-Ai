"""Independent risk manager and kill switch.

The risk manager checks every proposed order *after* the agent and its adapter. It
holds its own immutable copy of the limits. Agents never get a reference to it, so they
cannot change or bypass it.

Per-order checks, in order:
1. account halted -> reject;
2. symbol not allowed -> reject;
3. order size above ``max_order_shares`` or ``max_order_value`` -> reduce (cap);
4. a buy that would push the symbol's position over ``max_position_shares`` or
   ``max_position_value`` -> reject;
5. a buy that would push gross/net exposure over its limit -> reduce to fit, else reject;
6. ``max_orders_per_minute`` already reached -> reject.

Sells only reduce risk in a long-only, cash-funded account. They skip checks 4-5 and
are capped at the held quantity, but checks 1, 2, 3, and 6 still apply. A halted account
proposes nothing; positions stay as they are for manual review. Account-level checks run on every equity update: a daily loss or a
drawdown beyond its limit halts the account.

The kill switch is global and persistent. It is triggered manually (a ``KILL`` file
in the session directory, or the ``paper-kill`` command) or by a critical failure. It
stops all new actions, cancels all working simulated orders, and stays engaged across
restarts until an operator resets it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas.environment.orders import OrderIntent, Side
from atlas.paper.config import RiskConfig
from atlas.timeutils import NS_PER_S


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    quantity: int
    rule: str | None = None
    reason: str | None = None


class RiskManager:
    def __init__(self, cfg: RiskConfig, account_id: str, start_equity: float) -> None:
        self.cfg = cfg
        self.account_id = account_id
        self.day_start_equity = start_equity
        self.peak_equity = start_equity
        self.halted: str | None = None
        self._order_times: deque[int] = deque()
        self.rejections: dict[str, int] = {}

    def _no(self, rule: str, reason: str) -> RiskDecision:
        self.rejections[rule] = self.rejections.get(rule, 0) + 1
        return RiskDecision(False, 0, rule, reason)

    def check_order(self, intent: OrderIntent, now_ns: int, price: float, equity: float,
                    positions: dict[str, int], marks: dict[str, float]) -> RiskDecision:
        c = self.cfg
        if self.halted:
            return self._no("halted", f"account halted: {self.halted}")
        if intent.symbol not in c.allowed_symbols:
            return self._no("symbol", f"{intent.symbol} not in allowed_symbols")
        if price <= 0 or equity <= 0:
            return self._no("price", "no valid price or equity")

        qty = min(intent.quantity, c.max_order_shares, int(c.max_order_value // price))
        rule = "order_size" if qty < intent.quantity else None
        if intent.side is Side.BUY:
            held = positions.get(intent.symbol, 0)
            if held + qty > c.max_position_shares or (held + qty) * price > c.max_position_value:
                return self._no("max_position", f"buy would take {intent.symbol} to {held + qty} shares "
                                                 f"(${(held + qty) * price:,.0f}), above the per-symbol limit")
            gross = sum(q * marks.get(s, price) for s, q in positions.items())
            room = min(c.max_gross_exposure, c.max_net_exposure) * equity - gross
            fit = int(max(0.0, room) // price)
            if fit < qty:
                qty, rule = fit, "exposure"
        else:
            qty = min(qty, positions.get(intent.symbol, 0))
        if qty <= 0:
            return self._no(rule or "size", "order reduced to zero by limits")

        window_start = now_ns - 60 * NS_PER_S
        while self._order_times and self._order_times[0] <= window_start:
            self._order_times.popleft()
        if len(self._order_times) >= c.max_orders_per_minute:
            return self._no("order_rate", f"{c.max_orders_per_minute} orders already in the last minute")
        self._order_times.append(now_ns)
        return RiskDecision(True, qty, rule, None if rule is None else f"quantity reduced by {rule} limit")

    def on_equity(self, equity: float) -> str | None:
        """Account-level limits. Returns a breach reason the first time one occurs."""
        if self.halted:
            return None
        self.peak_equity = max(self.peak_equity, equity)
        if equity <= self.day_start_equity * (1 - self.cfg.max_daily_loss):
            self.halted = (f"daily loss {1 - equity / self.day_start_equity:.2%} reached limit "
                           f"{self.cfg.max_daily_loss:.2%}")
        elif equity <= self.peak_equity * (1 - self.cfg.max_drawdown):
            self.halted = f"drawdown {1 - equity / self.peak_equity:.2%} reached limit {self.cfg.max_drawdown:.2%}"
        return self.halted

    def to_state(self) -> dict[str, Any]:
        return {"day_start_equity": self.day_start_equity, "peak_equity": self.peak_equity, "halted": self.halted,
                "order_times": list(self._order_times), "rejections": self.rejections}

    def restore(self, d: dict[str, Any]) -> None:
        self.day_start_equity, self.peak_equity = d["day_start_equity"], d["peak_equity"]
        self.halted, self.rejections = d["halted"], dict(d["rejections"])
        self._order_times = deque(d["order_times"])


class KillSwitch:
    FILE = "KILL"

    def __init__(self, session_dir: Path) -> None:
        self.path = session_dir / self.FILE
        self.reason: str | None = None

    @property
    def engaged(self) -> bool:
        return self.reason is not None

    def trigger(self, reason: str) -> bool:
        """Engage; returns True if this call engaged it."""
        if self.engaged:
            return False
        self.reason = reason
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(reason, encoding="utf-8")
        return True

    def poll_file(self) -> str | None:
        """Returns the reason if an operator created a KILL file; the caller then calls ``trigger``."""
        if not self.engaged and self.path.exists():
            return "manual: " + (self.path.read_text(encoding="utf-8").strip() or "KILL file present")
        return None


def request_kill(session_dir: Path, reason: str) -> Path:
    """Operator command: engage the kill switch of a running or stopped session."""
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / KillSwitch.FILE
    path.write_text(reason or "operator kill", encoding="utf-8")
    return path
