"""Portfolio ledger: cash, positions, fills, fees, realized/unrealized P&L, and equity.

Initial scope is long-only and cash-funded. The ledger itself refuses fills that would
make cash or a position negative — those are accounting bugs, not risk decisions.

Conventions:
* ``avg_cost`` excludes fees; fees are tracked separately in ``fees_paid``.
* ``realized_pnl`` and ``unrealized_pnl`` are gross of fees.
* ``net_pnl = realized + unrealized - fees = equity - initial_cash``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from atlas.environment.orders import Fill, Side

_EPS = 1e-9


class AccountingError(RuntimeError):
    """A fill or state that violates ledger invariants."""


@dataclass(slots=True)
class Position:
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0


class Portfolio:
    def __init__(self, initial_cash: float) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fees_paid = 0.0
        self.positions: dict[str, Position] = defaultdict(Position)
        self.marks: dict[str, float] = {}
        self.fills: list[Fill] = []
        self._buy_notional = 0.0
        self._sell_notional = 0.0

    # ------------------------------------------------------------------ updates

    def max_affordable_qty(self, price: float, fee_for: Any) -> int:
        """Largest buy quantity whose notional plus fee fits in cash. ``fee_for(qty) -> fee``."""
        if price <= 0:
            return 0
        qty = int(self.cash // price)
        while qty > 0 and qty * price + fee_for(qty) > self.cash + _EPS:
            qty -= 1
        return qty

    def apply_fill(self, fill: Fill) -> None:
        if fill.quantity <= 0 or fill.price <= 0 or fill.fee < 0:
            raise AccountingError(f"invalid fill values: {fill}")
        pos = self.positions[fill.symbol]
        notional = fill.quantity * fill.price
        if fill.side is Side.BUY:
            cost = notional + fill.fee
            if cost > self.cash + _EPS * max(1.0, cost):
                raise AccountingError(f"buy of {cost:.2f} exceeds cash {self.cash:.2f} (no margin)")
            new_qty = pos.quantity + fill.quantity
            pos.avg_cost = (pos.quantity * pos.avg_cost + notional) / new_qty
            pos.quantity = new_qty
            self.cash -= cost
            self._buy_notional += notional
        else:
            if fill.quantity > pos.quantity:
                raise AccountingError(
                    f"sell {fill.quantity} {fill.symbol} exceeds position {pos.quantity} (no short selling)"
                )
            pos.realized_pnl += (fill.price - pos.avg_cost) * fill.quantity
            pos.quantity -= fill.quantity
            if pos.quantity == 0:
                pos.avg_cost = 0.0
            self.cash += notional - fill.fee
            self._sell_notional += notional
        self.fees_paid += fill.fee
        self.marks.setdefault(fill.symbol, fill.price)
        self.fills.append(fill)

    def mark(self, symbol: str, price: float) -> None:
        if price <= 0:
            raise AccountingError(f"invalid mark {price} for {symbol}")
        self.marks[symbol] = price

    # ------------------------------------------------------------------ queries

    def position_qty(self, symbol: str) -> int:
        return self.positions[symbol].quantity if symbol in self.positions else 0

    @property
    def market_value(self) -> float:
        return sum(p.quantity * self.marks[s] for s, p in self.positions.items() if p.quantity)

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.quantity * (self.marks[s] - p.avg_cost) for s, p in self.positions.items() if p.quantity)

    @property
    def gross_exposure(self) -> float:
        return self.market_value  # long-only: gross == net

    def reconcile(self) -> list[str]:
        """Independent cross-checks of ledger state. Returns a list of discrepancies."""
        problems: list[str] = []
        tol = 1e-6 * max(1.0, self.initial_cash)

        expected_cash = self.initial_cash - self._buy_notional + self._sell_notional - self.fees_paid
        if abs(self.cash - expected_cash) > tol:
            problems.append(f"cash {self.cash} != expected {expected_cash}")

        fee_sum = sum(f.fee for f in self.fills)
        if abs(fee_sum - self.fees_paid) > tol:
            problems.append(f"fees_paid {self.fees_paid} != sum of fill fees {fee_sum}")

        qty_from_fills: dict[str, int] = defaultdict(int)
        for f in self.fills:
            qty_from_fills[f.symbol] += f.quantity if f.side is Side.BUY else -f.quantity
        for symbol in set(qty_from_fills) | set(self.positions):
            if qty_from_fills[symbol] != self.position_qty(symbol):
                problems.append(f"{symbol}: position {self.position_qty(symbol)} != fills {qty_from_fills[symbol]}")

        net = self.realized_pnl + self.unrealized_pnl - self.fees_paid
        if abs((self.equity - self.initial_cash) - net) > tol:
            problems.append(f"equity change {self.equity - self.initial_cash} != net pnl {net}")
        if self.cash < -tol:
            problems.append(f"negative cash {self.cash}")
        return problems

    def assert_reconciled(self) -> None:
        problems = self.reconcile()
        if problems:
            raise AccountingError("; ".join(problems))

    def snapshot(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "market_value": self.market_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees_paid": self.fees_paid,
            "net_pnl": self.equity - self.initial_cash,
            "positions": {
                s: {"quantity": p.quantity, "avg_cost": p.avg_cost, "realized_pnl": p.realized_pnl}
                for s, p in sorted(self.positions.items())
            },
        }
