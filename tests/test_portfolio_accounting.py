from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from atlas.environment.orders import Fill, Side
from atlas.environment.portfolio import AccountingError, Portfolio


def fill(side: Side, qty: int, price: float, fee: float = 0.0, symbol: str = "SPY", oid: int = 0) -> Fill:
    return Fill(oid, symbol, side, qty, price, fee, ts=0, quote_bid=price, quote_ask=price, quote_event_ts=0)


def test_buy_then_sell_realizes_pnl_and_fees() -> None:
    p = Portfolio(10_000)
    p.apply_fill(fill(Side.BUY, 10, 100.0, fee=1.0))
    assert p.cash == pytest.approx(10_000 - 1_000 - 1)
    assert p.positions["SPY"].avg_cost == pytest.approx(100.0)

    p.mark("SPY", 105.0)
    assert p.unrealized_pnl == pytest.approx(50.0)
    assert p.equity == pytest.approx(10_000 - 1 + 50)

    p.apply_fill(fill(Side.SELL, 4, 110.0, fee=0.5))
    assert p.positions["SPY"].realized_pnl == pytest.approx(40.0)
    assert p.positions["SPY"].quantity == 6
    assert p.fees_paid == pytest.approx(1.5)
    assert p.reconcile() == []


def test_average_cost_across_buys() -> None:
    p = Portfolio(100_000)
    p.apply_fill(fill(Side.BUY, 10, 100.0))
    p.apply_fill(fill(Side.BUY, 30, 104.0))
    assert p.positions["SPY"].avg_cost == pytest.approx((1000 + 3120) / 40)


def test_flat_position_resets_avg_cost() -> None:
    p = Portfolio(10_000)
    p.apply_fill(fill(Side.BUY, 5, 100.0))
    p.apply_fill(fill(Side.SELL, 5, 99.0))
    assert p.positions["SPY"].quantity == 0
    assert p.positions["SPY"].avg_cost == 0.0
    assert p.equity == pytest.approx(10_000 - 5)
    p.assert_reconciled()


def test_cannot_spend_more_than_cash() -> None:
    p = Portfolio(1_000)
    with pytest.raises(AccountingError, match="exceeds cash"):
        p.apply_fill(fill(Side.BUY, 10, 100.0, fee=0.01))


def test_cannot_short_sell() -> None:
    p = Portfolio(1_000)
    p.apply_fill(fill(Side.BUY, 2, 100.0))
    with pytest.raises(AccountingError, match="no short selling"):
        p.apply_fill(fill(Side.SELL, 3, 100.0))


def test_rejects_invalid_fill_values() -> None:
    with pytest.raises(AccountingError):
        Portfolio(1_000).apply_fill(fill(Side.BUY, 1, -5.0))


def test_max_affordable_qty_respects_fee() -> None:
    p = Portfolio(1_000)
    assert p.max_affordable_qty(100.0, lambda q: 0.0) == 10
    assert p.max_affordable_qty(100.0, lambda q: 1.0) == 9


fills_strategy = st.lists(
    st.tuples(
        st.sampled_from(["SPY", "QQQ"]),
        st.booleans(),
        st.integers(1, 50),
        st.floats(1.0, 500.0, allow_nan=False),
        st.floats(0.0, 5.0, allow_nan=False),
        st.floats(1.0, 500.0, allow_nan=False),
    ),
    max_size=60,
)


@settings(max_examples=200, deadline=None)
@given(fills_strategy)
def test_ledger_always_reconciles(ops) -> None:
    """Property test: any sequence of legal fills keeps cash, positions, and P&L consistent."""
    p = Portfolio(50_000)
    for i, (sym, is_buy, qty, price, fee, mark) in enumerate(ops):
        if is_buy:
            if qty * price + fee > p.cash:
                continue
            p.apply_fill(fill(Side.BUY, qty, price, fee, sym, i))
        else:
            qty = min(qty, p.position_qty(sym))
            if qty == 0:
                continue
            p.apply_fill(fill(Side.SELL, qty, price, fee, sym, i))
        p.mark(sym, mark)
        assert p.reconcile() == []
    assert p.cash >= 0
