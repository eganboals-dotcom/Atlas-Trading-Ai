"""Historical Alpaca data is re-sampled exactly like the live poller sees it."""

from datetime import date, time

from atlas.data.alpaca_history import (STRESS_DAYS, merge_polls, polled_events, random_window_start, sample_days,
                                       to_shares, window_bounds)
from atlas.data.events import Quote, Trade
from atlas.timeutils import NS_PER_S


def iso(ns: int) -> str:
    from datetime import UTC, datetime

    sec, frac = divmod(ns, NS_PER_S)
    return datetime.fromtimestamp(sec, UTC).strftime("%Y-%m-%dT%H:%M:%S") + f".{frac:09d}Z"


def test_polls_take_the_latest_quote_at_each_second_and_skip_unchanged() -> None:
    a, b, _, _ = window_bounds(date(2024, 6, 3), time(10, 0), 1)
    quotes = [{"t": iso(a + ms * 1_000_000), "bp": 100 + ms / 1e4, "ap": 100.01 + ms / 1e4, "bs": 3, "as": 4}
              for ms in (100, 400, 900, 1200, 5500)]
    trades = [{"t": iso(a + ms * 1_000_000), "p": 100.0, "s": 10} for ms in (300, 1500, 2500)]
    pairs = polled_events(quotes, trades, "SPY", a, b, "alpaca-sip")
    q = [(k, e.event_ts - a) for k, e in pairs if isinstance(e, Quote)]
    # polls at 0 s, 1 s, 2 s, ...: nothing at 0 s; the 900 ms quote at 1 s; the 1200 ms quote at 2 s; 5500 ms at 6 s
    assert q == [(1, 900_000_000), (2, 1_200_000_000), (6, 5_500_000_000)]
    t = [(k, e.event_ts - a) for k, e in pairs if isinstance(e, Trade)]
    assert t == [(2, 1_500_000_000), (4, 2_500_000_000)]  # every other poll, only new trades


def test_merge_orders_by_poll_and_never_runs_the_receive_clock_backwards() -> None:
    a, b, _, _ = window_bounds(date(2024, 6, 3), time(10, 0), 1)
    mk = lambda sym, ms: {"t": iso(a + ms * 1_000_000), "bp": 1.0, "ap": 1.01, "bs": 1, "as": 1}
    spy = polled_events([mk("SPY", 500), mk("SPY", 1500)], [], "SPY", a, b, "s")
    qqq = polled_events([mk("QQQ", 200), mk("QQQ", 1900)], [], "QQQ", a, b, "s")
    events = merge_polls([spy, qqq], a)
    recv = [e.receive_ts for e in events]
    assert recv == sorted(recv) and len(set(recv)) == len(recv)
    assert [e.symbol for e in events] == ["QQQ", "SPY", "SPY", "QQQ"]
    assert all(e.event_ts < e.receive_ts for e in events)


def test_round_lot_sizes_are_converted_to_shares() -> None:
    q = lambda bs: (0, Quote(symbol="SPY", event_ts=1, source="s", bid=1.0, ask=1.01, bid_size=bs, ask_size=bs))
    lots, unit = to_shares([q(2), q(5), q(3)])
    assert unit.startswith("round lots") and [e.bid_size for _, e in lots] == [200, 500, 300]
    shares, unit = to_shares([q(200), q(1500)])
    assert unit == "shares" and [e.bid_size for _, e in shares] == [200, 1500]


def test_day_sample_is_spread_out_includes_stress_days_and_is_reproducible() -> None:
    days = sample_days(date(2016, 6, 1), date(2026, 10, 23), 40, seed=1)
    assert days == sample_days(date(2016, 6, 1), date(2026, 10, 23), 40, seed=1)
    assert set(STRESS_DAYS) <= set(days) and all(d.weekday() < 5 for d in days)
    assert {d.year for d in days} == set(range(2016, 2027))


def test_windows_stay_inside_regular_hours() -> None:
    import random

    rng = random.Random(0)
    for _ in range(200):
        s = random_window_start(rng, 30)
        assert time(9, 30) <= s <= time(15, 30)


def test_polled_dataset_is_validated_like_the_live_session(tmp_path) -> None:
    # A trade stamped before the previous quote is normal for a poller; only symbol_type ordering keeps it.
    from atlas.data.alpaca_history import publish_dataset_session
    from atlas.data.providers import ParquetProvider, session_validation

    a, b, _, _ = window_bounds(date(2024, 6, 3), time(10, 0), 1)
    events = [Quote(symbol="SPY", event_ts=a + 900, source="s", bid=100.0, ask=100.01, bid_size=100, ask_size=100,
                    receive_ts=a + 1000),
              Trade(symbol="SPY", event_ts=a + 500, source="s", price=100.0, size=10, receive_ts=a + 2000),
              Quote(symbol="SPY", event_ts=a + 2500, source="s", bid=100.0, ask=100.01, bid_size=100, ask_size=100,
                    receive_ts=a + 3000)]
    for scope, dropped in (("symbol", 1), ("symbol_type", 0)):
        out = tmp_path / scope
        publish_dataset_session(out, "t", date(2024, 6, 3), events, ["SPY"], {"order_scope": scope}, {})
        ref = ParquetProvider(out).session_ref(date(2024, 6, 3))
        assert ref.order_scope == scope
        assert session_validation(ref).summary()["dropped_events"] == dropped
