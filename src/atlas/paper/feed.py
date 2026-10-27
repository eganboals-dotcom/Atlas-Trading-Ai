"""Live market-data feed interface and a replay implementation.

``LiveFeed`` is what a real-time vendor adapter would implement: connect, yield events
in local-arrival order with receive timestamps, raise ``FeedDisconnected`` on drops, and
support bounded reconnects.

``ReplayLiveFeed`` replays a recorded or synthetic session through that interface. Its
clock is the replayed receive clock: with ``speed=0`` it runs as fast as possible on a
virtual clock (deterministic, used for tests and drills); with ``speed=1`` it paces
events to wall-clock time. It can inject faults (disconnects, failed reconnects, stale
silences, a clock going backwards, timestamps from the future, crashes) so every
failure path can be exercised without a live connection.

Unresolved by design: no vendor streaming adapter is implemented. Entitlements, message
formats, timestamp semantics, symbol limits, and rate limits must be verified from the
vendor's current official documentation first (see the specification, section 13).
"""

from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod

from atlas.data.events import MarketEvent
from atlas.data.providers import SessionRef, load_raw
from atlas.paper.config import FaultConfig
from atlas.timeutils import NS_PER_MS, NS_PER_S


class FeedDisconnected(ConnectionError):
    pass


class SimulatedCrash(RuntimeError):
    """Injected process failure (restart drills)."""


class LiveFeed(ABC):
    symbols: tuple[str, ...]

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def next_event(self) -> MarketEvent | None:
        """Next event in arrival order; ``None`` at end of session. May raise ``FeedDisconnected``."""

    @abstractmethod
    def reconnect(self) -> None:
        """One reconnect attempt; raises ``FeedDisconnected`` on failure."""

    @property
    @abstractmethod
    def position(self) -> int:
        """Number of events delivered so far (used to resume after a restart)."""

    @abstractmethod
    def seek(self, position: int) -> None:
        """Skip ahead so the next event delivered is the one at ``position``."""


def _arrival(e: MarketEvent) -> int:
    return e.receive_ts if e.receive_ts is not None else e.event_ts


class ReplayLiveFeed(LiveFeed):
    def __init__(self, ref: SessionRef, speed: float = 0.0, faults: FaultConfig | None = None) -> None:
        self.ref = ref
        self.symbols = ref.symbols
        self.speed = speed
        self.faults = faults or FaultConfig()
        self._events: list[MarketEvent] = []
        self._i = 0
        self._connected = False
        self._pending_failures = 0
        self._fired: set[tuple[str, int]] = set()
        self._wall0: float | None = None

    def connect(self) -> None:
        events = sorted(load_raw(self.ref), key=_arrival)  # local arrival order, not source order
        self._events = self._apply_static_faults(events)
        self._connected = True

    def _apply_static_faults(self, events: list[MarketEvent]) -> list[MarketEvent]:
        f = self.faults
        out = list(events)
        for at, gap_ms in f.stale_gaps:  # silence: drop everything that arrives within the gap
            if 0 <= at < len(out):
                start = _arrival(out[at])
                end = start + int(gap_ms * NS_PER_MS)
                out = out[: at + 1] + [e for e in out[at + 1:] if not (start < _arrival(e) < end)]
        for at in f.clock_backwards_at:
            if 0 < at < len(out):
                prev = _arrival(out[at - 1])
                e = out[at]
                out[at] = dataclasses.replace(e, receive_ts=prev - NS_PER_S, event_ts=min(e.event_ts, prev - NS_PER_S))
        for at in f.future_timestamp_at:
            if 0 <= at < len(out):
                e = out[at]
                out[at] = dataclasses.replace(e, event_ts=_arrival(e) + 10 * NS_PER_S)
        return out

    def next_event(self) -> MarketEvent | None:
        if not self._connected:
            raise FeedDisconnected("feed not connected")
        if self._i >= len(self._events):
            return None
        i = self._i
        if i in self.faults.disconnect_at_events and ("disconnect", i) not in self._fired:
            self._fired.add(("disconnect", i))
            self._connected = False
            self._pending_failures = self.faults.disconnect_failures
            raise FeedDisconnected(f"injected disconnect before event {i}")
        if self.faults.crash_at_event == i and ("crash", i) not in self._fired:
            self._fired.add(("crash", i))
            raise SimulatedCrash(f"injected crash before event {i}")
        event = self._events[i]
        self._i += 1
        if self.speed > 0:
            self._pace(event)
        return event

    def _pace(self, event: MarketEvent) -> None:
        t = _arrival(event)
        if self._wall0 is None:
            self._wall0, self._t0 = time.monotonic(), t
        wait = self._wall0 + (t - self._t0) / NS_PER_S / self.speed - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def reconnect(self) -> None:
        if self._pending_failures > 0:
            self._pending_failures -= 1
            raise FeedDisconnected("injected reconnect failure")
        self._connected = True

    @property
    def position(self) -> int:
        return self._i

    def seek(self, position: int) -> None:
        self._i = max(0, min(position, len(self._events)))
        # Faults scheduled before the resume point already happened in the previous process.
        for at in self.faults.disconnect_at_events:
            if at < self._i:
                self._fired.add(("disconnect", at))
        if self.faults.crash_at_event is not None and self.faults.crash_at_event < self._i:
            self._fired.add(("crash", self.faults.crash_at_event))
