"""The paper-trading process.

Live-style market data goes through the same validation, features, and frozen agents as
research. Order proposals go through the independent risk manager into the internal
``PaperBroker`` only.

Accounts: one **champion**, up to five **shadows** for comparison, and a buy-and-hold
**benchmark**. Each has its own paper broker, ledger, and risk state. Agents are frozen:
nothing in this module trains or mutates a model.

Safe state (global): when data is stale or missing, the feed is disconnected, or the
clock is anomalous, no new actions are taken and all working simulated orders are
cancelled. An alert is raised and the session resumes automatically once the condition
clears. If reconnection fails after the configured retries, the kill switch engages. A
daily-loss or drawdown breach halts only that account.

Restart: state is flushed atomically every ``state_flush_events`` events and at the end.
After a crash, ``resume`` restores every ledger and risk state. Orders that were working
at the crash are cancelled, never assumed filled. The feed continues from the last
flushed position, and an engaged kill switch stays engaged.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from atlas import __version__
from atlas.agents.adapter import AgentStrategy
from atlas.agents.base import POPULATION_VERSION, AgentRecord
from atlas.agents.baselines import baseline_records
from atlas.agents.registry import create_agent
from atlas.brokers.base import BrokerExecution, LiveTradingDisabledError
from atlas.config import SimulatorConfig
from atlas.data.events import MarketEvent, Quote, Trade
from atlas.data.providers import make_provider
from atlas.data.validation import IssueKind, StreamValidator
from atlas.environment.market_state import MarketView, PortfolioState
from atlas.environment.orders import OrderIntent, OrderStatus, Side, SimOrder
from atlas.logging_utils import get_logger
from atlas.paper.broker import PaperBroker
from atlas.paper.config import CandidateSource, PaperConfig, RiskConfig
from atlas.paper.feed import FeedDisconnected, LiveFeed, ReplayLiveFeed
from atlas.paper.risk import KillSwitch, RiskManager
from atlas.timeutils import MARKET_TZ, NS_PER_MS, NS_PER_S

log = get_logger(__name__)
STATE_VERSION = 1
MAX_BACKOFF_MS = 30_000.0
# Always present in every paper session, whatever the config says: the coin flip is a ranked bot (with
# its own fund share) that every strategy has to beat; buy-and-hold is the unranked market yardstick.
PERMANENT_BASELINES = {"Flip": "baseline_coin_flip", "Hold": "baseline_buy_hold"}  # account name -> family
ROLES = {"Flip": "control", "Hold": "benchmark"}
STOP_FILE = "STOP"  # graceful, resumable stop requested by `paper-stop` (unlike KILL, which halts trading)


class HealthCheckError(RuntimeError):
    pass


# ----------------------------------------------------------------- candidates


def resolve_candidate(src: CandidateSource) -> AgentRecord:
    if src.source == "baseline":
        for r in baseline_records():
            if r.family == src.baseline:
                return r
        raise HealthCheckError(f"unknown baseline {src.baseline!r}")
    assert src.path is not None
    if not src.path.is_file():
        raise HealthCheckError(f"candidate file not found: {src.path}")
    data = json.loads(src.path.read_text(encoding="utf-8"))
    if src.source == "design":
        if not data.get("design_fingerprint") or not data.get("final_candidate"):
            raise HealthCheckError(f"{src.path} is not a frozen design with a final candidate")
        return AgentRecord.from_dict(data["final_candidate"])
    champions = data.get("archives", {}).get("champions", [])
    if src.rank >= len(champions):
        raise HealthCheckError(f"{src.path} has {len(champions)} champions; rank {src.rank} unavailable")
    if data.get("population_version") not in (None, POPULATION_VERSION):
        raise HealthCheckError(f"model mismatch: checkpoint population {data['population_version']!r} "
                               f"!= {POPULATION_VERSION!r}")
    return AgentRecord.from_dict(champions[src.rank]["record"])


def make_broker(account_id: str, sim: SimulatorConfig, seed: int, audit,
                initial_cash: float | None = None) -> BrokerExecution:
    """The only broker factory in paper mode. It can only ever build a PaperBroker."""
    from atlas.environment.portfolio import Portfolio

    broker = PaperBroker(account_id, sim, seed, audit,
                         Portfolio(initial_cash) if initial_cash is not None else None)
    ensure_not_live(broker)
    return broker


def ensure_not_live(broker: object) -> None:
    if getattr(type(broker), "is_live", True) is not False or not isinstance(broker, PaperBroker):
        raise LiveTradingDisabledError(f"paper mode refuses broker {type(broker).__name__}: only PaperBroker is allowed")


def config_fingerprint(cfg: PaperConfig, sim: SimulatorConfig, risk: RiskConfig) -> str:
    """Identity of a session's settings. Fault injection is excluded so drills can be resumed."""
    payload = {
        "paper": cfg.model_dump(mode="json", exclude={"feed": {"faults"}, "output_dir": True}),
        "simulator": sim.model_dump(mode="json", exclude={"reports_dir", "demo_strategy"}),
        "risk": risk.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


LOCK_FILE = "RUNNING.lock"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # never os.kill on Windows: it terminates the process
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(session_dir: Path) -> None:
    lock = session_dir / LOCK_FILE
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip() or 0)
        except ValueError:
            pid = 0
        if pid != os.getpid() and pid_alive(pid):
            raise HealthCheckError(f"session {session_dir.name} is already running (process {pid}); use that window, "
                                   "or stop it first with paper-stop")
    lock.write_text(str(os.getpid()), encoding="utf-8")


def release_lock(session_dir: Path) -> None:
    lock = session_dir / LOCK_FILE
    try:
        if lock.exists() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def keep_awake(on: bool) -> bool:
    """Ask Windows not to sleep while a live session runs (released when it ends). No-op elsewhere."""
    if os.name != "nt":
        return False
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    flags = es_continuous | (es_system_required if on else 0)
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags)) and on


def today_market_date() -> date:
    return datetime.now(MARKET_TZ).date()


def session_date_for(cfg: PaperConfig, feed: LiveFeed | None) -> date:
    """Replays use the configured date; live feeds trade today (US/Eastern) only."""
    if cfg.feed.kind == "replay":
        assert cfg.feed.session_date is not None
        return cfg.feed.session_date
    if feed is not None and getattr(feed, "session_date", None):
        return feed.session_date
    return cfg.feed.session_date or today_market_date()


# ------------------------------------------------------------------ accounts


@dataclass
class Account:
    account_id: str
    role: str
    record: AgentRecord
    model: dict[str, Any]
    strategy: AgentStrategy
    broker: PaperBroker
    risk: RiskManager
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    decision_wall_ns: deque = field(default_factory=lambda: deque(maxlen=20_000))
    next_curve_ts: int = 0
    inception_capital: float | None = None  # fund mode: the bot's day-1 share


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")

    def flush(self) -> int:
        """Flush to disk; returns the durable byte offset."""
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self._fh.tell()

    def truncate(self, offset: int) -> None:
        """Drop records written after the last state flush (they are replayed after a restart)."""
        self._fh.flush()
        self._fh.truncate(offset)
        self._fh.seek(offset)

    def close(self) -> None:
        self._fh.close()


class Alerts:
    def __init__(self, path: Path, console: bool, file: bool) -> None:
        self.path, self.console, self.file = path, console, file
        self.count = 0

    def offset(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def truncate(self, offset: int) -> None:
        if self.path.exists():
            with self.path.open("r+b") as fh:
                fh.truncate(offset)

    def raise_(self, level: str, code: str, message: str, ts: int, account: str | None = None) -> None:
        self.count += 1
        rec = {"ts": ts, "level": level, "code": code, "message": message, "account": account,
               "wall_time": datetime.now(UTC).isoformat()}
        if self.file:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        if self.console:
            print(f"[ALERT {level}] {code}: {message}" + (f" ({account})" if account else ""), file=sys.stderr)
        log.warning("alert", extra={f"alert_{k}": v for k, v in rec.items()})  # 'message' is reserved by logging


# ------------------------------------------------------------------- session


class PaperSession:
    def __init__(self, cfg: PaperConfig, sim: SimulatorConfig, risk: RiskConfig, base_dir: Path = Path("."),
                 feed: LiveFeed | None = None, resume: bool = False, reset_kill_switch: bool = False,
                 reset_fund: bool = False) -> None:
        if cfg.mode != "paper" or cfg.live_orders_enabled is not False:  # defense in depth; config already forbids
            raise LiveTradingDisabledError("paper session requires mode=paper and live_orders_enabled=false")
        self.cfg, self.sim, self.risk_cfg = cfg, sim, risk
        self.session_date = session_date_for(cfg, feed)
        self.dir = cfg.output_dir / cfg.session_name.replace("{date}", self.session_date.isoformat())
        self.fingerprint = config_fingerprint(cfg, sim, risk)
        self.base_dir = base_dir
        self._feed_override = feed
        self.resume = resume
        self.reset_kill_switch = reset_kill_switch
        self.reset_fund = reset_fund
        self.fund = None
        self.symbols: list[str] = []
        self.accounts: list[Account] = []
        self.safe: dict[str, int] = {}  # reason -> since ts
        self.safe_periods: list[dict[str, Any]] = []
        self.counters: dict[str, int] = {"events": 0, "dropped_events": 0, "disconnects": 0, "reconnect_attempts": 0,
                                         "clock_anomalies": 0, "feed_gaps": 0, "risk_rejections": 0}
        self.validation_counts: dict[str, int] = {}
        self.started_wall = datetime.now(UTC).isoformat()

    # -- startup ---------------------------------------------------------------------

    def health_checks(self) -> None:
        """Pre-session checks. Any failure raises ``HealthCheckError`` and nothing starts."""
        problems: list[str] = []
        if self.dir.exists() and (self.dir / "state.json").exists() and not self.resume:
            problems.append(f"session {self.dir} already exists; pass --resume to continue it")
        if self.resume and not (self.dir / "state.json").exists():
            problems.append(f"nothing to resume in {self.dir}")
        kill = self.dir / KillSwitch.FILE
        if kill.exists():
            if self.reset_kill_switch:
                kill.unlink()
            else:
                problems.append(f"kill switch engaged ({kill}); an operator must review and pass --reset-kill-switch")
        if problems:
            raise HealthCheckError("; ".join(problems))

        self.feed = self._feed_override or self._make_feed()
        self.symbols = list(self.feed.symbols)
        not_allowed = sorted(set(self.symbols) - set(self.risk_cfg.allowed_symbols))
        if not_allowed:
            raise HealthCheckError(f"feed symbols {not_allowed} are not in risk.allowed_symbols")

        sources = [(self.cfg.candidate.name or "champion", "champion", self.cfg.candidate)]
        # Permanent accounts are always added here and can't be removed (or duplicated) by any config.
        shadows = [s for s in self.cfg.shadows
                   if not (s.source == "baseline" and s.baseline in PERMANENT_BASELINES.values())]
        sources += [(s.name or f"shadow-{i + 1}", "shadow", s) for i, s in enumerate(shadows)]
        sources += [(acct, ROLES[acct], CandidateSource(source="baseline", baseline=fam))
                    for acct, fam in PERMANENT_BASELINES.items()]
        names = [acct.lower() for acct, _, _ in sources]
        if len(set(names)) != len(names):
            raise HealthCheckError(f"bot names must be unique (Flip and Hold are reserved): {[a for a, _, _ in sources]}")
        self.dir.mkdir(parents=True, exist_ok=True)
        acquire_lock(self.dir)  # one process per session: a second launcher refuses instead of corrupting files
        self.audit = AuditLog(self.dir / "audit.jsonl")
        self.alerts = Alerts(self.dir / "alerts.jsonl", self.cfg.alerts.console, self.cfg.alerts.file)
        self.kill = KillSwitch(self.dir)
        records = [(acct, role, src, resolve_candidate(src)) for acct, role, src in sources]
        capital, inception = self._fund_capital(records)
        for i, (role_id, role, src, record) in enumerate(records):
            try:
                agent = create_agent(record)  # validates genome / pinned weights (model mismatch -> refuse)
            except (ValueError, KeyError, ImportError) as exc:
                raise HealthCheckError(f"model mismatch for {role_id} ({record.agent_id}): {exc}") from exc
            self._check_model_symbols(role_id, record)
            meta = agent.metadata()
            strategy = AgentStrategy(agent, self.symbols, self.cfg.agent_limits,
                                     int(self.cfg.decision_interval_ms * NS_PER_MS),
                                     int(self.risk_cfg.stale_data_ms * NS_PER_MS), seed=self.sim.seed + i)
            broker = make_broker(role_id, self.sim, self.sim.seed + 100 + i, self.audit.write,
                                 capital.get(role_id))
            self.accounts.append(Account(
                role_id, role, record,
                {"agent_id": meta.agent_id, "family": meta.family, "genome_hash": meta.genome_hash,
                 "population_version": meta.population_version, "artifact": record.artifact,
                 "source": src.model_dump(mode="json")},
                strategy, broker, RiskManager(self.risk_cfg, role_id, broker.portfolio.equity),
                inception_capital=inception.get(role_id),
            ))
        try:
            self.feed.connect()
        except FeedDisconnected as exc:
            raise HealthCheckError(f"market-data feed unavailable at startup: {exc}") from exc
        except PermissionError as exc:  # rejected or missing market-data credentials
            raise HealthCheckError(str(exc)) from exc
        offset = getattr(self.feed, "clock_offset_ms", None)
        if offset is not None and offset > self.risk_cfg.max_clock_skew_ms:
            raise HealthCheckError(f"local clock is {offset:.0f} ms behind the exchange timestamps (limit "
                                   f"{self.risk_cfg.max_clock_skew_ms:.0f} ms); sync the system clock first")

    def _make_feed(self) -> LiveFeed:
        if self.cfg.feed.kind == "alpaca":
            return self._make_alpaca_feed()
        provider = make_provider(self.cfg.feed.data_source, self.base_dir)
        if self.cfg.feed.session_date not in provider.sessions():
            raise HealthCheckError(f"feed session {self.cfg.feed.session_date} not available from the data source")
        return ReplayLiveFeed(provider.session_ref(self.cfg.feed.session_date), self.cfg.feed.speed,
                              self.cfg.feed.faults)

    def _make_alpaca_feed(self) -> LiveFeed:
        from atlas.feeds.alpaca import AlpacaAuthError, AlpacaPollingFeed, http_fetch, load_credentials

        a = self.cfg.feed.alpaca
        assert a is not None
        if self.session_date != today_market_date():
            raise HealthCheckError(f"live feed session_date {self.session_date} is not today "
                                   f"({today_market_date()}, US/Eastern)")
        if self.session_date.weekday() >= 5:
            raise HealthCheckError(f"{self.session_date} is a weekend; the market is closed")
        try:
            creds = load_credentials(a.env_file if a.env_file.is_absolute() else Path.cwd() / a.env_file)
        except AlpacaAuthError as exc:
            raise HealthCheckError(str(exc)) from exc
        symbols = [s for s in self.risk_cfg.allowed_symbols if s in self._live_symbols()]
        return AlpacaPollingFeed(symbols, self.session_date, http_fetch(creds), a.feed, a.poll_interval_ms,
                                 a.trade_poll_every, a.start_time, a.end_time, a.wait_for_open)

    def _live_symbols(self) -> list[str]:
        ds = self.cfg.feed.data_source
        return list(ds.symbols) if ds is not None and ds.symbols else list(self.risk_cfg.allowed_symbols)

    def _write_live(self, now: int | None, status: str) -> None:
        """Small, frequently refreshed snapshot for the live page (read-only consumers)."""
        if not getattr(self, "accounts", None):
            return
        accounts = []
        bench = next((a for a in self.accounts if a.role == "benchmark"), None)
        bench_ret = (bench.broker.portfolio.equity / bench.broker.portfolio.initial_cash - 1) if bench else 0.0
        trades = []
        for a in self.accounts:
            p = a.broker.portfolio
            ret = p.equity / p.initial_cash - 1
            step = max(1, len(a.equity_curve) // 600)
            inception = a.inception_capital or p.initial_cash
            accounts.append({
                "start_capital": p.initial_cash, "inception_capital": inception,
                "total_pnl": p.equity - inception, "total_return": p.equity / inception - 1,
                "account": a.account_id, "role": a.role, "agent_id": a.model["agent_id"],
                "family": a.model["family"], "equity": p.equity, "cash": p.cash, "net_pnl": p.equity - p.initial_cash,
                "net_return": ret, "vs_benchmark": ret - bench_ret, "realized_pnl": p.realized_pnl,
                "unrealized_pnl": p.unrealized_pnl, "fees": p.fees_paid, "fills": len(p.fills),
                "positions": {s: q.quantity for s, q in p.positions.items() if q.quantity},
                "working_orders": len(a.broker.working_orders), "halted": a.risk.halted,
                "risk_rejections": sum(a.risk.rejections.values()),
                "equity_curve": a.equity_curve[::step][-600:],
            })
            trades += [{"account": a.account_id, "ts": f.ts, "symbol": f.symbol, "side": str(f.side),
                        "qty": f.quantity, "price": f.price, "fee": f.fee} for f in p.fills[-40:]]
        trades.sort(key=lambda t: t["ts"], reverse=True)
        snap = {
            "session": self.dir.name, "session_date": self.session_date.isoformat(), "status": status,
            "updated_wall": datetime.now(UTC).isoformat(), "market_now": now, "feed": self.cfg.feed.kind,
            "symbols": self.symbols, "safe": list(self.safe), "kill": self.kill.reason if hasattr(self, "kill") else None,
            "counters": self.counters, "accounts": accounts, "recent_trades": trades[:60], "simulated": True,
        }
        market = getattr(self, "_market", None)
        if market is not None:
            snap["quotes"] = {s: {"bid": q.bid, "ask": q.ask, "ts": q.event_ts, "stale": market.is_stale(s)}
                              for s in self.symbols if (q := market.quote(s)) is not None}
        bots = [x for x in accounts if x["role"] != "benchmark"]
        if self.fund is not None and bots:
            snap["fund"] = {
                "total_capital": self.cfg.fund.total_capital, "bots": len(bots),
                "start_today": sum(x["start_capital"] for x in bots), "equity": sum(x["equity"] for x in bots),
                "day_pnl": sum(x["net_pnl"] for x in bots), "total_pnl": sum(x["total_pnl"] for x in bots),
                "days_settled": self.fund.days_traded(),
            }
        tmp = self.dir / "live.json.tmp"
        tmp.write_text(json.dumps(snap, default=str), encoding="utf-8")
        os.replace(tmp, self.dir / "live.json")

    def _fund_capital(self, records) -> tuple[dict[str, float], dict[str, float]]:
        """Fund mode: each bot's capital for today, and at inception. Empty dicts otherwise."""
        f = self.cfg.fund
        if f is None:
            return {}, {}
        from atlas.paper.fund import FundError, FundLedger

        path = f.ledger or (self.cfg.output_dir / "fund_ledger.json")
        self.fund = FundLedger(path, f.total_capital)
        if self.reset_fund and (archived := self.fund.archive()):
            print(f"fund ledger archived to {archived}; starting a new fund", file=sys.stderr)
        bots = {acct: rec.agent_id for acct, role, _, rec in records if role != "benchmark"}
        bench_ids = [acct for acct, role, _, _ in records if role == "benchmark"]
        try:
            start, inception, bench = self.fund.start_capitals(bots, self.session_date)
        except FundError as exc:
            raise HealthCheckError(str(exc)) from exc
        return {**start, **{b: bench for b in bench_ids}}, {**inception, **{b: bench for b in bench_ids}}

    def _flatten_at_close(self, now: int, quote_at) -> None:
        """Sell every open position at the latest bid (after costs), so the day settles in cash."""
        from atlas.environment.replay import execute_ioc

        for a in self.accounts:
            p = a.broker.portfolio
            for sym in sorted(p.positions):
                qty = p.positions[sym].quantity
                quote = quote_at(sym, now)
                if not qty or quote is None:
                    continue
                order = SimOrder(a.broker.next_order_id, OrderIntent(sym, Side.SELL, qty), now, now)
                a.broker.next_order_id += 1
                order.transition(now, OrderStatus.PROPOSED)
                order.transition(now, OrderStatus.ACCEPTED)
                record = execute_ioc(order, quote, now, a.broker.costs, 1.0, 1 << 62, p)
                a.broker.orders.append(order)
                self.audit.write({"account": a.account_id, **record, "type": "eod_flatten", "symbol": sym,
                                  "qty": qty, "simulated": True,
                                  "fill_assumption": "closing sale at the last received bid, full size"})

    def _settle_fund(self, final: bool, note: str | None = None) -> None:
        if self.fund is None or self.counters["events"] == 0:  # no market data: nothing traded, nothing to settle
            return
        results = {a.account_id: {"start": a.broker.portfolio.initial_cash, "end": a.broker.portfolio.equity}
                   for a in self.accounts if a.role != "benchmark"}
        note = note or ("settled at the close (positions sold at the bid)" if final else
                        f"provisional: session stopped by {getattr(self, 'stopped_by', None)}; open positions "
                        "valued at mid")
        self.fund.settle(self.session_date, results, final, note)

    def _check_model_symbols(self, role_id: str, record: AgentRecord) -> None:
        if record.family != "ppo" or not record.artifact:
            return
        meta = json.loads((Path(record.artifact.rpartition("#")[0]).parent / "policy.json").read_text(encoding="utf-8"))
        if list(meta.get("symbols", [])) != self.symbols:
            raise HealthCheckError(f"model mismatch for {role_id}: policy trained on {meta.get('symbols')}, "
                                   f"feed provides {self.symbols}")

    # -- state ------------------------------------------------------------------------

    def _restore(self, now_hint: int) -> None:
        state = json.loads((self.dir / "state.json").read_text(encoding="utf-8"))
        if state["version"] != STATE_VERSION or state["config_fingerprint"] != self.fingerprint:
            raise HealthCheckError("saved session state does not match the current configuration; refusing to resume")
        self.audit.truncate(state["audit_offset"])  # before anything new is written
        self.alerts.truncate(state["alerts_offset"])
        by_id = {a.account_id: a for a in self.accounts}
        now = state.get("market_now") or now_hint
        for acct_id, s in state["accounts"].items():
            a = by_id[acct_id]
            a.broker = PaperBroker.from_state(s["broker"], self.sim, self.audit.write, now)
            ensure_not_live(a.broker)
            a.risk.restore(s["risk"])
            a.equity_curve = [tuple(x) for x in s["equity_curve"]]
        self.counters.update(state["counters"])
        self.validation_counts = state["validation_counts"]
        self.safe_periods = state["safe_periods"]
        if state["kill"]:
            self.kill.trigger(state["kill"])
        self.feed.seek(state["feed_position"])
        self.restarts = state.get("restarts", 0) + 1
        self.alerts.raise_("warning", "restart", f"resumed from event {state['feed_position']}; working orders "
                           "at shutdown were cancelled", now)

    def _flush(self, market_now: int, provisional_fund: bool = True) -> None:
        state = {
            "version": STATE_VERSION, "config_fingerprint": self.fingerprint, "session": self.cfg.session_name,
            "feed_position": self.feed.position, "market_now": market_now, "kill": self.kill.reason,
            "safe": self.safe, "safe_periods": self.safe_periods, "counters": self.counters,
            "validation_counts": self.validation_counts, "restarts": getattr(self, "restarts", 0),
            "accounts": {a.account_id: {"broker": a.broker.to_state(), "risk": a.risk.to_state(),
                                        "equity_curve": a.equity_curve} for a in self.accounts},
            "audit_offset": self.audit.flush(),
            "alerts_offset": self.alerts.offset(),
        }
        if provisional_fund:  # so a day is never lost if the process dies and is not resumed
            self._settle_fund(final=False, note="provisional: in progress (open positions valued at mid)")
        if getattr(self, "recorder", None) is not None and not self.recorder.error:
            try:
                self.recorder.flush()
            except Exception as exc:  # recording must never stop trading
                self._recorder_failed(exc)
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(state, default=str), encoding="utf-8")
        os.replace(tmp, self.dir / "state.json")

    # -- safe state & kill --------------------------------------------------------------

    def _enter_safe(self, reason: str, ts: int) -> None:
        if reason in self.safe:
            return
        self.safe[reason] = ts
        for a in self.accounts:
            for o in a.broker.cancel_all(ts, f"safe state: {reason}"):
                a.strategy.on_order_update(o)
        self.alerts.raise_("critical", "safe_state", f"entering safe state: {reason}", ts)

    def _clear_safe(self, reason: str, ts: int) -> None:
        if reason not in self.safe:
            return
        since = self.safe.pop(reason)
        self.safe_periods.append({"reason": reason, "start": since, "end": ts, "duration_ms": (ts - since) / NS_PER_MS})
        self.alerts.raise_("info", "safe_state_cleared", f"resumed after: {reason}", ts)

    def _engage_kill(self, reason: str, ts: int) -> None:
        if self.kill.trigger(reason):
            for a in self.accounts:
                for o in a.broker.cancel_all(ts, f"kill switch: {reason}"):
                    a.strategy.on_order_update(o)
            self.audit.write({"type": "kill_switch", "ts": ts, "reason": reason})
            self.alerts.raise_("critical", "kill_switch", reason, ts)

    # -- main loop ------------------------------------------------------------------------

    def run(self, on_ready: Any = None) -> dict[str, Any]:
        """Run the session. ``on_ready(session)`` is called once startup checks have passed."""
        self.health_checks()
        if self.resume:
            self._restore(0)
        self.recorder = self._start_recorder()
        if on_ready is not None:
            on_ready(self)
        validator = StreamValidator(self.risk_cfg.stale_data_ms, order_scope="symbol_type")
        live_every = self.cfg.dashboard.live_snapshot_ms / 1_000
        self._write_live(None, "waiting for market data")
        stale_ns = int(self.risk_cfg.stale_data_ms * NS_PER_MS)
        skew_ns = int(self.risk_cfg.max_clock_skew_ms * NS_PER_MS)
        market = MarketView(self.symbols, stale_ns)
        self._market = market  # read-only reference for the live snapshot's quote ticker
        history: dict[str, deque[Quote]] = {s: deque(maxlen=512) for s in self.symbols}

        def quote_at(symbol: str, ts: int) -> Quote | None:
            for q in reversed(history[symbol]):
                if q.event_ts <= ts:
                    return q
            return None

        self.stopped_by: str | None = None
        awake = keep_awake(self.cfg.feed.kind != "replay")
        try:
            self._loop(validator, stale_ns, skew_ns, market, history, quote_at, live_every)
        except KeyboardInterrupt:  # Ctrl+C: stop gracefully so the day can be resumed
            self.stopped_by = "operator (Ctrl+C)"
        finally:
            if awake:
                keep_awake(False)
        return self._finish(market, quote_at)

    def _feed_now(self, market: MarketView) -> int:
        """The feed's own clock when it has one (live feeds), so outages get real durations."""
        clock = getattr(self.feed, "clock_ns", None)
        return max(market.now_ns, clock()) if clock else market.now_ns

    # -- market-data recording (isolated: any failure disables it, trading continues) --------

    def _start_recorder(self):
        rc = self.cfg.record
        if not rc.enabled:
            return None
        from atlas.paper.recorder import MarketDataRecorder

        try:
            return MarketDataRecorder(rc.root, rc.dataset, self.session_date,
                                      source="alpaca-iex" if self.cfg.feed.kind == "alpaca" else "replay",
                                      license_notes="Alpaca market data (IEX), recorded for personal research; check "
                                                    "Alpaca's data terms before sharing or redistributing")
        except Exception as exc:
            self.alerts.raise_("warning", "recording_off", f"market-data recording disabled: {exc}", 0)
            return None

    def _record(self, event: MarketEvent) -> None:
        if self.recorder.error:
            return
        try:
            self.recorder.record(event)
        except Exception as exc:
            self._recorder_failed(exc)

    def _recorder_failed(self, exc: BaseException) -> None:
        self.recorder.fail(exc)
        self.alerts.raise_("warning", "recording_off", f"market-data recording disabled ({exc}); trading continues", 0)

    def _finish_recording(self) -> None:
        self.recording = None
        r = getattr(self, "recorder", None)
        if r is None or r.error:
            self.recording = {"enabled": r is not None, "error": r.error if r else None}
            return
        try:
            entry = r.finalize(self.symbols)
            self.recording = {"enabled": True, "events": r.rows, "raw": str(r.raw_path),
                              "processed": str(r.processed_dir), "session": entry}
        except Exception as exc:
            self.recording = {"enabled": True, "error": f"finalize failed: {exc}", "raw": str(r.raw_path)}
            self.alerts.raise_("warning", "recording_off", self.recording["error"], 0)

    def _stop_requested(self) -> bool:
        stop = self.dir / STOP_FILE
        if stop.exists():
            stop.unlink()
            self.stopped_by = "operator (paper-stop)"
            return True
        return False

    def _loop(self, validator, stale_ns, skew_ns, market, history, quote_at, live_every) -> None:
        last_arrival: int | None = None
        clock_streak = 0
        first_ts: int | None = None
        processed = 0
        next_kill_check = next_live = 0.0
        while True:
            wall = time.monotonic()
            if processed % 256 == 0 or wall >= next_kill_check:
                if reason := self.kill.poll_file():
                    self._engage_kill(reason, market.now_ns)
                if self._stop_requested():
                    break
            try:
                event = self.feed.next_event()
            except FeedDisconnected as exc:
                self.counters["disconnects"] += 1
                self._enter_safe("feed disconnected", self._feed_now(market))
                if not self._reconnect(self._feed_now(market), str(exc)):
                    reason = getattr(self, "fatal_feed_error", None) or                         f"feed lost: {self.risk_cfg.reconnect_max_retries} reconnect attempts failed"
                    self._engage_kill(reason, market.now_ns)
                    break
                self._clear_safe("feed disconnected", self._feed_now(market))
                continue
            except PermissionError as exc:  # e.g. market-data keys revoked mid-session
                self._engage_kill(f"market-data credentials rejected: {exc}", market.now_ns)
                break
            if event is None:
                break
            processed += 1
            self.counters["events"] += 1
            if self.recorder is not None:
                self._record(event)
            t = event.receive_ts if event.receive_ts is not None else event.event_ts
            first_ts = t if first_ts is None else first_ts

            # Clock: arrival times must not run backwards, and no event may claim to come from the future.
            anomaly = None
            if last_arrival is not None and t < last_arrival:
                anomaly = "local receive clock moved backwards"
            elif event.event_ts - t > skew_ns:
                anomaly = "event timestamp ahead of local clock beyond max_clock_skew_ms"
            if anomaly:
                self.counters["clock_anomalies"] += 1
                clock_streak = 0
                self.audit.write({"type": "clock_anomaly", "ts": market.now_ns, "detail": anomaly,
                                  "event_ts": event.event_ts, "receive_ts": t})
                self._enter_safe("clock anomaly", market.now_ns)
                continue
            if last_arrival is not None and t - last_arrival > stale_ns:
                # The whole feed went silent. Data became stale at last_arrival + stale_ns: orders due
                # before then still execute, everything else is cancelled by the safe state from that moment.
                went_stale = last_arrival + stale_ns
                for a in self.accounts:
                    for o in a.broker.process_arrivals(went_stale, quote_at):
                        a.strategy.on_order_update(o)
                self._enter_safe("stale or missing data", went_stale)
            last_arrival = t
            clock_streak += 1
            if "clock anomaly" in self.safe and clock_streak >= self.risk_cfg.clock_recovery_events:
                self._clear_safe("clock anomaly", t)

            keep, issues = validator.check(event, processed)
            for issue in issues:
                self.validation_counts[str(issue.kind)] = self.validation_counts.get(str(issue.kind), 0) + 1
                if issue.kind is IssueKind.STALE_GAP:
                    self.counters["feed_gaps"] += 1
                self.audit.write({"type": "data_issue", "ts": t, "kind": str(issue.kind), "symbol": issue.symbol,
                                  "detail": issue.detail})
            if not keep:
                self.counters["dropped_events"] += 1
                continue

            now = max(t, market.now_ns)
            market._advance(now)
            for a in self.accounts:
                for o in a.broker.process_arrivals(now, quote_at):
                    a.strategy.on_order_update(o)
            if isinstance(event, Quote):
                if market._receive_quote(event):
                    history[event.symbol].append(event)
                    for a in self.accounts:
                        a.broker.portfolio.mark(event.symbol, event.mid)
            elif isinstance(event, Trade):
                market._receive_trade(event)

            # Global data health: every symbol needs a fresh quote (after a start-up grace period).
            stale = [s for s in self.symbols if market.is_stale(s)]
            started = now - first_ts > int(self.cfg.feed.first_quote_timeout_ms * NS_PER_MS)
            if stale and (started or any(market.quote(s) for s in stale)):
                self._enter_safe("stale or missing data", now)
            elif not stale:
                self._clear_safe("stale or missing data", now)

            for a in self.accounts:
                equity = a.broker.portfolio.equity
                breach = a.risk.on_equity(equity)
                if breach:
                    for o in a.broker.cancel_all(now, f"account halted: {breach}"):
                        a.strategy.on_order_update(o)
                    self.audit.write({"type": "risk_halt", "ts": now, "account": a.account_id, "reason": breach})
                    self.alerts.raise_("critical", "risk_halt", breach, now, a.account_id)
                if now >= a.next_curve_ts:
                    a.equity_curve.append((now, equity))
                    a.next_curve_ts = now + NS_PER_S

            blocked = self.kill.engaged or bool(self.safe)
            for a in self.accounts:
                if blocked or a.risk.halted:
                    a.strategy.features.update(market)  # keep features current; propose nothing
                    continue
                self._decide(a, market, now)

            if processed % self.cfg.state_flush_events == 0:
                self._flush(market.now_ns)
            if wall >= next_kill_check:
                next_kill_check = wall + 1.0
            if wall >= next_live:
                next_live = wall + live_every
                self._write_live(market.now_ns, "running")


    def _reconnect(self, ts: int, error: str) -> bool:
        self.audit.write({"type": "disconnect", "ts": ts, "error": error})
        real_time = self.cfg.feed.kind != "replay" or self.cfg.feed.speed > 0
        for attempt in range(1, self.risk_cfg.reconnect_max_retries + 1):
            self.counters["reconnect_attempts"] += 1
            if real_time:  # linear backoff, capped, so a short network outage is ridden out
                time.sleep(min(self.risk_cfg.reconnect_backoff_ms * attempt, MAX_BACKOFF_MS) / 1_000)
            try:
                self.feed.reconnect()
                self.audit.write({"type": "reconnected", "ts": ts, "attempt": attempt})
                return True
            except FeedDisconnected as exc:
                self.audit.write({"type": "reconnect_failed", "ts": ts, "attempt": attempt, "error": str(exc)})
            except PermissionError as exc:  # credentials rejected: retrying cannot help
                self.audit.write({"type": "reconnect_failed", "ts": ts, "attempt": attempt, "error": str(exc)})
                self.fatal_feed_error = str(exc)
                return False
        return False

    def _decide(self, a: Account, market: MarketView, now: int) -> None:
        if not a.strategy.poll(market):
            return
        p = a.broker.portfolio
        positions = {s: q.quantity for s, q in p.positions.items() if q.quantity}
        state = PortfolioState.of(p.cash, p.equity, positions)
        snapshot = a.strategy.snapshot(now)
        t0 = time.perf_counter_ns()
        try:
            targets = a.strategy.agent.act(a.strategy.agent.observe(snapshot, state, now))
            intents = a.strategy.intents_for(targets, snapshot, state)
        except Exception as exc:  # a faulty bot must not take the session (or the other bots) down
            a.risk.halted = f"agent error: {type(exc).__name__}: {exc}"
            for o in a.broker.cancel_all(now, a.risk.halted):
                a.strategy.on_order_update(o)
            self.audit.write({"type": "agent_error", "ts": now, "account": a.account_id, "error": repr(exc)})
            self.alerts.raise_("critical", "agent_error", a.risk.halted, now, a.account_id)
            return
        a.decision_wall_ns.append(time.perf_counter_ns() - t0)
        if not intents:
            return
        self.audit.write({
            "type": "decision", "ts": now, "account": a.account_id, "targets": targets,
            "snapshot": {s: {"bid": f.bid, "ask": f.ask, "quote_age_ms": f.quote_age_ms, "valid": f.valid}
                         for s, f in snapshot.items()},
            "cash": p.cash, "equity": p.equity, "positions": positions,
        })
        for intent in intents:
            q = market.quote(intent.symbol)
            price = (q.ask if intent.side is Side.BUY else q.bid) if q else 0.0
            d = a.risk.check_order(intent, now, price or 0.0, p.equity, positions, dict(p.marks))
            self.audit.write({"type": "risk_check", "ts": now, "account": a.account_id, "symbol": intent.symbol,
                              "side": str(intent.side), "proposed_qty": intent.quantity, "approved": d.approved,
                              "approved_qty": d.quantity, "rule": d.rule, "reason": d.reason})
            if not d.approved:
                self.counters["risk_rejections"] += 1
                a.strategy.release(intent.symbol)
                continue
            final = intent if d.quantity == intent.quantity else OrderIntent(
                intent.symbol, intent.side, d.quantity, intent.order_type, intent.limit_price, intent.time_in_force)
            order = a.broker.submit(final, now, q)
            if order.status is not OrderStatus.ACCEPTED:
                a.strategy.on_order_update(order)

    def _finish(self, market: MarketView, quote_at) -> dict[str, Any]:
        now = market.now_ns
        for a in self.accounts:
            for o in a.broker.process_arrivals(now, quote_at):
                a.strategy.on_order_update(o)
            for o in a.broker.cancel_all(now, "session ended before simulated arrival"):
                a.strategy.on_order_update(o)
        ended = not getattr(self, "stopped_by", None)
        if self.fund is not None and ended and self.cfg.fund.flatten_at_close:
            self._flatten_at_close(now, quote_at)
        for a in self.accounts:
            a.equity_curve.append((now, a.broker.portfolio.equity))
            a.broker.portfolio.assert_reconciled()
        self._settle_fund(final=ended, note=(f"settled early after kill switch ({self.kill.reason}); positions sold "
                                             "at the last bid") if ended and self.kill.engaged else None)
        for reason in list(self.safe):
            self._clear_safe(reason, now)
        self._flush(now, provisional_fund=False)  # the settlement above is the day's real one
        self._finish_recording()
        self._write_live(now, f"stopped by {self.stopped_by}" if getattr(self, "stopped_by", None) else "ended")
        from atlas.paper.report import write_daily_report

        report = write_daily_report(self)
        self.audit.close()
        release_lock(self.dir)
        return report
