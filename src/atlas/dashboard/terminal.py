"""Terminal leaderboard: a live, read-only view of a paper session in its own console window.

It reads the session's ``live.json`` (written every ~2 s) and redraws the board. It has
no controls that affect trading. Close the window, or press Ctrl+C, to stop watching;
the session itself keeps running.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas.dashboard import data

RESET, BOLD, DIM, GREEN, RED, YELLOW, CYAN = "\x1b[0m", "\x1b[1m", "\x1b[2m", "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[36m"


def _money(x: float, color: bool) -> str:
    s = f"{x:+,.2f}"
    if not color or abs(x) < 0.005:
        return s
    return f"{GREEN if x > 0 else RED}{s}{RESET}"


def _pad(s: str, width: int, right: bool = False) -> str:
    """Pad text that may contain ANSI color codes to a visible width."""
    visible = len(s.replace(RESET, "").replace(GREEN, "").replace(RED, ""))
    gap = " " * max(0, width - visible)
    return gap + s if right else s + gap


def _et_clock(ns: int | None) -> str:
    if not ns:
        return "--:--:--"
    from zoneinfo import ZoneInfo

    return datetime.fromtimestamp(ns / 1e9, tz=UTC).astimezone(ZoneInfo("America/New_York")).strftime("%H:%M:%S")


def render(live: dict[str, Any], color: bool = True, trades: int = 12) -> str:
    b = (lambda s: f"{BOLD}{s}{RESET}") if color else (lambda s: s)
    age = (datetime.now(UTC) - datetime.fromisoformat(live["updated_wall"])).total_seconds()
    lines = [
        b(f"ATLAS PAPER TRADING  |  {live['session']}  |  {', '.join(live['symbols'])}"),
        f"status: {b(live['status'])}   market time {_et_clock(live.get('market_now'))} ET   "
        f"updated {age:.0f}s ago   feed: {live['feed']}   (simulated orders only)",
    ]
    if live.get("kill"):
        lines.append((RED if color else "") + f"KILL SWITCH ENGAGED: {live['kill']}" + (RESET if color else ""))
    elif live.get("safe"):
        lines.append((YELLOW if color else "") + f"SAFE STATE (no new orders): {', '.join(live['safe'])}"
                     + (RESET if color else ""))
    elif live["status"] == "running" and age > 15:
        lines.append((YELLOW if color else "") + f"no update for {age:.0f}s: waiting for data or the session stopped"
                     + (RESET if color else ""))

    lines += ["", b("LEADERBOARD")]
    header = f"{'#':>2}  {'bot':<10} {'model':<30} {'P&L $':>12} {'return':>10} {'vs B&H':>10} {'fills':>6}  status"
    lines += [header, "-" * len(header)]
    for r in data.leaderboard(live):
        rank = str(r["rank"]) if r["rank"] else "-"
        fam = r["family"]
        model = (r["agent_id"] if fam.startswith("baseline") else f"{r['agent_id']} ({fam})")[:30]
        vs = f"{r['vs_benchmark'] * 1e4:+.2f}bp" if r["rank"] else "--"
        pos = ", ".join(f"{s} {q}" for s, q in r["positions"].items()) or "flat"
        status = f"HALTED: {r['halted']}" if r["halted"] else pos
        if r["halted"] and color:
            status = f"{RED}{status}{RESET}"
        lines.append(f"{rank:>2}  {r['account']:<10} {model:<30} {_pad(_money(r['net_pnl'], color), 12, True)} "
                     f"{r['net_return'] * 1e4:>+8.2f}bp {vs:>10} {r['fills']:>6}  {status}")

    lines += ["", b("RECENT SIMULATED TRADES")]
    recent = live.get("recent_trades", [])[:trades]
    if not recent:
        lines.append(DIM + "  none yet" + RESET if color else "  none yet")
    for t in recent:
        raw = t["side"].upper()
        side = f"{raw:<4}"
        if color:
            side = f"{GREEN if raw == 'BUY' else RED}{side}{RESET}"
        lines.append(f"  {_et_clock(t['ts'])}  {t['account']:<10} {side} {t['qty']:>5} {t['symbol']:<5} "
                     f"@ {t['price']:>10.4f}   fee {t['fee']:.4f}")
    c = live["counters"]
    lines += ["", (DIM if color else "") + f"events {c['events']}  dropped {c['dropped_events']}  disconnects "
              f"{c['disconnects']}  clock anomalies {c['clock_anomalies']}  risk rejections {c['risk_rejections']}"
              f"   |   Ctrl+C closes this window; the session keeps running" + (RESET if color else "")]
    return "\n".join(lines)


def watch(session_dir: Path, interval_s: float = 2.0, once: bool = False) -> int:
    if os.name == "nt":
        os.system("")  # enables ANSI escape sequences in the Windows console
    color = sys.stdout.isatty()
    try:
        while True:
            live = data.load_live(session_dir)
            out = render(live, color) if live else f"Waiting for session {session_dir.name} to start..."
            if color:
                sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
            if once:
                return 0
            if live and (live["status"] == "ended" or live["status"].startswith("stopped")):
                sys.stdout.write("\nSession finished. Final standings above.\n")
                return 0
            time.sleep(interval_s)
    except KeyboardInterrupt:
        return 0


def open_window(session_dir: Path, title: str = "Atlas leaderboard") -> None:
    """Open the terminal leaderboard in its own console window (Windows) or terminal."""
    import subprocess

    if os.name == "nt":
        # A raw command line, so cmd.exe sees the title unquoted; /k keeps the final standings on screen.
        line = (f'cmd /k title {title} && "{sys.executable}" -m atlas.cli paper-watch '
                f'--session-dir "{session_dir}"')
        subprocess.Popen(line, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # best effort elsewhere: run detached in the background of the current terminal
        subprocess.Popen([sys.executable, "-m", "atlas.cli", "paper-watch", "--session-dir", str(session_dir)])
