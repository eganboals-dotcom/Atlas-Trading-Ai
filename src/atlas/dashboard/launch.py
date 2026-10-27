"""Start the read-only dashboard in the background and open it in the browser."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def launch(paper_dir: Path, session: str | None, port: int, open_browser: bool = True) -> str:
    url = f"http://localhost:{port}"
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(port),
           "--server.headless", "true", "--server.address", "localhost", "--browser.gatherUsageStats", "false",
           "--", "--paper-dir", str(paper_dir)] + (["--session", session] if session else [])
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)  # keep running after the session ends
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(4), webbrowser.open(url)), daemon=True).start()
    return url
