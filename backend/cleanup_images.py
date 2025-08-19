import os
import time
import threading
from pathlib import Path

IMAGE_DIR = Path(".")          # same folder where you save PNGs
RETENTION_SECONDS = 60 * 60    # 1 hour

def _cleanup_loop():
    """Runs forever in a daemon thread."""
    while True:
        now = time.time()
        for p in IMAGE_DIR.glob("*.png"):
            if now - p.stat().st_mtime > RETENTION_SECONDS:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        time.sleep(300)  # check every 5 min

def start_cleanup_daemon():
    t = threading.Thread(target=_cleanup_loop, daemon=True)
    t.start()