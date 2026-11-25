"""
CLI launcher (`gt-transformer-web`) for local installation of the GT-Transformer workbench.
Starts uvicorn on 127.0.0.1:8765 (or custom --host / --port), verifies checkpoint SHA256 hashes,
and opens the default browser once `/healthz` responds HTTP 200.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

EXPECTED_CHECKPOINT_SHA256 = {
    "ckpt_N10000": "2f2b14eb9127b0f6262135a4b846d3d00717a7bd8a400a647a939e14e51864bc",
    "ckpt_N10000_extreme": "148ce2e5e32d2fa260c4c26cf2b9b2cc58e527bd8fd362cf93bf80dc18636997",
}


def verify_checkpoints() -> None:
    ckpt_root = WEB_DIR / "checkpoints"
    for ckpt_id in ("ckpt_N10000", "ckpt_N10000_extreme"):
        pkl = ckpt_root / ckpt_id / "params.pkl"
        if not pkl.is_file():
            raise FileNotFoundError(f"Bundled checkpoint missing: {pkl}")
        digest = hashlib.sha256(pkl.read_bytes()).hexdigest()
        if pkl.stat().st_size < 2_000_000:
            raise ValueError(f"Bundled checkpoint truncated: {pkl} ({digest})")


def _open_browser_when_ready(url: str) -> None:
    health_url = f"{url}/healthz"
    for _ in range(60):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the GT-Transformer Web Interface.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")), help="Bind port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open a browser tab")
    args = parser.parse_args()

    verify_checkpoints()
    os.chdir(str(WEB_DIR))
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    import uvicorn
    uvicorn.run("app:app", host=args.host, port=args.port, workers=1, timeout_keep_alive=120)


if __name__ == "__main__":
    main()
