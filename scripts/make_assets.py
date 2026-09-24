"""Regenerate the documentation images (reference chart and a demo-mode screenshot).

    python scripts/make_assets.py

The screenshot needs a desktop session (it opens the real window for ~12 seconds).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SIGNSCRIBE_HOME": tempfile.mkdtemp(prefix="signscribe_assets_")}
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    run = [sys.executable, "-m", "signscribe"]

    subprocess.run([*run, "reference", "--out", str(OUT / "alphabet_reference.png")], check=True, env=env)
    subprocess.run(
        [*run, "gui", "--demo", "--reset-settings", "--screenshot", str(OUT / "screenshot.png"), "--screenshot-delay", "11.75"],
        check=True,
        env=env,
    )
    print(f"wrote images to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
