"""User settings and on-disk locations."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def user_dir() -> Path:
    """Per-user data folder for settings, recorded samples and trained models."""
    override = os.environ.get("SIGNSCRIBE_HOME")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home())) / "SignScribe"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "SignScribe"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "signscribe"
    base.mkdir(parents=True, exist_ok=True)
    return base


def settings_path() -> Path:
    return user_dir() / "settings.json"


def samples_path() -> Path:
    return user_dir() / "samples.npz"


def model_path() -> Path:
    return user_dir() / "model.npz"


def gestures_path() -> Path:
    return user_dir() / "gestures.json"


@dataclass
class Settings:
    """Everything the user can tune. Persisted as JSON; unknown keys are ignored."""

    # --- camera / tracking ---------------------------------------------------------
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    model_complexity: int = 1  # 0 = fastest, 1 = most accurate
    max_hands: int = 2
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5
    dominant_hand: str = "Auto"  # Auto | Right | Left
    lighting: str = "auto"  # off | auto | always
    smoothing: float = 0.5  # 0 = raw / lowest latency, 1 = steadiest

    # --- recognition -------------------------------------------------------------
    recognition_mode: str = "auto"  # auto (hybrid if a model exists) | rules | model
    min_confidence: float = 0.55
    speed_gate: float = 1.4  # palm-lengths / second; faster = "in transit", never commits
    word_gestures: bool = True

    # --- text composition ---------------------------------------------------------
    hold_time: float = 0.55  # seconds a sign must be held to be typed
    repeat_delay: float = 1.2  # keep holding this long to type the letter again (0 = off)
    space_timeout: float = 1.4  # hand out of view this long => space (0 = off)
    case_mode: str = "smart"  # smart | upper | lower

    # --- display / output ---------------------------------------------------------
    show_angles: bool = False
    show_trail: bool = True
    show_labels: bool = True
    type_into_apps: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or settings_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        with contextlib.suppress(OSError):  # settings are a convenience; never crash over them
            path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
