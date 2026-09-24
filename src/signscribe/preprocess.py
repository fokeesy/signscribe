"""Adaptive exposure so hand detection survives dim rooms, glare and dark or pale skin.

MediaPipe's hand models are convolutional networks trained on very diverse imagery: they do not
segment by skin *colour*, which is why the system is not biased toward any skin tone. What they
*do* suffer from is poor exposure: a dark hand in a dim room, or a bright window behind the
signer that makes the camera underexpose the hand.

:class:`LightingAdapter` fixes exposure before detection by

1. **metering on the hand** - once a hand has been seen it meters the previous hand box rather
   than the whole frame, so a dark hand against a bright background (or the reverse) is
   corrected for the *hand*, not for the background;
2. **auto-gamma** - a look-up-table brightening / darkening toward mid-grey, with hysteresis so
   it never flickers on and off;
3. **CLAHE** - local contrast equalisation on the lightness channel when the scene is flat or
   the correction is active, which restores finger edges.

The corrected image is only fed to the tracker; the preview window shows the camera as-is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LightingInfo:
    mean: float = 0.0
    std: float = 0.0
    gamma: float = 1.0
    clahe: bool = False
    state: str = "ok"  # dark | bright | flat | ok

    def describe(self) -> str:
        return {"dark": "Low light (boosting)", "bright": "Very bright (toning down)", "flat": "Flat contrast (enhancing)"}.get(
            self.state, "Lighting OK"
        )


class LightingAdapter:
    TARGET = 118.0  # mid-grey the metering area is pulled toward

    def __init__(self, mode: str = "auto") -> None:
        self.mode = mode  # off | auto | always
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._mean: float | None = None
        self._active = ""  # "" | "dark" | "bright"
        self._lut: np.ndarray | None = None
        self._lut_gamma = 1.0
        self.info = LightingInfo()

    def _meter(self, bgr: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[float, float]:
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            x0, y0, x1, y1 = roi
            sx, sy = 160 / w, 120 / h
            a, b = max(int(x0 * sx), 0), max(int(y0 * sy), 0)
            c, d = min(int(math.ceil(x1 * sx)), 160), min(int(math.ceil(y1 * sy)), 120)
            if (c - a) * (d - b) >= 60:
                patch = gray[b:d, a:c]
                # Blend so a tiny box cannot dominate exposure decisions.
                return 0.7 * float(patch.mean()) + 0.3 * float(gray.mean()), float(gray.std())
        return float(gray.mean()), float(gray.std())

    def process(self, bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Return an exposure-corrected copy of ``bgr`` (or the same array when nothing to do)."""
        if self.mode == "off":
            self.info = LightingInfo(state="ok")
            return bgr
        mean, std = self._meter(bgr, roi)
        self._mean = mean if self._mean is None else 0.8 * self._mean + 0.2 * mean
        m = float(np.clip(self._mean, 8.0, 247.0))

        # Hysteresis: engage at the outer limits, release well inside them.
        if self._active == "":
            if m < 78:
                self._active = "dark"
            elif m > 172:
                self._active = "bright"
        elif (self._active == "dark" and m > 100) or (self._active == "bright" and m < 150):
            self._active = ""

        gamma = 1.0
        if self._active or self.mode == "always":
            gamma = float(np.clip(math.log(self.TARGET / 255.0) / math.log(m / 255.0), 0.45, 2.2))
        flat = std < 36.0
        use_clahe = self.mode == "always" or bool(self._active) or flat
        state = self._active or ("flat" if flat else "ok")
        self.info = LightingInfo(mean=m, std=std, gamma=gamma, clahe=use_clahe, state=state)
        if abs(gamma - 1.0) < 0.05 and not use_clahe:
            return bgr

        out = bgr
        if abs(gamma - 1.0) >= 0.05:
            if self._lut is None or abs(gamma - self._lut_gamma) > 0.04:
                x = np.arange(256, dtype=np.float32) / 255.0
                self._lut = np.clip(np.power(x, gamma) * 255.0, 0, 255).astype(np.uint8)
                self._lut_gamma = gamma
            out = cv2.LUT(out, self._lut)
        if use_clahe:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return out

    def force_enhance(self, bgr: np.ndarray) -> np.ndarray:
        """Unconditional gamma-lite + CLAHE, used for the periodic 'rescue' detection attempt."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
