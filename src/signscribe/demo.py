"""A scripted virtual signer: fingerspells a text with the procedural hand, no camera needed.

Used for ``signscribe --demo``, for the "Demo" button in the GUI, and as the deterministic
input of the end-to-end tests (feed it virtual timestamps and check the recognised text).
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import FramePacket
from .hand import HandObservation
from .synthetic import POSES, ResolvedPose, make_observation, resolved

FRAME = (640, 480)
PX_PER_CM = 12.0
PALM_CM = 8.9


@dataclass
class _Segment:
    kind: str  # "letter" | "gap" | "motion" | "wave"
    pose: str
    start: float
    enter: float
    hold: float
    path: str = ""

    @property
    def end(self) -> float:
        return self.start + self.enter + self.hold


def _j_path(u: float) -> tuple[float, float]:
    """Stem down then hook, in palm lengths; u in 0..1."""
    if u < 0.7:
        return 0.0, 1.7 * u / 0.7
    th = math.radians(160.0 * (u - 0.7) / 0.3)
    return -0.5 + 0.5 * math.cos(th), 1.7 + 0.5 * math.sin(th)


def _z_path(u: float) -> tuple[float, float]:
    """Across, diagonal back, across."""
    pts = [(0.0, 0.0), (1.3, 0.0), (0.0, 1.3), (1.3, 1.3)]
    seg = min(int(u * 3), 2)
    f = u * 3 - seg
    (x0, y0), (x1, y1) = pts[seg], pts[seg + 1]
    return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class VirtualSigner:
    """Deterministic timeline of a hand fingerspelling ``text``.

    Letters are held long enough to be typed at the default settings, spaces become the hand
    leaving the frame, and J/Z are traced in the air. ``~`` inserts an open-hand wave ("Hello").
    """

    def __init__(self, text: str, seed: int = 0, noise: float = 0.006, hold: float = 1.15, mirror: bool = False) -> None:
        self.text = text
        self.noise = noise
        self.mirror = mirror  # True = a left hand
        self.rng = np.random.default_rng(seed)
        self.segments: list[_Segment] = []
        t = 0.6  # initial empty frame
        prev = None
        for ch in text.upper():
            if ch == " ":
                self.segments.append(_Segment("gap", "", t, 0.0, 2.0))
                t += 2.0
                prev = None
                continue
            if ch == "~":
                self.segments.append(_Segment("wave", "OPEN", t, 0.4, 2.2))
                t += 2.6
                prev = "~"
                continue
            if ch not in POSES:
                continue
            if prev == ch:  # double letter: hand briefly drops between the two
                self.segments.append(_Segment("gap", "", t, 0.0, 0.35))
                t += 0.35
            motion = ch in ("J", "Z")
            kind = "motion" if motion else "letter"
            dur = 1.3 if motion else hold
            self.segments.append(_Segment(kind, ch, t, 0.4, dur, path=ch if motion else ""))
            t += 0.4 + dur
            prev = ch
        self.segments.append(_Segment("gap", "", t, 0.0, 2.2))
        self.duration = t + 2.2

    # ---------------------------------------------------------------------------
    def _segment_at(self, tau: float) -> tuple[_Segment | None, int]:
        for i, seg in enumerate(self.segments):
            if seg.start <= tau < seg.end:
                return seg, i
        return None, -1

    def observation(self, tau: float, t_abs: float | None = None) -> HandObservation | None:
        """The hand at script time ``tau`` (seconds), or None when it is out of view."""
        seg, i = self._segment_at(tau)
        if seg is None or seg.kind == "gap":
            return None
        local = tau - seg.start
        centre = [0.5, 0.42]
        pose: ResolvedPose = resolved(seg.pose)
        # Where the previous segment left the hand, for smooth transitions.
        prev = self.segments[i - 1] if i > 0 else None
        if local < seg.enter:
            u = _smooth(local / seg.enter)
            if prev is not None and prev.kind in ("letter", "motion", "wave"):
                pose = resolved(prev.pose).lerp(pose, u)
            else:  # rise into view from below
                centre[1] = 0.42 + 0.55 * (1 - u)
        else:
            u = (local - seg.enter) / max(seg.hold, 1e-6)
            if seg.kind == "motion":
                run = min(u / 0.7, 1.0)  # trace during the first 70% of the hold
                fn = _j_path if seg.path == "J" else _z_path
                dx, dy = fn(_smooth(run) if seg.path == "Z" else run)
                if seg.path == "Z":
                    dx -= 0.65
                    dy -= 0.3
                else:
                    dx += 0.3
                    dy -= 0.6
                centre[0] += dx * PALM_CM * PX_PER_CM / FRAME[0]
                centre[1] += dy * PALM_CM * PX_PER_CM / FRAME[1]
            elif seg.kind == "wave":
                centre[0] += 0.09 * math.sin((local - seg.enter) * 2 * math.pi * 2.2)
        # Leaving: nothing to do, the next gap segment simply removes the hand.
        # Gentle life-like drift so nothing is perfectly static.
        centre[0] += 0.004 * math.sin(tau * 1.7)
        centre[1] += 0.004 * math.cos(tau * 1.3)
        return make_observation(
            pose,
            center=(centre[0], centre[1]),
            px_per_cm=PX_PER_CM,
            frame_size=FRAME,
            noise=self.noise,
            rng=self.rng,
            mirror=self.mirror,
            t=tau if t_abs is None else t_abs,
            yaw=pose.yaw + 8 * math.sin(tau * 0.9),
            pitch=pose.pitch - 6 * math.cos(tau * 0.7),
        )

    def frames(self, fps: float = 30.0) -> Iterator[tuple[float, HandObservation | None]]:
        """Virtual-time iterator of (timestamp, observation) for offline tests."""
        n = int(self.duration * fps)
        for k in range(n):
            tau = k / fps
            yield tau, self.observation(tau)


def make_background(size: tuple[int, int] = FRAME) -> np.ndarray:
    """A calm dark backdrop with a faint grid, used behind simulated hands."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    base = (46 - 26 * np.clip(d, 0, 1.2)).clip(12, 60)
    img = np.stack([base * 1.10, base * 1.02, base * 0.95], axis=-1).astype(np.uint8)
    grid = np.zeros_like(img)
    grid[::40, :] = 6
    grid[:, ::40] = 6
    return cv2.add(img, grid)


class SimulatedSource:
    """Real-time frame source that plays a :class:`VirtualSigner` (no camera needed)."""

    def __init__(self, text: str = "hello world", loop: bool = True, speed: float = 1.0, fps: float = 30.0) -> None:
        self.signer = VirtualSigner(text)
        self.loop, self.speed, self.fps = loop, speed, fps
        self.name = "Demo (simulated hand)"
        self.error: str | None = None
        self._t0 = 0.0
        self._next = 0.0
        self._bg = make_background()

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._next = self._t0

    @property
    def alive(self) -> bool:
        return self.error is None

    def read(self, timeout: float = 0.5) -> FramePacket | None:
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(min(delay, timeout))
        now = time.perf_counter()
        self._next = max(self._next + 1.0 / self.fps, now - 1.0 / self.fps)
        tau = (now - self._t0) * self.speed
        if tau > self.signer.duration:
            if not self.loop:
                self.error = "Demo finished."
                return None
            self._t0 = now
            tau = 0.0
        obs = self.signer.observation(tau, t_abs=now)
        return FramePacket(self._bg.copy(), now, hands=[obs] if obs is not None else [], info={"demo": True})

    def stop(self) -> None:
        pass
