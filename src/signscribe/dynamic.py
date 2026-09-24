"""Motion-based recognition: air-traced letters (J, Z) and short word gestures.

Static handshapes are handled by :mod:`signscribe.rules` / :mod:`signscribe.model`; this
module watches how the hand *moves*.

* **J** - pinky-only handshape; the pinky tip draws a downward stroke that hooks upward.
* **Z** - index-only handshape; the index tip draws a Z (across, diagonal, across).
* **Hello** - open hand waved side to side.
* **Yes** - closed fist nodded up and down.

All geometry is measured in *palm lengths* so it does not depend on how far you sit from the
camera, and both hooks/strokes are accepted in either direction so left-handed signers work
without any setting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .constants import INDEX_TIP, PINKY_TIP
from .hand import HandObservation


@dataclass
class MotionEvent:
    label: str  # e.g. "J", "Z", "Hello", "Yes"
    kind: str  # "letter" or "word"
    confidence: float
    t: float


# ----------------------------------------------------------------------- geometry helpers
def resample(points: np.ndarray, n: int = 64) -> np.ndarray:
    """Resample a polyline to ``n`` points evenly spaced along its length."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n)
    return np.stack([np.interp(targets, cum, pts[:, 0]), np.interp(targets, cum, pts[:, 1])], axis=1)


def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-9:
        d = np.linalg.norm(points - a, axis=1)
    else:
        d = np.abs(ab[0] * (points[:, 1] - a[1]) - ab[1] * (points[:, 0] - a[0])) / denom
    i = int(np.argmax(d))
    if d[i] > eps:
        left = rdp(points[: i + 1], eps)
        right = rdp(points[i:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


def _smooth(points: np.ndarray, k: int = 2) -> np.ndarray:
    if len(points) < 2 * k + 1:
        return points
    padded = np.pad(points, ((k, k), (0, 0)), mode="edge")
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(2)], axis=1)


def match_j(stroke: np.ndarray) -> float:
    """Confidence 0..1 that ``stroke`` (palm units, y down) is a J. 0 = not a J."""
    if len(stroke) < 6:
        return 0.0
    pts = resample(_smooth(stroke), 48)
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if length < 1.2:
        return 0.0
    k = int(np.argmax(pts[:, 1]))  # lowest point of the stroke
    n = len(pts) - 1
    drop = pts[k, 1] - pts[0, 1]
    if drop < 0.8 or not (0.35 * n <= k <= 0.97 * n):
        return 0.0
    stem = pts[: k + 1]
    if np.mean(np.diff(stem[:, 1]) > -0.005) < 0.75:  # stem must head downward
        return 0.0
    if abs(pts[k, 0] - pts[0, 0]) > 0.7 * drop:
        return 0.0
    end = pts[-1]
    rise = pts[k, 1] - end[1]
    lateral = abs(end[0] - pts[k, 0])
    if rise < 0.18 or lateral < 0.22:
        return 0.0
    return float(np.clip(0.62 + 0.25 * min(drop / 1.6, 1.0) + 0.2 * min(min(rise, lateral) / 0.5, 1.0), 0.0, 0.98))


def match_z(stroke: np.ndarray) -> float:
    """Confidence 0..1 that ``stroke`` is a Z (either handedness). 0 = not a Z."""
    if len(stroke) < 6:
        return 0.0
    pts = resample(_smooth(stroke), 64)
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if length < 1.6:
        return 0.0
    for eps in (0.12, 0.18, 0.26, 0.36):
        poly = rdp(pts, eps)
        if len(poly) != 4:
            continue
        s1, s2, s3 = poly[1] - poly[0], poly[2] - poly[1], poly[3] - poly[2]
        long_enough = min(np.linalg.norm(s1), np.linalg.norm(s2), np.linalg.norm(s3)) >= 0.35
        flat1 = abs(s1[0]) >= 0.4 and abs(s1[0]) >= 1.3 * abs(s1[1])
        flat3 = abs(s3[0]) >= 0.4 and abs(s3[0]) >= 1.3 * abs(s3[1])
        diag = abs(s2[0]) >= 0.3 and s2[1] >= 0.3 and s2[1] >= 0.35 * abs(s2[0])
        reverses = np.sign(s2[0]) == -np.sign(s1[0]) and np.sign(s3[0]) == np.sign(s1[0])
        if long_enough and flat1 and flat3 and diag and reverses:
            return 0.85
    return 0.0


def best_suffix(match, stroke: np.ndarray) -> float:
    """Best ``match`` score over the stroke's tails.

    A real letter is often drawn straight out of a preceding movement (the hand sweeping into
    view, or the previous letter), so the letter is judged on whichever *tail* of the motion
    fits best rather than on the whole thing.
    """
    n = len(stroke)
    step = max(n // 14, 1)
    return max((match(stroke[s:]) for s in range(0, max(n - 8, 1), step)), default=0.0)


def count_swings(values: np.ndarray, amp: float) -> int:
    """Number of direction reversals of at least ``amp`` in a 1-D signal (zig-zag counter)."""
    if len(values) < 3:
        return 0
    ext = values[0]
    direction = 0
    swings = 0
    for v in values[1:]:
        if direction == 0:
            if v - ext >= amp:
                direction, ext = 1, v
            elif ext - v >= amp:
                direction, ext = -1, v
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext - v >= amp:
                direction, ext, swings = -1, v, swings + 1
        else:
            if v < ext:
                ext = v
            elif v - ext >= amp:
                direction, ext, swings = 1, v, swings + 1
    return swings


# --------------------------------------------------------------------------- dynamics
class HandDynamics:
    """Palm-centre velocity in palm lengths per second, with light smoothing."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last: tuple[float, np.ndarray] | None = None
        self.palm: float | None = None
        self.velocity = np.zeros(2)
        self.speed = 0.0

    def update(self, obs: HandObservation) -> float:
        palm = obs.palm_px
        self.palm = palm if self.palm is None else 0.9 * self.palm + 0.1 * palm
        centre = obs.center_px
        if self._last is not None and obs.t > self._last[0]:
            dt = obs.t - self._last[0]
            v = (centre - self._last[1]) / dt / self.palm
            self.velocity = 0.5 * self.velocity + 0.5 * v
            self.speed = float(np.linalg.norm(self.velocity))
        self._last = (obs.t, centre)
        return self.speed


class _StrokeTracker:
    """Segments one fingertip's motion into strokes.

    A stroke starts when the tip speeds up and ends when it pauses, when the handshape stops
    matching, or when the hand leaves the frame, whichever comes first.
    """

    V_START = 1.1  # palm lengths / s
    V_END = 0.5
    END_HOLD = 0.15  # seconds of near-stillness that closes a stroke
    MIN_DUR = 0.3
    MAX_DUR = 2.6
    INVALID_RUN = 4  # consecutive wrong-handshape frames that close a stroke

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.pts: list[tuple[float, float, float]] = []
        self.valid = 0
        self.frames = 0
        self.invalid_run = 0
        self.last: tuple[float, np.ndarray] | None = None
        self.speed = 0.0
        self.slow_since: float | None = None

    def finish(self) -> np.ndarray | None:
        """Close the current stroke now and return its (N, 2) points if it is usable."""
        if not self.active:
            return None
        arr = np.array(self.pts)
        if self.slow_since is not None:
            arr = arr[arr[:, 0] <= self.slow_since]
        ok = len(arr) >= 6 and arr[-1, 0] - arr[0, 0] >= self.MIN_DUR and self.valid / max(self.frames, 1) >= 0.7
        self.active, self.pts, self.slow_since = False, [], None
        self.valid = self.frames = self.invalid_run = 0
        return arr[:, 1:3] if ok else None

    def update(self, t: float, pos: np.ndarray, valid: bool) -> np.ndarray | None:
        if self.last is not None and t > self.last[0]:
            v = float(np.linalg.norm(pos - self.last[1])) / (t - self.last[0])
            self.speed = 0.5 * self.speed + 0.5 * v
        prev = self.last
        self.last = (t, pos)
        if not self.active:
            if valid and self.speed >= self.V_START and prev is not None:
                self.active = True
                self.pts = [(prev[0], prev[1][0], prev[1][1]), (t, pos[0], pos[1])]
                self.valid, self.frames, self.invalid_run, self.slow_since = 1, 1, 0, None
            return None
        self.pts.append((t, pos[0], pos[1]))
        self.frames += 1
        self.valid += int(valid)
        self.invalid_run = 0 if valid else self.invalid_run + 1
        if t - self.pts[0][0] > self.MAX_DUR:
            self.finish()  # too long to be a single letter; discard
            return None
        if self.invalid_run >= self.INVALID_RUN:
            return self.finish()
        if self.speed < self.V_END:
            self.slow_since = self.slow_since if self.slow_since is not None else t
            if t - self.slow_since >= self.END_HOLD:
                return self.finish()
        else:
            self.slow_since = None
        return None


@dataclass
class _Sample:
    t: float
    center: np.ndarray
    open_hand: bool
    fist: bool


class MotionRecognizer:
    """Watches fingertip and palm trajectories and emits :class:`MotionEvent` s."""

    WINDOW = 1.7  # seconds of history for wave / nod detection
    COOLDOWN = 2.0  # after a word gesture
    LETTER_COOLDOWN = 0.5  # after a traced letter

    def __init__(self, enable_words: bool = True) -> None:
        self.enable_words = enable_words
        self._j = _StrokeTracker()
        self._z = _StrokeTracker()
        self._hist: deque[_Sample] = deque()
        self._palm: float | None = None
        self._cooldown_until = 0.0
        self._appeared: float | None = None

    SETTLE = 0.3  # seconds after the hand appears before motion is trusted

    def reset(self) -> None:
        self._j.reset()
        self._z.reset()
        self._hist.clear()

    def lost(self, t: float = 0.0) -> MotionEvent | None:
        """Call when the hand disappears.

        A letter traced right before the hand drops out of view is still judged, so signers
        do not have to hold still for the recogniser's benefit.
        """
        ev = None
        if t >= self._cooldown_until and self._appeared is not None:
            for tracker, match, label in ((self._j, match_j, "J"), (self._z, match_z, "Z")):
                stroke = tracker.finish()
                if stroke is not None and ev is None:
                    conf = best_suffix(match, stroke)
                    if conf > 0:
                        ev = MotionEvent(label, "letter", conf, t)
        self.reset()
        self._appeared = None
        return self._fire(ev) if ev else None

    def update(self, obs: HandObservation, feats: dict[str, float]) -> MotionEvent | None:
        t = obs.t
        palm = obs.palm_px
        self._palm = palm if self._palm is None else 0.9 * self._palm + 0.1 * palm
        if self._appeared is None:
            self._appeared = t
        if t - self._appeared < self.SETTLE:  # tracking is still locking on; ignore entry motion
            self._j.reset()
            self._z.reset()
            return None
        px = obs.pixels / self._palm
        ext = (feats["ext_i"], feats["ext_m"], feats["ext_r"], feats["ext_p"])

        only_pinky = ext[3] > 0.6 and max(ext[0], ext[1], ext[2]) < 0.5
        only_index = ext[0] > 0.6 and max(ext[1], ext[2], ext[3]) < 0.5
        j_stroke = self._j.update(t, px[PINKY_TIP], only_pinky)
        z_stroke = self._z.update(t, px[INDEX_TIP], only_index)

        if t >= self._cooldown_until:
            if j_stroke is not None:
                conf = best_suffix(match_j, j_stroke)
                if conf > 0:
                    return self._fire(MotionEvent("J", "letter", conf, t))
            if z_stroke is not None:
                conf = best_suffix(match_z, z_stroke)
                if conf > 0:
                    return self._fire(MotionEvent("Z", "letter", conf, t))

        self._hist.append(_Sample(t, obs.center_px / self._palm, min(ext) >= 0.6, max(ext) <= 0.35))
        while self._hist and t - self._hist[0].t > self.WINDOW:
            self._hist.popleft()

        if self.enable_words and t >= self._cooldown_until and len(self._hist) >= 10:
            ev = self._check_oscillation(t)
            if ev:
                return self._fire(ev)
        return None

    def _fire(self, ev: MotionEvent) -> MotionEvent:
        # Letters may follow each other quickly (J then Z); a wave or nod must not re-fire.
        self._cooldown_until = ev.t + (self.LETTER_COOLDOWN if ev.kind == "letter" else self.COOLDOWN)
        self.reset()
        return ev

    def _check_oscillation(self, t: float) -> MotionEvent | None:
        centres = np.array([s.center for s in self._hist])
        span = self._hist[-1].t - self._hist[0].t
        if span < 0.8:
            return None
        open_frac = np.mean([s.open_hand for s in self._hist])
        fist_frac = np.mean([s.fist for s in self._hist])
        x_rng = np.ptp(centres[:, 0])
        y_rng = np.ptp(centres[:, 1])
        waving = open_frac >= 0.7 and x_rng >= 0.7 and x_rng >= 1.5 * y_rng
        if waving and count_swings(centres[:, 0], 0.32) >= 3:
            return MotionEvent("Hello", "word", 0.9, t)
        nodding = fist_frac >= 0.7 and y_rng >= 0.55 and y_rng >= 1.5 * x_rng
        if nodding and count_swings(centres[:, 1], 0.28) >= 3:
            return MotionEvent("Yes", "word", 0.88, t)
        return None
