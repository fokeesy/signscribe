"""Procedural hand model used for tests, the no-camera demo and reference renderings.

A small forward-kinematics hand (palm + 4 three-joint fingers + a thumb solved with IK)
generates the same 21 landmarks MediaPipe produces. Every ASL letter has a pose defined
here, which gives the project:

* a way to exercise the whole recognition pipeline without a webcam (unit tests, CI, demo)
* reference skeletons for the on-screen "how to sign this letter" hints

The poses are *approximations* of the ASL manual alphabet built from anatomical
proportions; they are not measurements of real signers.

Local frame (right hand, palm facing the viewer): x toward the thumb side, y along the
fingers, z out of the palm. Lengths are centimetres.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .constants import ALPHABET, FINGER_CHAINS, LONG_FINGERS
from .hand import HandObservation

# --------------------------------------------------------------------- hand anatomy
REST_MCP = {
    "index": np.array([2.4, 8.7, 0.0]),
    "middle": np.array([0.6, 9.2, 0.0]),
    "ring": np.array([-1.2, 8.8, 0.0]),
    "pinky": np.array([-3.3, 7.9, 0.0]),
}
BONE_LEN = {
    "index": (4.2, 2.4, 2.0),
    "middle": (4.6, 2.8, 2.1),
    "ring": (4.3, 2.6, 2.0),
    "pinky": (3.4, 1.9, 1.8),
}
THUMB_BASE = np.array([2.2, 1.6, 0.4])
THUMB_LEN = (4.0, 3.2, 2.6)

# Finger curl presets: (MCP, PIP, DIP) flexion in degrees.
EXT = (0.0, 0.0, 0.0)
FIST = (88.0, 100.0, 55.0)
CURVE = (35.0, 40.0, 30.0)
ROUND = (55.0, 58.0, 38.0)
HOOK = (15.0, 95.0, 15.0)
CLAW = (75.0, 95.0, 45.0)


def F(curl: tuple[float, float, float], spread: float = 0.0) -> tuple[float, float, float, float]:
    """A finger: preset curl plus in-plane spread in degrees (+ve leans toward the thumb)."""
    return (*curl, spread)


@dataclass(frozen=True)
class PoseSpec:
    """Human-authored pose: finger angles plus a symbolic thumb target."""

    fingers: tuple  # 4 x (mcp, pip, dip, spread) for index, middle, ring, pinky
    thumb: tuple  # ("abs", x, y, z) | ("tip"/"pip"/"dip", finger, dx, dy, dz) | ("dir", az, el, extension)
    roll: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


@dataclass
class ResolvedPose:
    """Numeric pose that can be interpolated: finger angles + thumb tip target."""

    fingers: np.ndarray  # (4, 4)
    thumb_target: np.ndarray  # (3,)
    roll: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    def lerp(self, other: ResolvedPose, u: float) -> ResolvedPose:
        u = float(np.clip(u, 0.0, 1.0))
        return ResolvedPose(
            fingers=(1 - u) * self.fingers + u * other.fingers,
            thumb_target=(1 - u) * self.thumb_target + u * other.thumb_target,
            roll=(1 - u) * self.roll + u * other.roll,
            yaw=(1 - u) * self.yaw + u * other.yaw,
            pitch=(1 - u) * self.pitch + u * other.pitch,
        )


# --------------------------------------------------------------------------- kinematics
def _finger_points(name: str, mcp: float, pip: float, dip: float, spread: float) -> np.ndarray:
    """(4, 3) joint positions [MCP, PIP, DIP, TIP] of one finger in the local frame."""
    th = np.radians(spread)
    phis = np.radians(np.cumsum([mcp, pip, dip]))
    pts = [REST_MCP[name].copy()]
    for length, phi in zip(BONE_LEN[name], phis):
        d = np.array([np.sin(th) * np.cos(phi), np.cos(th) * np.cos(phi), np.sin(phi)])
        pts.append(pts[-1] + length * d)
    return np.array(pts)


def _thumb_ik(target: np.ndarray, iterations: int = 24) -> np.ndarray:
    """FABRIK solve for (CMC, MCP, IP, TIP) reaching ``target``; bends toward +z."""
    base = THUMB_BASE
    lengths = np.array(THUMB_LEN)
    total = lengths.sum()
    to_t = target - base
    dist = np.linalg.norm(to_t)
    direction = to_t / (dist + 1e-9)
    if dist >= total * 0.995:
        cum = np.concatenate([[0.0], np.cumsum(lengths)])
        return np.array([base + direction * c for c in cum])

    # Initial guess: a shallow arc bulging toward the palm side (+z).
    bulge = np.array([0.0, 0.0, 1.0])
    bulge = bulge - direction * float(np.dot(bulge, direction))
    if np.linalg.norm(bulge) < 1e-6:
        bulge = np.array([1.0, 0.0, 0.0])
    bulge /= np.linalg.norm(bulge)
    slack = total - dist
    cum = np.concatenate([[0.0], np.cumsum(lengths)]) / total
    pts = np.array([base + direction * dist * c + bulge * slack * 0.9 * np.sin(np.pi * c) for c in cum])
    pts[-1] = target
    for _ in range(iterations):
        pts[-1] = target
        for i in range(2, -1, -1):
            v = pts[i] - pts[i + 1]
            pts[i] = pts[i + 1] + v / (np.linalg.norm(v) + 1e-9) * lengths[i]
        pts[0] = base
        for i in range(1, 4):
            v = pts[i] - pts[i - 1]
            pts[i] = pts[i - 1] + v / (np.linalg.norm(v) + 1e-9) * lengths[i - 1]
    return pts


def resolve(spec: PoseSpec) -> ResolvedPose:
    fingers = np.array(spec.fingers, dtype=np.float64)
    kind = spec.thumb[0]
    if kind == "abs":
        target = np.array(spec.thumb[1:4], dtype=np.float64)
    elif kind in ("tip", "pip", "dip"):
        fname = spec.thumb[1]
        idx = LONG_FINGERS.index(fname)
        pts = _finger_points(fname, *fingers[idx])
        joint = {"pip": 1, "dip": 2, "tip": 3}[kind]
        target = pts[joint] + np.array(spec.thumb[2:5], dtype=np.float64)
    elif kind == "dir":
        _, az, el, ext = spec.thumb
        az_r, el_r = np.radians(az), np.radians(el)
        d = np.array([np.sin(az_r) * np.cos(el_r), np.cos(az_r) * np.cos(el_r), np.sin(el_r)])
        target = THUMB_BASE + d * sum(THUMB_LEN) * ext
    else:
        raise ValueError(f"unknown thumb spec {spec.thumb!r}")
    return ResolvedPose(fingers, target, spec.roll, spec.yaw, spec.pitch)


def build_local(pose: ResolvedPose) -> np.ndarray:
    """(21, 3) landmark positions in the local hand frame (cm)."""
    lm = np.zeros((21, 3))
    thumb = _thumb_ik(pose.thumb_target)
    for i, idx in enumerate(FINGER_CHAINS["thumb"]):
        lm[idx] = thumb[i]
    for row, name in enumerate(LONG_FINGERS):
        pts = _finger_points(name, *pose.fingers[row])
        for i, idx in enumerate(FINGER_CHAINS[name]):
            lm[idx] = pts[i]
    return lm


def local_to_camera(
    local: np.ndarray,
    roll: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    mirror: bool = False,
) -> np.ndarray:
    """Rotate local landmarks into camera axes (x right, y down, z away). Angles in degrees.

    Roll spins the hand in the image plane (clockwise positive), applied last.
    """
    pts = local.copy()
    if mirror:  # left hand
        pts[:, 0] = -pts[:, 0]
    cam = np.stack([pts[:, 0], -pts[:, 1], -pts[:, 2]], axis=1)  # palm faces the camera
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return cam @ (rz @ rx @ ry).T


def make_observation(
    pose: ResolvedPose,
    center: tuple[float, float] = (0.5, 0.5),
    px_per_cm: float = 12.0,
    frame_size: tuple[int, int] = (640, 480),
    roll: float | None = None,
    yaw: float | None = None,
    pitch: float | None = None,
    mirror: bool = False,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
    t: float = 0.0,
) -> HandObservation:
    """Build a :class:`HandObservation` (image + world landmarks) from a resolved pose."""
    rng = rng or np.random.default_rng(0)
    local = build_local(pose)
    cam = local_to_camera(
        local,
        pose.roll if roll is None else roll,
        pose.yaw if yaw is None else yaw,
        pose.pitch if pitch is None else pitch,
        mirror,
    )
    cam = cam - cam[[0, 5, 9, 13, 17]].mean(axis=0)  # centre on the palm
    world = cam / 100.0
    if noise > 0:
        world = world + rng.normal(0.0, noise * 0.089, world.shape)
    w, h = frame_size
    img = np.zeros((21, 3))
    img[:, 0] = center[0] + cam[:, 0] * px_per_cm / w
    img[:, 1] = center[1] + cam[:, 1] * px_per_cm / h
    img[:, 2] = cam[:, 2] * px_per_cm / w
    if noise > 0:
        img[:, :2] += rng.normal(0.0, noise * 0.089 * px_per_cm, (21, 2)) / np.array([w, h])
    return HandObservation(
        image=img,
        world=world,
        handedness="Left" if mirror else "Right",
        score=0.99,
        frame_size=frame_size,
        t=t,
        extras={"synthetic": True, "chirality_right": not mirror},
    )


# ------------------------------------------------------------------------ pose library
_S_THUMB = ("abs", 0.2, 7.6, 5.8)  # across the front of a fist
_TUCK_RING_PINKY = ("abs", -1.6, 7.2, 5.0)


def _fist(thumb, **kw) -> PoseSpec:
    return PoseSpec(fingers=(F(FIST), F(FIST), F(FIST), F(FIST)), thumb=thumb, **kw)


def _build_library() -> dict[str, PoseSpec]:
    lib: dict[str, PoseSpec] = {}
    lib["A"] = _fist(("abs", 4.0, 9.8, 2.2))
    lib["B"] = PoseSpec(
        (F(EXT, 2), F(EXT, 0), F(EXT, -1), F(EXT, -3)),
        ("abs", -0.5, 4.5, 1.6),
    )
    lib["C"] = PoseSpec(
        (F(CURVE, 3), F(CURVE, 0), F(CURVE, -3), F(CURVE, -6)),
        ("tip", "index", 1.0, -5.2, 0.0),
        roll=90.0,
    )
    lib["D"] = PoseSpec(
        (F(EXT), F((55, 60, 40)), F((58, 62, 42)), F((62, 66, 45))),
        ("tip", "middle", 0.3, 0.0, 0.0),
    )
    lib["E"] = PoseSpec(
        (F(CLAW), F(CLAW), F(CLAW), F(CLAW)),
        ("tip", "ring", 0.0, 0.0, -1.4),
    )
    lib["F"] = PoseSpec(
        (F((45, 55, 25)), F(EXT, 2), F(EXT, -6), F(EXT, -12)),
        ("tip", "index", 0.0, 0.0, 0.0),
    )
    lib["G"] = PoseSpec(
        (F(EXT), F(FIST), F(FIST), F(FIST)),
        ("dir", 10.0, 6.0, 1.0),  # thumb straight, parallel to the index
        roll=-90.0,
    )
    lib["H"] = PoseSpec(
        (F(EXT, 1), F(EXT, -1), F(FIST), F(FIST)),
        ("abs", -1.4, 7.6, 5.0),
        roll=-90.0,
    )
    lib["I"] = PoseSpec(
        (F(FIST), F(FIST), F(FIST), F(EXT, -6)),
        _S_THUMB,
    )
    lib["J"] = lib["I"]
    lib["K"] = PoseSpec(
        (F(EXT, 13), F(EXT, -9), F(FIST), F(FIST)),
        ("pip", "middle", 1.2, 0.8, 1.2),
    )
    lib["L"] = PoseSpec(
        (F(EXT), F(FIST), F(FIST), F(FIST)),
        ("dir", 85.0, 5.0, 1.0),
    )
    lib["M"] = _fist(("abs", -2.2, 7.4, 2.4))
    lib["N"] = _fist(("abs", -0.3, 8.0, 2.6))
    lib["O"] = PoseSpec(
        (F(ROUND, 2), F(ROUND, 0), F((62, 66, 42), -2), F((68, 72, 46), -4)),
        ("tip", "middle", -0.3, 0.0, 0.2),
    )
    lib["P"] = replace(lib["K"], roll=180.0)
    lib["Q"] = replace(lib["G"], roll=180.0)
    lib["R"] = PoseSpec(
        (F(EXT, -8), F(EXT, 8), F(FIST), F(FIST)),
        _TUCK_RING_PINKY,
    )
    lib["S"] = _fist(_S_THUMB)
    lib["T"] = _fist(("abs", 1.5, 8.4, 4.9))
    lib["U"] = PoseSpec(
        (F(EXT, 1), F(EXT, -1), F(FIST), F(FIST)),
        _TUCK_RING_PINKY,
    )
    lib["V"] = PoseSpec(
        (F(EXT, 14), F(EXT, -10), F(FIST), F(FIST)),
        _TUCK_RING_PINKY,
    )
    lib["W"] = PoseSpec(
        (F(EXT, 18), F(EXT, 0), F(EXT, -18), F(FIST)),
        ("abs", -2.8, 6.6, 4.8),
    )
    lib["X"] = PoseSpec(
        (F(HOOK), F(FIST), F(FIST), F(FIST)),
        ("abs", 0.2, 7.4, 5.5),
    )
    lib["Y"] = PoseSpec(
        (F(FIST), F(FIST), F(FIST), F(EXT, -10)),
        ("dir", 85.0, 5.0, 1.0),
    )
    lib["Z"] = PoseSpec(
        (F(EXT), F(FIST), F(FIST), F(FIST)),
        _S_THUMB,
    )
    # Word-level handshape: "I love you" = thumb + index + pinky out.
    lib["ILY"] = PoseSpec(
        (F(EXT, 4), F(FIST), F(FIST), F(EXT, -12)),
        ("dir", 75.0, 5.0, 1.0),
    )
    # Non-letter shapes used by the demo / motion gestures.
    lib["OPEN"] = PoseSpec(
        (F(EXT, 12), F(EXT, 3), F(EXT, -5), F(EXT, -16)),
        ("dir", 60.0, 10.0, 1.0),
    )
    return lib


POSES: dict[str, PoseSpec] = _build_library()
_RESOLVED: dict[str, ResolvedPose] = {}


def resolved(name: str) -> ResolvedPose:
    """Cached resolved pose by letter name."""
    if name not in _RESOLVED:
        _RESOLVED[name] = resolve(POSES[name])
    return _RESOLVED[name]


def letter_observation(letter: str, **kw) -> HandObservation:
    """Observation of a canonical letter pose (see :func:`make_observation` for kwargs)."""
    return make_observation(resolved(letter), **kw)


def random_observation(
    letter: str,
    rng: np.random.Generator,
    noise: float = 0.02,
    roll_jitter: float = 12.0,
    tilt_jitter: float = 14.0,
    mirror: bool | None = None,
    finger_jitter: float = 5.0,
) -> HandObservation:
    """A perturbed sample of ``letter``: random size, position, tilt, hand and joint noise."""
    base = resolved(letter)
    jitter = rng.normal(0.0, finger_jitter, base.fingers.shape)
    jitter[:, 3] *= 0.6  # finger spread wobbles less than finger curl
    pose = ResolvedPose(
        fingers=base.fingers + jitter,
        thumb_target=base.thumb_target + rng.normal(0.0, 0.25, 3),
        roll=base.roll + rng.normal(0.0, roll_jitter),
        yaw=rng.normal(0.0, tilt_jitter),
        pitch=rng.normal(0.0, tilt_jitter),
    )
    return make_observation(
        pose,
        center=(rng.uniform(0.3, 0.7), rng.uniform(0.35, 0.65)),
        px_per_cm=rng.uniform(8.0, 18.0),
        mirror=bool(rng.integers(0, 2)) if mirror is None else mirror,
        noise=noise,
        rng=rng,
    )


def all_letters() -> tuple[str, ...]:
    return ALPHABET
