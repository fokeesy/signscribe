"""Rotation-, scale-, position- and handedness-invariant hand-shape features.

Everything here is computed from metric world landmarks (so it does not care about skin
colour, lighting or background: those only affect *detection*, which the tracker handles)
plus a single scalar, the in-plane roll of the hand. Distances are divided by the palm
length and angles are joint angles, so a hand looks the same to the classifiers whether it
is big or small, left or right, tilted or upright.

All functions are vectorised: pass ``world`` with shape (N, 21, 3) and ``roll`` with shape
(N,). ``extract`` is the single-hand convenience wrapper.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    FINGER_CHAINS,
    INDEX_MCP,
    LONG_FINGERS,
    MIDDLE_MCP,
    PINKY_MCP,
    RING_MCP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)

EPS = 1e-9
_ANGLE_DEADZONE = 7.0  # degrees of joint bend treated as measurement noise
_SHORT = {"index": "i", "middle": "m", "ring": "r", "pinky": "p"}

#: Ordered feature names fed to the learned classifier.
VECTOR_KEYS: tuple[str, ...] = (
    *[f"ext_{s}" for s in "imrp"],
    *[f"{j}_{s}" for j in ("mcp", "pip", "dip") for s in "imrp"],
    "th_ext", "th_mcp", "th_ip", "th_reach",
    "d_ti", "d_tm", "d_tr", "d_tp", "d_im", "d_mr", "d_rp", "d_ir", "d_mp",
    *[f"t_pip_{s}" for s in "imrp"],
    *[f"t_mcp_{s}" for s in "imrp"],
    *[f"t_dip_{s}" for s in "imrp"],
    *[f"tip_pc_{s}" for s in "imrp"],
    "t_pc", "ang_ti", "spread_im", "spread_mr", "spread_rp",
    "t_along", "t_height", "cross_im", "up", "side",
)  # fmt: skip

_ANGLE_KEYS = {k for k in VECTOR_KEYS if k.split("_")[0] in ("mcp", "pip", "dip", "th") and k != "th_ext" and k != "th_reach"}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def _norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=-1)


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between vectors in degrees, shape (N,)."""
    c = np.sum(a * b, axis=-1) / (_norm(a) * _norm(b) + EPS)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def extract_batch(world: np.ndarray, roll: np.ndarray | float) -> dict[str, np.ndarray]:
    """Compute the named feature dictionary for a batch of hands.

    Args:
        world: (N, 21, 3) or (21, 3) metric landmarks.
        roll: (N,) or scalar in-plane hand rotation in radians (0 = fingers up).

    Returns:
        dict of feature name -> (N,) arrays. Lengths are in palm units, angles in degrees.
    """
    w = np.asarray(world, dtype=np.float64)
    if w.ndim == 2:
        w = w[None]
    n = w.shape[0]
    roll = np.broadcast_to(np.asarray(roll, dtype=np.float64), (n,))

    # Palm scale: mean wrist -> knuckle distance. Robust to foreshortening because the
    # landmarks are 3D.
    knuckles = w[:, [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]]
    scale = _norm(knuckles - w[:, WRIST : WRIST + 1]).mean(axis=1) + EPS
    p = (w - w[:, WRIST : WRIST + 1]) / scale[:, None, None]

    f: dict[str, np.ndarray] = {}
    tips: dict[str, np.ndarray] = {}
    dirs: dict[str, np.ndarray] = {}

    # ---- long fingers: joint flexion, curl, extension --------------------------------
    for name in LONG_FINGERS:
        s = _SHORT[name]
        mcp, pip, dip, tip = FINGER_CHAINS[name]
        v0 = p[:, mcp] - p[:, WRIST]
        v1 = p[:, pip] - p[:, mcp]
        v2 = p[:, dip] - p[:, pip]
        v3 = p[:, tip] - p[:, dip]
        f[f"mcp_{s}"] = _angle(v0, v1)
        f[f"pip_{s}"] = _angle(v1, v2)
        f[f"dip_{s}"] = _angle(v2, v3)
        # Angles are magnitudes, so landmark noise always *adds* apparent bend to a straight
        # finger. A small dead-zone removes that bias without touching genuine curls.
        curl = sum(np.maximum(f[f"{j}_{s}"] - _ANGLE_DEADZONE, 0.0) for j in ("mcp", "pip", "dip"))
        f[f"curl_{s}"] = curl
        # 1 = ruler straight, 0 = tightly closed fist.
        f[f"ext_{s}"] = np.clip(1.0 - curl / 235.0, 0.0, 1.0)
        seg = _norm(v1) + _norm(v2) + _norm(v3)
        f[f"reach_{s}"] = _norm(p[:, tip] - p[:, mcp]) / (seg + EPS)
        tips[s] = p[:, tip]
        dirs[s] = _unit(p[:, tip] - p[:, mcp])

    # ---- thumb -------------------------------------------------------------------------
    t1 = p[:, THUMB_MCP] - p[:, THUMB_CMC]
    t2 = p[:, THUMB_IP] - p[:, THUMB_MCP]
    t3 = p[:, THUMB_TIP] - p[:, THUMB_IP]
    f["th_mcp"] = _angle(t1, t2)
    f["th_ip"] = _angle(t2, t3)
    f["th_ext"] = np.clip(1.0 - (f["th_mcp"] + f["th_ip"]) / 110.0, 0.0, 1.0)
    f["th_reach"] = _norm(p[:, THUMB_TIP] - p[:, THUMB_CMC]) / (_norm(t1) + _norm(t2) + _norm(t3) + EPS)
    th_tip = p[:, THUMB_TIP]
    th_dir = _unit(p[:, THUMB_TIP] - p[:, THUMB_MCP])

    # ---- thumb <-> fingers proximity ---------------------------------------------------
    for name in LONG_FINGERS:
        s = _SHORT[name]
        mcp, pip, dip, _tip = FINGER_CHAINS[name]
        f[f"d_t{s}"] = _norm(th_tip - tips[s])
        f[f"t_mcp_{s}"] = _norm(th_tip - p[:, mcp])
        f[f"t_pip_{s}"] = _norm(th_tip - p[:, pip])
        f[f"t_dip_{s}"] = _norm(th_tip - p[:, dip])

    # ---- finger <-> finger -------------------------------------------------------------
    f["d_im"] = _norm(tips["i"] - tips["m"])
    f["d_mr"] = _norm(tips["m"] - tips["r"])
    f["d_rp"] = _norm(tips["r"] - tips["p"])
    f["d_ir"] = _norm(tips["i"] - tips["r"])
    f["d_mp"] = _norm(tips["m"] - tips["p"])
    f["spread_im"] = _angle(dirs["i"], dirs["m"])
    f["spread_mr"] = _angle(dirs["m"], dirs["r"])
    f["spread_rp"] = _angle(dirs["r"], dirs["p"])
    f["ang_ti"] = _angle(th_dir, dirs["i"])

    # ---- palm frame --------------------------------------------------------------------
    centre = p[:, [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=1)
    for s in "imrp":
        f[f"tip_pc_{s}"] = _norm(tips[s] - centre)
    f["t_pc"] = _norm(th_tip - centre)

    up_ax = _unit(p[:, MIDDLE_MCP] - p[:, WRIST])
    lat = p[:, PINKY_MCP] - p[:, INDEX_MCP]  # index knuckle -> pinky knuckle
    lat_len2 = np.sum(lat * lat, axis=-1) + EPS
    # Where the thumb tip sits across the knuckle line: 0 = index side, 1 = pinky side.
    f["t_along"] = np.sum((th_tip - p[:, INDEX_MCP]) * lat, axis=-1) / lat_len2
    # Height of the thumb tip along the palm axis (1 = knuckle level).
    f["t_height"] = np.sum(th_tip * up_ax, axis=-1)
    across = _unit(-lat)  # pinky -> index direction, handedness free
    x_idx = np.sum((tips["i"] - p[:, PINKY_MCP]) * across, axis=-1)
    x_mid = np.sum((tips["m"] - p[:, PINKY_MCP]) * across, axis=-1)
    # >0 when index and middle fingertips have swapped sides (crossed fingers, letter R).
    f["cross_im"] = x_mid - x_idx

    # ---- orientation in the image plane -----------------------------------------------
    f["up"] = np.cos(roll)  # +1 fingers up, -1 fingers down
    f["side"] = np.abs(np.sin(roll))  # 1 = pointing left or right
    f["roll"] = roll
    f["scale"] = scale
    return f


def extract(world: np.ndarray, roll: float) -> dict[str, float]:
    """Single-hand wrapper around :func:`extract_batch` returning plain floats."""
    feats = extract_batch(world, roll)
    return {k: float(v[0]) for k, v in feats.items()}


def to_vector(feats: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the learned-model features into an (N, D) matrix (angles scaled to ~0..1)."""
    cols = []
    for k in VECTOR_KEYS:
        col = feats[k]
        if k in _ANGLE_KEYS or k.startswith(("spread_", "ang_")):
            col = col / 90.0
        cols.append(col)
    return np.stack(cols, axis=1).astype(np.float32)


def vector_dim() -> int:
    return len(VECTOR_KEYS)
