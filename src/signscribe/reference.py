"""Reference skeleton renderings and one-line signing tips for every supported sign.

The renderings are generated from :mod:`signscribe.synthetic` poses, so they ship as code, not
as copyrighted artwork. They are schematic aids (hand skeleton seen slightly from the side),
not photographs; consult a standard ASL chart if a description is unclear.
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import ALPHABET
from .overlay import DARK, WHITE, draw_skeleton, put_text
from .synthetic import POSES, build_local, local_to_camera, resolved

TIPS: dict[str, str] = {
    "A": "Fist, thumb resting along the side of the index finger.",
    "B": "Four fingers straight up together, thumb folded across the palm.",
    "C": "Fingers and thumb curved to make a 'C'.",
    "D": "Index finger up; other fingers curl to touch the thumb.",
    "E": "Fingertips curled down, thumb tucked underneath.",
    "F": "Thumb and index touch in a circle; other three fingers up.",
    "G": "Index and thumb point sideways, parallel, like a small pinch.",
    "H": "Index and middle fingers together, pointing sideways.",
    "I": "Fist with the little finger up.",
    "J": "Sign 'I', then draw a J in the air with the little finger.",
    "K": "Index and middle up in a V, thumb touching the middle finger.",
    "L": "Index up and thumb out: an 'L' shape.",
    "M": "Fist with the thumb tucked under three fingers.",
    "N": "Fist with the thumb tucked under two fingers.",
    "O": "All fingertips meet the thumb, forming an 'O'.",
    "P": "Like K, but pointing down.",
    "Q": "Like G, but pointing down.",
    "R": "Index and middle fingers crossed, pointing up.",
    "S": "Fist with the thumb across the front of the fingers.",
    "T": "Fist with the thumb poking out between index and middle.",
    "U": "Index and middle fingers straight up, together.",
    "V": "Index and middle fingers up and apart (peace sign).",
    "W": "Index, middle and ring fingers up and apart.",
    "X": "Index finger hooked, other fingers in a fist.",
    "Y": "Thumb and little finger out ('hang loose').",
    "Z": "Index finger draws a Z in the air.",
    "I LOVE YOU": "Thumb, index and little finger out.",
    "Hello": "Wave an open hand side to side.",
    "Yes": "Nod a closed fist up and down.",
}


def render_reference(name: str, size: int = 160, yaw: float = 22.0, pitch: float = -8.0) -> np.ndarray:
    """BGR image of the reference skeleton for ``name`` (a key of ``POSES``)."""
    key = "ILY" if name == "I LOVE YOU" else name
    img = np.full((size, size, 3), 26, np.uint8)
    if key not in POSES:
        return img
    pose = resolved(key)
    cam = local_to_camera(build_local(pose), pose.roll, yaw, pitch)
    cam = cam - (cam.min(axis=0) + cam.max(axis=0)) / 2  # centre the bounding box
    span = max(np.ptp(cam[:, 0]), np.ptp(cam[:, 1]), 1.0)
    sc = size * 0.8 / span
    pts = np.stack([cam[:, 0] * sc + size / 2, cam[:, 1] * sc + size / 2], axis=1)
    draw_skeleton(img, pts, scale=size / 220.0)
    return img


def contact_sheet(names: list[str] | None = None, cols: int = 7, size: int = 200) -> np.ndarray:
    """Grid of labelled reference skeletons (used for the docs)."""
    names = names or list(ALPHABET)
    tiles = []
    for n in names:
        tile = render_reference(n, size)
        put_text(tile, n, (10, 26), 0.8, WHITE, 2)
        cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (55, 55, 55), 1)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], DARK[0]))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)
