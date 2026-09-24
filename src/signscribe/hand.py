"""Data model for one tracked hand plus small geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import INDEX_MCP, MIDDLE_MCP, PALM_POINTS, PINKY_MCP, RING_MCP, WRIST


@dataclass
class HandObservation:
    """One hand in one frame.

    Attributes:
        image: (21, 3) landmarks. x and y are normalised to the frame (0..1), z is the
            relative depth in the same units as x (smaller = closer to the camera).
        world: (21, 3) metric landmarks in metres, centred on the hand and aligned with
            the camera axes. Used for every shape feature because it is scale free.
        handedness: anatomical hand of the *user* ("Left" or "Right").
        score: tracker confidence for this hand (0..1).
        frame_size: (width, height) of the frame the landmarks refer to.
        t: capture timestamp in seconds (monotonic clock).
    """

    image: np.ndarray
    world: np.ndarray
    handedness: str = "Right"
    score: float = 1.0
    frame_size: tuple[int, int] = (640, 480)
    t: float = 0.0
    extras: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ pixel space
    @property
    def pixels(self) -> np.ndarray:
        """(21, 2) landmark positions in pixels."""
        w, h = self.frame_size
        return self.image[:, :2] * np.array([w, h], dtype=np.float64)

    @property
    def palm_px(self) -> float:
        """Apparent palm length in pixels (wrist -> knuckle centroid)."""
        p = self.pixels
        knuckles = p[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)
        return float(max(np.linalg.norm(knuckles - p[WRIST]), 1.0))

    @property
    def center_px(self) -> np.ndarray:
        """Palm centre in pixels (mean of wrist and the four knuckles)."""
        return self.pixels[list(PALM_POINTS)].mean(axis=0)

    @property
    def bbox_px(self) -> tuple[int, int, int, int]:
        """Tight bounding box (x0, y0, x1, y1) in pixels."""
        p = self.pixels
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)

    @property
    def roll(self) -> float:
        """In-plane rotation of the hand in radians (0 = fingers up, +ve = clockwise)."""
        return roll_from_pixels(self.pixels)


def roll_from_pixels(pixels: np.ndarray) -> float:
    """Angle of the wrist -> middle-knuckle axis; 0 points up, +pi/2 points right."""
    v = pixels[MIDDLE_MCP] - pixels[WRIST]
    return float(np.arctan2(v[0], -v[1]))


def rotation_angles(world: np.ndarray, chirality_right: bool = True) -> tuple[float, float, float]:
    """Estimate hand (roll, pitch, yaw) in degrees from metric landmarks.

    Conventions (camera axes: x right, y down, z away from the lens):
        roll  - in-plane tilt, 0 = fingers straight up, positive = clockwise on screen
        pitch - fingers tilting toward (+) or away from (-) the camera
        yaw   - palm turning to the signer's left/right, 0 = palm square to the lens

    Args:
        world: (21, 3) landmarks.
        chirality_right: True when the hand *looks* like a right hand in the image.
    """
    up = world[MIDDLE_MCP] - world[WRIST]
    across = world[INDEX_MCP] - world[PINKY_MCP]
    up_n = up / (np.linalg.norm(up) + 1e-9)
    across = across - up_n * float(np.dot(across, up_n))
    across_n = across / (np.linalg.norm(across) + 1e-9)
    normal = np.cross(across_n, up_n)  # points out of the palm for a right-looking hand
    if not chirality_right:
        normal = -normal

    roll = np.degrees(np.arctan2(up_n[0], -up_n[1]))
    # Fingers pointing toward the camera => z component negative (z points away).
    pitch = np.degrees(np.arcsin(np.clip(-up_n[2], -1.0, 1.0)))
    # Palm normal pointing at the lens has z < 0; yaw grows as it swings sideways.
    yaw = np.degrees(np.arctan2(normal[0], -normal[2] + 1e-9))
    return float(roll), float(pitch), float(yaw)
