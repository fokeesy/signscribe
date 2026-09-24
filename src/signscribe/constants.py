"""Landmark indices and skeleton topology for the 21-point hand model."""

from __future__ import annotations

# --- MediaPipe hand landmark indices -------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
LONG_FINGERS = ("index", "middle", "ring", "pinky")

#: (base, joint1, joint2, tip) landmark indices for every finger.
FINGER_CHAINS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

#: Skeleton edges as (finger name, from, to) so the overlay can colour by finger.
BONES = (
    ("palm", WRIST, THUMB_CMC),
    ("palm", WRIST, INDEX_MCP),
    ("palm", INDEX_MCP, MIDDLE_MCP),
    ("palm", MIDDLE_MCP, RING_MCP),
    ("palm", RING_MCP, PINKY_MCP),
    ("palm", WRIST, PINKY_MCP),
    ("thumb", THUMB_CMC, THUMB_MCP),
    ("thumb", THUMB_MCP, THUMB_IP),
    ("thumb", THUMB_IP, THUMB_TIP),
    ("index", INDEX_MCP, INDEX_PIP),
    ("index", INDEX_PIP, INDEX_DIP),
    ("index", INDEX_DIP, INDEX_TIP),
    ("middle", MIDDLE_MCP, MIDDLE_PIP),
    ("middle", MIDDLE_PIP, MIDDLE_DIP),
    ("middle", MIDDLE_DIP, MIDDLE_TIP),
    ("ring", RING_MCP, RING_PIP),
    ("ring", RING_PIP, RING_DIP),
    ("ring", RING_DIP, RING_TIP),
    ("pinky", PINKY_MCP, PINKY_PIP),
    ("pinky", PINKY_PIP, PINKY_DIP),
    ("pinky", PINKY_DIP, PINKY_TIP),
)

#: BGR colours per finger for the overlay.
FINGER_COLORS_BGR = {
    "palm": (200, 200, 200),
    "thumb": (60, 170, 255),
    "index": (120, 220, 80),
    "middle": (255, 200, 60),
    "ring": (255, 120, 180),
    "pinky": (110, 110, 255),
}

#: Letters the built-in rules and the trainable model know about.
ALPHABET = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#: Letters that need motion (J and Z are traced in the air).
MOTION_LETTERS = ("J", "Z")
