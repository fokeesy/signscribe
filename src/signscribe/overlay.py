"""OpenCV drawing helpers for the live preview (skeleton, fingertips, HUD widgets)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .constants import BONES, FINGER_CHAINS, FINGER_COLORS_BGR, FINGERTIPS, WRIST

ACCENT = (120, 220, 90)  # BGR green
WARN = (60, 180, 255)
DIM = (150, 150, 150)
WHITE = (245, 245, 245)
DARK = (20, 20, 20)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(img, text, org, scale=0.5, color=WHITE, thick=1, outline=True) -> None:
    """Text with a dark halo so it stays readable on any background."""
    if outline:
        cv2.putText(img, text, org, FONT, scale, DARK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def alpha_rect(img, p0, p1, color, alpha=0.55) -> None:
    x0, y0 = max(int(p0[0]), 0), max(int(p0[1]), 0)
    x1, y1 = min(int(p1[0]), img.shape[1]), min(int(p1[1]), img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, roi)


def draw_skeleton(img, pts: np.ndarray, scale: float = 1.0, dim: bool = False, joints: bool = True) -> None:
    """Draw bones (outlined, coloured per finger) and joint dots. ``pts`` is (21, 2) pixels."""
    p = pts.astype(np.int32)
    thick = max(int(round(3 * scale)), 1)
    order = sorted(range(len(BONES)), key=lambda i: p[BONES[i][1]][1] + p[BONES[i][2]][1])
    for i in order:  # dark under-stroke first for contrast against any skin tone / background
        _, a, b = BONES[i]
        cv2.line(img, tuple(p[a]), tuple(p[b]), DARK, thick + 3, cv2.LINE_AA)
    for i in order:
        name, a, b = BONES[i]
        col = FINGER_COLORS_BGR[name]
        if dim:
            col = tuple(int(c * 0.55 + 60) for c in col)
        cv2.line(img, tuple(p[a]), tuple(p[b]), col, thick, cv2.LINE_AA)
    if not joints:
        return
    tips = set(FINGERTIPS)
    for k in range(21):
        if k in tips:
            r = max(int(round(6.5 * scale)), 3)
            cv2.circle(img, tuple(p[k]), r + 2, DARK, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p[k]), r, WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p[k]), max(r - 3, 1), _finger_color_of(k), -1, cv2.LINE_AA)
        else:
            r = max(int(round((5.5 if k == WRIST else 3.5) * scale)), 2)
            cv2.circle(img, tuple(p[k]), r + 1, DARK, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p[k]), r, (235, 235, 235), -1, cv2.LINE_AA)


def _finger_color_of(idx: int) -> tuple[int, int, int]:
    for name, chain in FINGER_CHAINS.items():
        if idx in chain:
            return FINGER_COLORS_BGR[name]
    return WHITE


def draw_brackets(img, bbox, color, pad=16, length=18, thick=2) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thick, cv2.LINE_AA)
    return x0, y0, x1, y1


def draw_label_chip(img, box, text: str, sub: str, color) -> None:
    """Chip above (or inside, near the top edge) the hand box showing the current letter."""
    x0, y0, x1, _ = box
    w, h = 96, 44
    cx = int((x0 + x1) / 2 - w / 2)
    cy = y0 - h - 18
    if cy < 4:
        cy = 4
    cx = int(np.clip(cx, 4, img.shape[1] - w - 4))
    alpha_rect(img, (cx, cy), (cx + w, cy + h), DARK, 0.7)
    cv2.rectangle(img, (cx, cy), (cx + w, cy + h), color, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.1, 2)
    put_text(img, text, (cx + 10, cy + h - 10), 1.1, color, 2, outline=False)
    put_text(img, sub, (cx + 12 + tw, cy + h - 12), 0.45, WHITE, 1, outline=False)


def draw_arrow(img, start, vec, color, min_len=14.0, max_len=90.0) -> None:
    """Velocity arrow; ``vec`` in pixels (already scaled)."""
    n = math.hypot(*vec)
    if n < min_len:
        return
    if n > max_len:
        vec = (vec[0] * max_len / n, vec[1] * max_len / n)
    end = (int(start[0] + vec[0]), int(start[1] + vec[1]))
    s = (int(start[0]), int(start[1]))
    cv2.arrowedLine(img, s, end, DARK, 5, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(img, s, end, color, 2, cv2.LINE_AA, tipLength=0.3)


def draw_trail(img, points: list[tuple[float, float, float]], now: float, color, life: float = 1.2) -> None:
    """Fading polyline of recent (t, x, y) positions."""
    pts = [(t, x, y) for t, x, y in points if now - t <= life]
    for (_t0, x0, y0), (t1, x1, y1) in zip(pts[:-1], pts[1:]):
        a = 1.0 - (now - t1) / life
        col = tuple(int(c * a + 30 * (1 - a)) for c in color)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), col, max(int(1 + 4 * a), 1), cv2.LINE_AA)


def draw_dial(img, centre, radius, roll_deg: float, color=ACCENT) -> None:
    """Small compass: the tick shows where the hand's fingers point (top = fingers up)."""
    c = (int(centre[0]), int(centre[1]))
    cv2.circle(img, c, radius + 2, DARK, -1, cv2.LINE_AA)
    cv2.circle(img, c, radius, (70, 70, 70), 1, cv2.LINE_AA)
    a = math.radians(roll_deg)
    tip = (int(c[0] + math.sin(a) * radius * 0.85), int(c[1] - math.cos(a) * radius * 0.85))
    cv2.line(img, c, tip, color, 2, cv2.LINE_AA)
    cv2.circle(img, tip, 3, color, -1, cv2.LINE_AA)
    cv2.line(img, (c[0], c[1] - radius), (c[0], c[1] - radius + 4), DIM, 1, cv2.LINE_AA)


def draw_finger_annotations(img, pts: np.ndarray, feats: dict[str, float], show_angles: bool) -> None:
    """Tiny extension percentage next to each fingertip; optionally PIP joint angles."""
    for name, key in (("index", "i"), ("middle", "m"), ("ring", "r"), ("pinky", "p")):
        tip = pts[FINGER_CHAINS[name][3]]
        put_text(img, f"{int(feats['ext_' + key] * 100)}", (int(tip[0]) - 10, int(tip[1]) - 14), 0.42, FINGER_COLORS_BGR[name], 1)
        if show_angles:
            pip = pts[FINGER_CHAINS[name][1]]
            put_text(img, f"{int(feats['pip_' + key])}", (int(pip[0]) + 6, int(pip[1]) + 4), 0.36, WHITE, 1)
    tip = pts[FINGER_CHAINS["thumb"][3]]
    put_text(img, f"{int(feats['th_ext'] * 100)}", (int(tip[0]) - 10, int(tip[1]) - 14), 0.42, FINGER_COLORS_BGR["thumb"], 1)


def draw_hud_line(img, text: str, y: int, color=WHITE) -> None:
    put_text(img, text, (10, y), 0.5, color, 1)


def centered_message(img, lines: list[str], color=WHITE) -> None:
    h, w = img.shape[:2]
    y = h // 2 - 12 * (len(lines) - 1)
    for i, line in enumerate(lines):
        scale = 0.75 if i == 0 else 0.5
        (tw, _), _ = cv2.getTextSize(line, FONT, scale, 1)
        put_text(img, line, ((w - tw) // 2, y + i * 30), scale, color if i == 0 else DIM, 1)
