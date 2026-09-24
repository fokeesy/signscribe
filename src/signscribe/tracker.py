"""MediaPipe hand tracking wrapped into :class:`~signscribe.hand.HandObservation` objects.

MediaPipe Hands runs a palm detector followed by a 21-landmark regression network. It returns
both image-space landmarks (for drawing) and metric 3D "world" landmarks (for shape analysis),
plus a left/right classification. Detection quality is independent of skin colour and
background clutter; exposure problems are handled upstream by
:class:`~signscribe.preprocess.LightingAdapter`.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .filters import OneEuroFilter
from .hand import HandObservation


class TrackerUnavailable(RuntimeError):
    """Raised when MediaPipe is missing or is a version without the Hands solution."""


def _swap(label: str) -> str:
    return "Left" if label == "Right" else "Right"


class HandTracker:
    """Detects up to ``max_hands`` hands per frame and smooths their landmarks.

    Args:
        max_hands: how many hands to track.
        complexity: 0 = fastest model, 1 = most accurate landmark model.
        min_detection_conf / min_tracking_conf: MediaPipe confidence thresholds.
        mirrored_input: True when frames were flipped horizontally before being passed in
            (selfie view). MediaPipe's handedness label assumes this, so it is swapped
            otherwise.
        smoothing: 0..1. 0 = raw landmarks (lowest latency), 1 = steadiest.
        static: treat every frame as an unrelated still image (dataset processing).
    """

    def __init__(
        self,
        max_hands: int = 2,
        complexity: int = 1,
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        mirrored_input: bool = True,
        smoothing: float = 0.5,
        static: bool = False,
    ) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("GLOG_minloglevel", "3")
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TrackerUnavailable("MediaPipe is not installed. Run: pip install mediapipe") from exc
        if not hasattr(mp, "solutions"):  # pragma: no cover
            raise TrackerUnavailable(
                "This MediaPipe build no longer ships mp.solutions. Install a compatible version: "
                "pip install 'mediapipe>=0.10.5,<0.10.22'"
            )
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static,
            max_num_hands=max_hands,
            model_complexity=complexity,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self.mirrored_input = mirrored_input
        self.set_smoothing(smoothing)
        self._filters: dict[str, tuple[OneEuroFilter, OneEuroFilter]] = {}
        self._label_hist: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ configuration
    def set_smoothing(self, smoothing: float) -> None:
        s = float(np.clip(smoothing, 0.0, 1.0))
        # min_cutoff falls (steadier) as s rises; beta keeps fast moves responsive.
        self._image_params = {"min_cutoff": 8.0 - 7.0 * s, "beta": 30.0, "d_cutoff": 3.0}
        self._world_params = {"min_cutoff": 6.0 - 5.0 * s, "beta": 40.0, "d_cutoff": 3.0}
        for f_img, f_world in getattr(self, "_filters", {}).values():
            f_img.min_cutoff, f_img.beta = self._image_params["min_cutoff"], self._image_params["beta"]
            f_world.min_cutoff, f_world.beta = self._world_params["min_cutoff"], self._world_params["beta"]
        self._smoothing = s

    def close(self) -> None:
        self._hands.close()

    # --------------------------------------------------------------------- processing
    def process(self, frame_bgr: np.ndarray, t: float) -> list[HandObservation]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            self._filters.clear()
            return []

        world_lists = getattr(res, "multi_hand_world_landmarks", None) or [None] * len(res.multi_hand_landmarks)
        found: list[HandObservation] = []
        for lm, wlm, handed in zip(res.multi_hand_landmarks, world_lists, res.multi_handedness):
            cls = handed.classification[0]
            raw_label, score = cls.label, float(cls.score)
            image = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float64)
            if wlm is not None:
                world = np.array([[p.x, p.y, p.z] for p in wlm.landmark], dtype=np.float64)
            else:  # very old MediaPipe: approximate metric coordinates from the image
                world = self._approx_world(image, w, h)
            anatomical = raw_label if self.mirrored_input else _swap(raw_label)
            obs = HandObservation(
                image=image,
                world=world,
                handedness=anatomical,
                score=score,
                frame_size=(w, h),
                t=t,
                # MediaPipe reports "Left" for shapes that look like a right hand in the image.
                extras={"chirality_right": raw_label == "Left"},
            )
            found.append(obs)

        found.sort(key=lambda o: -o.score)
        found = found[:2]
        # Smooth per handedness slot so left/right filters never mix.
        seen = set()
        for obs in found:
            key = obs.handedness
            if key in seen:
                key = key + "2"
            seen.add(key)
            self._smooth(obs, key, t)
        for stale in set(self._filters) - seen:
            self._filters.pop(stale, None)
        return found

    def _smooth(self, obs: HandObservation, key: str, t: float) -> None:
        if self._smoothing <= 0.0:
            return
        if key not in self._filters:
            self._filters[key] = (OneEuroFilter(**self._image_params), OneEuroFilter(**self._world_params))
        f_img, f_world = self._filters[key]
        obs.image = f_img(obs.image, t)
        obs.world = f_world(obs.world, t)

    @staticmethod
    def _approx_world(image: np.ndarray, w: int, h: int) -> np.ndarray:  # pragma: no cover
        pts = image * np.array([w, h, w])
        pts = pts - pts[[0, 5, 9, 13, 17]].mean(axis=0)
        palm = np.linalg.norm(pts[9] - pts[0]) + 1e-6
        return pts / palm * 0.09
