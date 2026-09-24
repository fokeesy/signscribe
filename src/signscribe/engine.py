"""The real-time pipeline: frame -> detect -> smooth -> recognise -> compose text -> draw.

:class:`Engine` owns every stage and runs them on a worker thread so the GUI never blocks.
Frame handling lives in :meth:`Engine.process_packet`, which is thread-free and deterministic;
tests and the CLI ``--headless`` mode call it directly with virtual timestamps.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import overlay
from .camera import CameraError, FramePacket
from .composer import Composer, TextEvent
from .config import Settings
from .dynamic import HandDynamics, MotionEvent, MotionRecognizer
from .hand import HandObservation, rotation_angles
from .model import MLPClassifier
from .preprocess import LightingAdapter, LightingInfo
from .recognizer import Prediction, Recognizer
from .tracker import HandTracker
from .training import SampleStore

log = logging.getLogger("signscribe")


@dataclass
class FrameResult:
    """Everything the UI needs to show for one processed frame."""

    frame: np.ndarray | None = None
    t: float = 0.0
    present: bool = False
    hands: int = 0
    prediction: Prediction = field(default_factory=Prediction)
    stable_label: str | None = None
    progress: float = 0.0
    fingers: dict[str, float] = field(default_factory=dict)  # thumb..pinky extension 0..1
    rotation: tuple[float, float, float] | None = None  # roll, pitch, yaw (degrees)
    speed: float = 0.0  # palm lengths / second
    handedness: str = ""
    hand_score: float = 0.0
    lighting: LightingInfo = field(default_factory=LightingInfo)
    fps: float = 0.0
    latency_ms: float = 0.0
    mode: str = "rules"
    motion: MotionEvent | None = None
    recording: dict | None = None
    source_name: str = ""
    demo: bool = False


class Engine:
    def __init__(
        self,
        settings: Settings,
        source=None,
        model: MLPClassifier | None = None,
        store: SampleStore | None = None,
    ) -> None:
        self.settings = settings
        self.source = source
        self.recognizer = Recognizer(settings.recognition_mode, model, settings.word_gestures)
        self.composer = Composer(settings)
        self.motion = MotionRecognizer(settings.word_gestures)
        self.dynamics = HandDynamics()
        self.lighting = LightingAdapter(settings.lighting)
        self.store = store if store is not None else SampleStore()
        self.tracker: HandTracker | None = None
        self.events: queue.Queue[TextEvent] = queue.Queue()

        self._tracker_key: tuple | None = None
        self._latest: FrameResult | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: str | None = None

        self._fps = 0.0
        self._last_done = 0.0
        self._miss = 0
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_seen = -1e9
        self._primary_side: str | None = None
        self._trail: deque[tuple[float, float, float]] = deque(maxlen=48)
        self._rec: dict | None = None
        self.paused = False  # when True, recognition still runs but nothing is typed
        self.calibrating = False  # set by the calibration wizard: typing is off for its whole run

    # ------------------------------------------------------------------ model / config
    def set_model(self, model: MLPClassifier | None) -> None:
        self.recognizer.set_model(model)

    @property
    def model(self) -> MLPClassifier | None:
        return self.recognizer.model

    def _sync_settings(self) -> None:
        s = self.settings
        self.recognizer.mode = s.recognition_mode
        self.recognizer.include_words = s.word_gestures
        self.motion.enable_words = s.word_gestures
        self.lighting.mode = s.lighting
        if self.tracker is not None:
            self.tracker.set_smoothing(s.smoothing)

    def _ensure_tracker(self) -> HandTracker:
        s = self.settings
        key = (s.max_hands, s.model_complexity, s.min_detection_conf, s.min_tracking_conf, s.mirror)
        if self.tracker is None or key != self._tracker_key:
            if self.tracker is not None:
                self.tracker.close()
            self.tracker = HandTracker(
                max_hands=s.max_hands,
                complexity=s.model_complexity,
                min_detection_conf=s.min_detection_conf,
                min_tracking_conf=s.min_tracking_conf,
                mirrored_input=s.mirror,
                smoothing=s.smoothing,
            )
            self._tracker_key = key
        return self.tracker

    # ----------------------------------------------------------------------- recording
    def start_recording(self, label: str, target: int = 60) -> None:
        """Capture ``target`` frames of the primary hand as samples of ``label``."""
        self._rec = {"label": label, "target": target, "count": 0, "done": False, "skipped": 0}
        self.composer.reset_state()

    def stop_recording(self) -> None:
        self._rec = None

    def recording_state(self) -> dict | None:
        """Snapshot of the active recording (label / count / target / done) or None."""
        return dict(self._rec) if self._rec else None

    # ------------------------------------------------------------------- one frame
    def _detect(self, frame: np.ndarray, t: float) -> list[HandObservation]:
        tracker = self._ensure_tracker()
        roi = self._last_bbox if t - self._last_seen < 0.6 else None
        hands = tracker.process(self.lighting.process(frame, roi), t)
        if hands:
            self._miss = 0
            return hands
        self._miss += 1
        # Nothing found for a few frames: periodically retry on a contrast-enhanced copy,
        # which rescues hands in awkward lighting without paying for it every frame.
        if self._miss >= 4 and self._miss % 3 == 0 and self.settings.lighting != "off":
            hands = tracker.process(self.lighting.force_enhance(frame), t)
            if hands:
                self._miss = 0
        return hands

    def _pick_primary(self, hands: list[HandObservation]) -> HandObservation | None:
        dom = self.settings.dominant_hand
        cands = hands if dom == "Auto" else [h for h in hands if h.handedness == dom]
        if not cands:
            return None
        best = max(cands, key=lambda h: h.palm_px * (0.5 + h.score))
        if len(cands) > 1 and self._primary_side:  # stick with the current hand unless clearly outsized
            same = [h for h in cands if h.handedness == self._primary_side]
            if same and same[0].palm_px >= 0.75 * best.palm_px:
                best = same[0]
        self._primary_side = best.handedness
        return best

    def process_packet(self, pkt: FramePacket) -> FrameResult:
        s = self.settings
        self._sync_settings()
        t = pkt.t
        simulated = pkt.hands is not None
        frame = pkt.frame
        if not simulated and s.mirror:
            frame = cv2.flip(frame, 1)
        elif not simulated:
            frame = frame.copy()

        hands = pkt.hands if simulated else self._detect(frame, t)
        primary = self._pick_primary(hands)

        pred = Prediction()
        rotation = None
        motion_ev: MotionEvent | None = None
        text_events: list[TextEvent] = []
        speed = 0.0
        fingers: dict[str, float] = {}
        live = not (self.paused or self.calibrating or self._rec is not None)  # is typing allowed?

        if primary is not None:
            self._last_seen = t
            self._last_bbox = primary.bbox_px
            speed = self.dynamics.update(primary)
            pred = self.recognizer.predict(primary)
            f = pred.features
            fingers = {"thumb": f["th_ext"], "index": f["ext_i"], "middle": f["ext_m"], "ring": f["ext_r"], "pinky": f["ext_p"]}
            rotation = rotation_angles(primary.world, bool(primary.extras.get("chirality_right", True)))
            if live:
                motion_ev = self.motion.update(primary, f)
                text_events += self.composer.update(t, True, pred.label, pred.confidence, speed)
                if motion_ev is not None:
                    text_events += self.composer.on_motion(motion_ev)
            else:
                self.composer.reset_state()
            self._record(primary, speed)
            self._update_trail(primary, f, t, speed)
        else:
            self.dynamics.reset()
            motion_ev = self.motion.lost(t) if live else None
            if motion_ev is not None:  # a letter traced just before the hand dropped out of view
                text_events += self.composer.on_motion(motion_ev)
            if live:
                text_events += self.composer.update(t, False, None, 0.0, 0.0)
            self._trail.clear()

        for e in text_events:
            self.events.put(e)

        self._draw(frame, hands, primary, pred, rotation, speed, t, simulated)

        now = time.perf_counter()
        if self._last_done:
            inst = 1.0 / max(now - self._last_done, 1e-4)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_done = now
        latency = 0.0 if simulated else (now - t) * 1000.0

        return FrameResult(
            frame=frame,
            t=t,
            present=primary is not None,
            hands=len(hands),
            prediction=pred,
            stable_label=self.composer.stable_label,
            progress=self.composer.progress,
            fingers=fingers,
            rotation=rotation,
            speed=speed,
            handedness=primary.handedness if primary else "",
            hand_score=primary.score if primary else 0.0,
            lighting=self.lighting.info,
            fps=self._fps,
            latency_ms=max(latency, 0.0),
            mode=self.recognizer.effective_mode,
            motion=motion_ev,
            recording=dict(self._rec) if self._rec else None,
            source_name=getattr(self.source, "name", ""),
            demo=bool(pkt.info.get("demo")),
        )

    def _record(self, primary: HandObservation, speed: float) -> None:
        rec = self._rec
        if rec is None or rec["done"]:
            return
        if primary.score < 0.6 or speed > 2.5:
            rec["skipped"] += 1
            return
        self.store.add(rec["label"], primary)
        rec["count"] += 1
        if rec["count"] >= rec["target"]:
            rec["done"] = True

    def _update_trail(self, obs: HandObservation, f: dict[str, float], t: float, speed: float) -> None:
        """Record the tracing fingertip, but only for the shapes that draw letters (J and Z)."""
        only_pinky = f["ext_p"] > 0.6 and max(f["ext_i"], f["ext_m"], f["ext_r"]) < 0.5
        only_index = f["ext_i"] > 0.6 and max(f["ext_m"], f["ext_r"], f["ext_p"]) < 0.5
        if speed < 0.5 or not (only_pinky or only_index):
            return
        x, y = obs.pixels[20 if only_pinky else 8]
        self._trail.append((t, float(x), float(y)))

    # ------------------------------------------------------------------------ drawing
    def _draw(self, frame, hands, primary, pred, rotation, speed, t, simulated) -> None:
        s = self.settings
        h, w = frame.shape[:2]
        for obs in hands:
            is_primary = obs is primary
            pts = obs.pixels
            scale = float(np.clip(obs.palm_px / 105.0, 0.6, 1.7))
            overlay.draw_skeleton(frame, pts, scale, dim=not is_primary)
            if not is_primary:
                continue
            good = pred.label is not None and pred.confidence >= s.min_confidence
            colour = overlay.ACCENT if good else overlay.WARN
            box = overlay.draw_brackets(frame, obs.bbox_px, colour)
            if s.show_labels:
                overlay.draw_finger_annotations(frame, pts, pred.features, s.show_angles)
                text = pred.label if pred.label else "?"
                if pred.label and len(pred.label) > 2:
                    text = pred.label.split()[0][:3]
                overlay.draw_label_chip(frame, box, text, f"{int(pred.confidence * 100)}%", colour)
            centre = obs.center_px
            if self.composer.progress > 0:
                r = int(obs.palm_px * 0.9)
                cv2.ellipse(frame, (int(centre[0]), int(centre[1])), (r, r), -90, 0, 360, (60, 60, 60), 2, cv2.LINE_AA)
                cv2.ellipse(
                    frame,
                    (int(centre[0]), int(centre[1])),
                    (r, r),
                    -90,
                    0,
                    360 * self.composer.progress,
                    overlay.ACCENT,
                    4,
                    cv2.LINE_AA,
                )
            if speed > 0.6:
                v = self.dynamics.velocity * obs.palm_px * 0.35
                overlay.draw_arrow(frame, centre, (float(v[0]), float(v[1])), overlay.WARN)
        if s.show_trail and len(self._trail) > 1:
            overlay.draw_trail(frame, list(self._trail), t, overlay.WARN)

        # HUD
        overlay.put_text(frame, f"{self._fps:4.0f} FPS", (10, 22), 0.55, overlay.ACCENT, 1)
        if rotation is not None:
            overlay.draw_dial(frame, (w - 34, 38), 24, rotation[0])
            overlay.put_text(frame, f"roll {rotation[0]:+.0f}", (w - 108, h - 46), 0.45, overlay.WHITE, 1)
            overlay.put_text(frame, f"pitch {rotation[1]:+.0f}", (w - 108, h - 28), 0.45, overlay.WHITE, 1)
            overlay.put_text(frame, f"yaw {rotation[2]:+.0f}", (w - 108, h - 10), 0.45, overlay.WHITE, 1)
        if simulated:
            overlay.put_text(frame, "DEMO - simulated hand", (10, h - 12), 0.5, overlay.WARN, 1)
        if primary is None:
            overlay.centered_message(frame, ["Show your hand to the camera", "Palm toward the lens, fingers clear of your body"])
        if self._rec is not None:
            rec = self._rec
            overlay.put_text(frame, f"REC {rec['label']}  {rec['count']}/{rec['target']}", (10, 46), 0.6, (80, 80, 255), 2)

    # -------------------------------------------------------------------------- thread
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="signscribe-engine", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.source.start()
        except CameraError as exc:
            self.error = str(exc)
            return
        consecutive_errors = 0
        while not self._stop.is_set():
            pkt = self.source.read(0.5)
            if pkt is None:
                if not self.source.alive:
                    self.error = getattr(self.source, "error", None) or "The video source stopped."
                    break
                continue
            try:
                result = self.process_packet(pkt)
                consecutive_errors = 0
            except Exception as exc:  # keep the UI alive; surface persistent failures
                consecutive_errors += 1
                log.exception("frame processing failed")
                if consecutive_errors >= 30:
                    self.error = f"Processing keeps failing: {exc}"
                    break
                continue
            with self._lock:
                self._latest = result
        self.source.stop()

    def latest(self) -> FrameResult | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
