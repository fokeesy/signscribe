"""Guided calibration: walks through the alphabet recording samples of *your* hand."""

from __future__ import annotations

import time

#: Order of the guided session. J and Z are traced motions that reuse the I / pointing shapes.
WIZARD_LETTERS = [*"ABCDEFGHIKLMNOPQRSTUVWXY", "I LOVE YOU"]

COUNTDOWN = 3.0  # seconds to form the sign before recording starts
RECORD_FRAMES = 70
TIMEOUT = 14.0  # give up on a letter if no hand is seen for this long
SAVED_PAUSE = 0.6


class Calibrator:
    """State machine: countdown -> recording -> short pause -> next letter."""

    def __init__(self, engine_getter, clock=time.perf_counter) -> None:
        self._engine = engine_getter  # callable returning the *current* engine
        self._clock = clock
        self.letters: list[str] = []
        self.index = 0
        self.phase = "idle"  # idle | countdown | recording | saved | done
        self._phase_t = 0.0
        self.paused = False
        self.recorded: list[str] = []

    # ------------------------------------------------------------------- controls
    @property
    def active(self) -> bool:
        return self.phase not in ("idle", "done")

    @property
    def current(self) -> str | None:
        return self.letters[self.index] if self.active and self.index < len(self.letters) else None

    def start(self, letters: list[str] | None = None) -> None:
        self.letters = list(letters or WIZARD_LETTERS)
        self.index = 0
        self.recorded = []
        self.paused = False
        eng = self._engine()
        if eng:
            eng.calibrating = True  # nothing may be typed while you are being recorded
        self._begin_countdown()

    def stop(self) -> None:
        eng = self._engine()
        if eng:
            eng.stop_recording()
            eng.calibrating = False
        self.phase = "idle"

    def skip(self) -> None:
        if self.active:
            self._advance()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            eng = self._engine()
            if eng:
                eng.stop_recording()
        elif self.phase in ("countdown", "recording"):
            self._begin_countdown()

    # ---------------------------------------------------------------------- ticks
    def _begin_countdown(self) -> None:
        eng = self._engine()
        if eng:
            eng.stop_recording()
        self.phase = "countdown"
        self._phase_t = self._clock()

    def _advance(self) -> None:
        eng = self._engine()
        if eng:
            eng.stop_recording()
        self.index += 1
        if self.index >= len(self.letters):
            self.phase = "done"
            if eng:
                eng.calibrating = False
        else:
            self._begin_countdown()

    def tick(self) -> str:
        """Advance the state machine; returns a status string for the UI."""
        if not self.active:
            return ""
        eng = self._engine()
        if eng is None:
            return "Waiting for the camera..."
        now = self._clock()
        label = self.current
        if self.paused:
            return f"Paused at {label}"
        if self.phase == "countdown":
            left = COUNTDOWN - (now - self._phase_t)
            if left <= 0:
                eng.start_recording(label, RECORD_FRAMES)
                self.phase = "recording"
                self._phase_t = now
                return f"Recording {label}..."
            return f"Make the sign for {label}  -  starting in {int(left) + 1}"
        if self.phase == "recording":
            rec = eng.recording_state()
            if rec is None:
                self._begin_countdown()
                return "Restarting..."
            if rec["done"]:
                self.recorded.append(label)
                self.phase = "saved"
                self._phase_t = now
                eng.stop_recording()
                return f"{label} captured"
            elapsed = now - self._phase_t
            if elapsed > TIMEOUT and rec["count"] < 10:
                self._advance()
                return f"No hand seen for {label}, skipping"
            if elapsed > 2 * TIMEOUT:  # hand kept moving too fast: settle for what we have
                self.recorded.append(label)
                self._advance()
                return f"{label} captured ({rec['count']} frames)"
            return f"Recording {label}: {rec['count']}/{rec['target']}  (turn your hand slightly)"
        if self.phase == "saved":
            if now - self._phase_t > SAVED_PAUSE:
                self._advance()
            return f"{label} captured"
        return ""

    @property
    def fraction(self) -> float:
        return self.index / max(len(self.letters), 1)
