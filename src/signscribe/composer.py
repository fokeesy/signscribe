"""Turns a noisy stream of per-frame guesses into deliberate typed text.

The recogniser answers "what does this frame look like?" many times a second. Typing needs a
different question: "did the signer *mean* this letter?" The composer answers it with:

1. **Voting** - a short sliding window; the sign must win a clear majority, so single-frame
   glitches never show up.
2. **Hold to type** - the sign must stay stable for ``hold_time`` seconds. A progress value
   (0..1) is exposed so the UI can draw a ring filling up.
3. **Motion gate** - a hand that is moving fast is "in transit" and never types.
4. **Re-arming** - after typing a letter you must change sign, drop the hand, or keep holding
   for ``repeat_delay`` seconds (to type double letters such as the L L in "hello").
5. **Word breaks** - the hand leaving the frame for ``space_timeout`` seconds adds a space.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import Settings
from .dynamic import MotionEvent

#: Display text for word-level handshapes whose label is upper-case shorthand.
WORD_DISPLAY = {"I LOVE YOU": "I love you"}


@dataclass
class TextEvent:
    """A change to apply to the sentence buffer."""

    kind: str  # "text" (append) | "backspace" (delete n characters) | "clear"
    text: str = ""
    n: int = 0
    source: str = ""  # what produced it, e.g. "A", "space", "Hello"


class Composer:
    VOTE_WINDOW = 0.28  # seconds
    MIN_VOTES = 3
    MOTION_LOCK = 1.0  # seconds after a J/Z/word gesture during which static signs cannot type

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.text = ""
        self.reset_state()

    # ------------------------------------------------------------------- bookkeeping
    def reset_state(self) -> None:
        self._votes: deque[tuple[float, str | None, float]] = deque()
        self.stable_label: str | None = None
        self.stable_conf = 0.0
        self._stable_since = 0.0
        self.armed = True
        self._last_commit_label: str | None = None
        self._last_commit_t = -1e9
        self._absent_since: float | None = None
        self._space_pending = False
        self._lock_until = 0.0
        self._last_raw_label: str | None = None
        self._suppress: str | None = None
        self.progress = 0.0

    def set_text(self, text: str) -> None:
        """Sync with edits made by the user in the text box (affects auto-capitalisation)."""
        self.text = text

    # ---------------------------------------------------------------------- inputs
    def update(
        self,
        t: float,
        present: bool,
        label: str | None,
        confidence: float,
        speed: float,
    ) -> list[TextEvent]:
        """Feed one frame's result; returns any text changes it triggers."""
        s = self.settings
        events: list[TextEvent] = []

        if not present:
            self._votes.clear()
            self.stable_label = None
            self.progress = 0.0
            self.armed = True
            self._last_commit_label = None
            self._suppress = None
            self._last_raw_label = None
            if self._absent_since is None:
                self._absent_since = t
            if s.space_timeout > 0 and self._space_pending and t - self._absent_since >= s.space_timeout:
                events += self._space()
                self._space_pending = False
            return events
        self._absent_since = None
        if label is not None:
            self._last_raw_label = label

        usable = label if (label is not None and confidence >= s.min_confidence and speed <= s.speed_gate) else None
        self._votes.append((t, usable, confidence))
        while self._votes and t - self._votes[0][0] > self.VOTE_WINDOW:
            self._votes.popleft()

        winner, conf = self._tally()
        if winner != self.stable_label:
            self.stable_label = winner
            self.stable_conf = conf
            self._stable_since = t
            if winner is None or winner != self._last_commit_label:
                self.armed = True
            if winner != self._suppress:
                self._suppress = None
        else:
            self.stable_conf = conf

        if self.stable_label is None or t < self._lock_until or self.stable_label == self._suppress:
            self.progress = 0.0
            return events

        if self.armed:
            self.progress = min(1.0, (t - self._stable_since) / max(s.hold_time, 0.05))
            if self.progress >= 1.0:
                events += self._commit(self.stable_label, t)
        elif s.repeat_delay > 0 and self.stable_label == self._last_commit_label:
            self.progress = min(1.0, (t - self._last_commit_t) / s.repeat_delay)
            if self.progress >= 1.0:
                events += self._commit(self.stable_label, t)
        else:
            self.progress = 0.0
        return events

    def on_motion(self, ev: MotionEvent) -> list[TextEvent]:
        """A motion recogniser fired (J, Z, Hello, Yes...). Commit it immediately."""
        self._votes.clear()
        self.stable_label = None
        self.progress = 0.0
        self.armed = True
        self._last_commit_label = None
        self._suppress = self._last_raw_label  # the shape we are still holding must be released
        self._lock_until = ev.t + self.MOTION_LOCK
        return self._commit(ev.label, ev.t, motion=True)

    # ----------------------------------------------------------------------- internals
    def _tally(self) -> tuple[str | None, float]:
        total = len(self._votes)
        if total < self.MIN_VOTES:
            return None, 0.0
        weights: dict[str | None, int] = {}
        confs: dict[str | None, list[float]] = {}
        for _, lab, c in self._votes:
            weights[lab] = weights.get(lab, 0) + 1
            confs.setdefault(lab, []).append(c)
        winner = max(weights, key=lambda k: weights[k])
        if winner is None or weights[winner] / total < 0.6:
            return None, 0.0
        return winner, sum(confs[winner]) / len(confs[winner])

    def _commit(self, label: str, t: float, motion: bool = False) -> list[TextEvent]:
        self._space_pending = True
        if not motion:
            self.armed = False
            self._last_commit_label = label
            self._last_commit_t = t
            self._stable_since = t
        events = [TextEvent("text", self._letter_case(label), source=label)] if len(label) == 1 else self._word(label)
        for e in events:
            self._apply(e)
        return events

    def _at_sentence_start(self) -> bool:
        t = self.text
        if not t.strip():
            return True
        if t.endswith("\n"):
            return True
        return t.endswith(" ") and t.rstrip(" ").endswith((".", "!", "?"))

    def _letter_case(self, letter: str) -> str:
        mode = self.settings.case_mode
        if mode == "upper":
            return letter.upper()
        if mode == "lower":
            return letter.lower()
        return letter.upper() if self._at_sentence_start() else letter.lower()

    def _word(self, label: str) -> list[TextEvent]:
        display = WORD_DISPLAY.get(label, label)
        mode = self.settings.case_mode
        if mode == "smart" and label not in WORD_DISPLAY and display.isupper():
            display = display.capitalize()  # a custom label typed as "THANK YOU" reads "Thank you"
        if mode == "upper":
            body = display.upper()
        elif mode == "lower":
            body = display.lower()
        elif self._at_sentence_start():
            body = display[:1].upper() + display[1:]
        elif display == "I" or display.startswith("I "):
            body = display
        else:
            body = display[:1].lower() + display[1:]
        out: list[TextEvent] = []
        if self.text and not self.text.endswith((" ", "\n")):
            out.append(TextEvent("text", " ", source="space"))
        out.append(TextEvent("text", body + " ", source=label))
        return out

    def _space(self) -> list[TextEvent]:
        if not self.text or self.text.endswith((" ", "\n")):
            return []
        events: list[TextEvent] = []
        last_word = self.text.rsplit("\n", 1)[-1].rsplit(" ", 1)[-1]
        if last_word == "i" and self.settings.case_mode == "smart":  # the pronoun
            events.append(TextEvent("backspace", n=1, source="fix-I"))
            events.append(TextEvent("text", "I", source="fix-I"))
        events.append(TextEvent("text", " ", source="space"))
        for e in events:
            self._apply(e)
        return events

    def _apply(self, e: TextEvent) -> None:
        if e.kind == "text":
            self.text += e.text
        elif e.kind == "backspace":
            self.text = self.text[: max(0, len(self.text) - e.n)]
        elif e.kind == "clear":
            self.text = ""
