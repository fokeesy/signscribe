"""Optional: send recognised text to whatever application has keyboard focus.

Requires the optional ``pynput`` package (``pip install signscribe[typing]``). Off by default.
"""

from __future__ import annotations

from .composer import TextEvent


class SystemTyper:
    """Types :class:`TextEvent` s into the focused window using OS-level key events."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key  # imported lazily: optional dependency

        self._kb = Controller()
        self._key = Key

    @staticmethod
    def available() -> bool:
        try:
            import pynput  # noqa: F401
        except Exception:  # ImportError, or a display-less backend failure
            return False
        return True

    def send(self, event: TextEvent) -> None:
        if event.kind == "text" and event.text:
            self._kb.type(event.text)
        elif event.kind == "backspace":
            for _ in range(event.n):
                self._kb.press(self._key.backspace)
                self._kb.release(self._key.backspace)
