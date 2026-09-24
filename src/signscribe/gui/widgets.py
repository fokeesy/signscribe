"""Small custom widgets: the big letter card, value bars and the reference-hand card."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from ..reference import TIPS, render_reference
from . import theme


class LetterCard(tk.Canvas):
    """Large readout of the current sign with a ring that fills as you hold it."""

    SIZE = 176

    def __init__(self, master, fonts: theme.Fonts) -> None:
        super().__init__(master, width=self.SIZE, height=self.SIZE, bg=theme.PANEL, highlightthickness=0)
        self.fonts = fonts
        pad = 10
        box = (pad, pad, self.SIZE - pad, self.SIZE - pad)
        self.create_oval(*box, outline=theme.PANEL2, width=9)
        self._arc = self.create_arc(*box, start=90, extent=0, style="arc", outline=theme.ACCENT, width=9)
        self._text = self.create_text(
            self.SIZE / 2, self.SIZE / 2 - 8, text="", fill=theme.MUTED, font=fonts.huge, width=self.SIZE - 46, justify="center"
        )
        self._sub = self.create_text(self.SIZE / 2, self.SIZE - 34, text="", fill=theme.MUTED, font=fonts.small)
        self._last: tuple = ()

    def show(self, label: str | None, progress: float, confident: bool, subtitle: str = "") -> None:
        key = (label, round(progress, 2), confident, subtitle)
        if key == self._last:
            return
        self._last = key
        text = label or "·"
        if label and len(label) > 1:
            font = (self.fonts.family, 22, "bold")
        else:
            font = self.fonts.huge
        colour = theme.FG if (label and confident) else theme.MUTED
        self.itemconfigure(self._text, text=text, fill=colour, font=font)
        self.itemconfigure(
            self._arc, extent=-max(min(progress, 1.0), 0.0) * 359.9, outline=theme.ACCENT if confident else theme.WARN
        )
        self.itemconfigure(self._sub, text=subtitle)


class BarMeter(tk.Canvas):
    """Thin labelled horizontal bar (0..1)."""

    def __init__(
        self, master, label: str, colour: str, fonts: theme.Fonts, width: int = 330, label_w: int = 62, value_w: int = 46
    ) -> None:
        super().__init__(master, width=width, height=22, bg=theme.PANEL, highlightthickness=0)
        self.colour, self.width, self.label_w, self.value_w = colour, width, label_w, value_w
        self._label = self.create_text(0, 11, text=label, anchor="w", fill=theme.MUTED, font=fonts.small)
        x0, x1 = label_w, width - value_w
        self._x0, self._x1 = x0, x1
        self.create_rectangle(x0, 8, x1, 15, fill=theme.PANEL2, outline="")
        self._bar = self.create_rectangle(x0, 8, x0, 15, fill=colour, outline="")
        self._val = self.create_text(width, 11, text="", anchor="e", fill=theme.FG, font=fonts.small)
        self._last: tuple = ()

    def set(self, value: float, text: str | None = None, label: str | None = None) -> None:
        v = max(0.0, min(1.0, value))
        key = (round(v, 2), text, label)
        if key == self._last:
            return
        self._last = key
        self.coords(self._bar, self._x0, 8, self._x0 + (self._x1 - self._x0) * v, 15)
        self.itemconfigure(self._val, text=text if text is not None else f"{int(v * 100)}%")
        if label is not None:
            self.itemconfigure(self._label, text=label)


class HintCard(ttk.Frame):
    """Reference skeleton for a sign, with a one-line tip."""

    SIZE = 116

    def __init__(self, master, fonts: theme.Fonts) -> None:
        super().__init__(master, style="Card.TFrame")
        self._img_label = tk.Label(self, bg=theme.PANEL, bd=0)
        self._img_label.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        self._title = ttk.Label(self, text="", style="Card.TLabel", font=fonts.h2)
        self._title.grid(row=0, column=1, sticky="sw")
        self._tip = ttk.Label(self, text="", style="Muted.TLabel", wraplength=190, justify="left")
        self._tip.grid(row=1, column=1, sticky="nw", pady=(2, 0))
        self.columnconfigure(1, weight=1)
        self._cache: dict[str, ImageTk.PhotoImage] = {}
        self._current: str | None = "<unset>"

    def show(self, name: str | None) -> None:
        if name == self._current:
            return
        self._current = name
        if not name or name not in TIPS:
            self._img_label.configure(image="")
            self._title.configure(text="How to sign it")
            self._tip.configure(text="Show a letter to the camera and its reference appears here.")
            return
        if name not in self._cache:
            bgr = render_reference(name, self.SIZE)
            self._cache[name] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        self._img_label.configure(image=self._cache[name])
        self._title.configure(text=name)
        self._tip.configure(text=TIPS[name])
