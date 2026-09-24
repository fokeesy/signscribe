"""Dark theme for ttk plus a few shared colours and fonts."""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

BG = "#0d1117"
PANEL = "#161b22"
PANEL2 = "#1f2630"
BORDER = "#30363d"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#3fb950"
ACCENT2 = "#58a6ff"
WARN = "#d29922"
DANGER = "#f85149"

#: Finger colours (RGB hex) matching the overlay's BGR values.
FINGER_HEX = {
    "thumb": "#ffaa3c",
    "index": "#50dc78",
    "middle": "#3cc8ff",
    "ring": "#b478ff",
    "pinky": "#ff6e6e",
}


def pick_family(root: tk.Misc) -> str:
    """Best available UI font for this platform."""
    have = set(tkfont.families(root))
    prefs = {
        "win32": ("Segoe UI Variable Text", "Segoe UI", "Arial"),
        "darwin": ("SF Pro Text", "Helvetica Neue", "Helvetica"),
    }.get(sys.platform, ("Inter", "Noto Sans", "DejaVu Sans", "Liberation Sans"))
    for name in prefs:
        if name in have:
            return name
    return "TkDefaultFont"


class Fonts:
    def __init__(self, root: tk.Misc) -> None:
        fam = pick_family(root)
        self.family = fam
        self.base = (fam, 10)
        self.small = (fam, 9)
        self.bold = (fam, 10, "bold")
        self.title = (fam, 15, "bold")
        self.h2 = (fam, 11, "bold")
        self.sentence = (fam, 21)
        self.huge = (fam, 74, "bold")
        self.big = (fam, 26, "bold")
        self.mono = ("Consolas" if sys.platform == "win32" else "Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono", 10)


def apply_theme(root: tk.Tk, fonts: Fonts) -> ttk.Style:
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=BG)
    style.configure(
        ".",
        background=BG,
        foreground=FG,
        fieldbackground=PANEL2,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        font=fonts.base,
    )
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=FG)
    style.configure("Card.TLabel", background=PANEL, foreground=FG)
    style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=fonts.small)
    style.configure("Heading.TLabel", background=PANEL, foreground=MUTED, font=(fonts.family, 9, "bold"))
    style.configure("Title.TLabel", background=BG, foreground=FG, font=fonts.title)
    style.configure("Status.TLabel", background=PANEL, foreground=MUTED, font=fonts.small)
    style.configure(
        "TButton", background=PANEL2, foreground=FG, borderwidth=1, focusthickness=0, padding=(12, 6), relief="flat", width=0
    )
    style.map(
        "TButton",
        background=[("active", "#2a3441"), ("pressed", "#2f3b4a"), ("disabled", PANEL)],
        foreground=[("disabled", MUTED)],
    )
    style.configure("Small.TButton", padding=(9, 4))
    style.configure("Accent.TButton", background=ACCENT, foreground="#04120a", font=fonts.bold)
    style.map("Accent.TButton", background=[("active", "#56d364"), ("pressed", "#2ea043"), ("disabled", PANEL)])
    style.configure("Danger.TButton", background="#3a1d1f", foreground=DANGER)
    style.map("Danger.TButton", background=[("active", "#4c2427")])
    style.configure(
        "TCheckbutton",
        background=PANEL,
        foreground=FG,
        focuscolor=PANEL,
        indicatorbackground=PANEL2,
        indicatorforeground="#04120a",
        upperbordercolor=MUTED,
        lowerbordercolor=MUTED,
    )
    style.map(
        "TCheckbutton",
        background=[("active", PANEL)],
        indicatorbackground=[("selected", ACCENT), ("pressed", PANEL2)],
    )
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(14, 8), borderwidth=0, font=fonts.bold)
    style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", FG)])
    style.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2, foreground=FG, arrowcolor=FG, bordercolor=BORDER)
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", PANEL2)],
        foreground=[("readonly", FG)],
        selectbackground=[("readonly", PANEL2)],
        selectforeground=[("readonly", FG)],
    )
    style.configure("TSpinbox", fieldbackground=PANEL2, background=PANEL2, foreground=FG, arrowcolor=FG)
    style.configure(
        "Horizontal.TScale", background=ACCENT2, troughcolor=PANEL2, bordercolor=BORDER, lightcolor=ACCENT2, darkcolor=ACCENT2
    )
    style.map("Horizontal.TScale", background=[("active", "#79b8ff")])
    style.configure(
        "Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL2, bordercolor=PANEL2, lightcolor=ACCENT, darkcolor=ACCENT
    )
    style.configure("Treeview", background=PANEL2, fieldbackground=PANEL2, foreground=FG, bordercolor=BORDER, rowheight=22)
    style.configure("Treeview.Heading", background=PANEL, foreground=MUTED, font=fonts.small)
    style.map("Treeview", background=[("selected", "#264f78")])
    style.configure("Vertical.TScrollbar", background=PANEL2, troughcolor=PANEL, arrowcolor=MUTED, bordercolor=PANEL)
    root.option_add("*TCombobox*Listbox.background", PANEL2)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", "#264f78")
    root.option_add("*Menu.background", PANEL)
    root.option_add("*Menu.foreground", FG)
    root.option_add("*Menu.activeBackground", "#264f78")
    root.option_add("*Menu.activeForeground", FG)
    return style
