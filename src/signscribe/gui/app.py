"""Main window: live camera view, big sign readout, sentence box, training and settings."""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .. import __version__
from ..camera import CameraStream, VideoFileSource
from ..config import Settings, model_path
from ..demo import SimulatedSource
from ..engine import Engine, FrameResult
from ..reference import TIPS
from ..training import SampleStore, load_model, train_model
from ..typing_out import SystemTyper
from . import theme
from .widgets import BarMeter, HintCard, LetterCard
from .wizard import WIZARD_LETTERS, Calibrator

POLL_MS = 12


class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame (used for the Settings and Help tabs)."""

    def __init__(self, master) -> None:
        super().__init__(master, style="Card.TFrame")
        self.canvas = tk.Canvas(self, bg=theme.PANEL, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Card.TFrame")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", lambda e: self._bind_wheel(True))
            w.bind("<Leave>", lambda e: self._bind_wheel(False))

    def _bind_wheel(self, on: bool) -> None:
        if on:
            self.canvas.bind_all("<MouseWheel>", self._wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _wheel(self, e) -> None:
        self.canvas.yview_scroll(int(-e.delta / 120), "units")


def _card(parent, title: str, fonts: theme.Fonts, pady=(0, 10)) -> ttk.Frame:
    """A titled panel; returns the inner frame to put widgets in."""
    outer = ttk.Frame(parent, style="Card.TFrame", padding=12)
    outer.pack(fill="x", pady=pady)
    if title:
        ttk.Label(outer, text=title.upper(), style="Heading.TLabel").pack(anchor="w", pady=(0, 8))
    return outer


class SignScribeApp(tk.Tk):
    def __init__(
        self,
        settings: Settings,
        demo: bool = False,
        demo_text: str = "hello world",
        video: str | None = None,
        screenshot: str | None = None,
        screenshot_delay: float = 9.5,
        start_tab: str | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.fonts = theme.Fonts(self)
        theme.apply_theme(self, self.fonts)
        self.title("SignScribe - ASL translator")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1280, sw - 40)}x{min(860, sh - 90)}+{max((sw - 1280) // 2, 0)}+{max((sh - 860) // 3, 0)}")
        self.minsize(1080, 680)

        self.store = SampleStore.load()
        self.model = load_model()
        self.engine: Engine | None = None
        self.source_kind = "demo" if demo else "video" if video else "camera"
        self._video_path = video
        self._demo_text = demo_text
        self._last_t = -1.0
        self._photo: ImageTk.PhotoImage | None = None
        self._train_q: queue.Queue = queue.Queue()
        self._training = False
        self._screenshot = screenshot
        self.calibrator = Calibrator(lambda: self.engine)
        self.typer: SystemTyper | None = None

        self._build_menu()
        self._build_layout()
        self._refresh_model_status()
        self._refresh_sample_table()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind_all("<space>", self._on_space_key)
        self.bind_all("<Control-l>", lambda e: self._clear_text())
        self.bind_all("<F11>", lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen")))
        self.bind_all("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.bind_all("<Control-d>", lambda e: self._toggle_demo())

        if start_tab:
            for i, name in enumerate(("live", "train", "settings", "help")):
                if name == start_tab.lower():
                    self.tabs.select(i)
        self._start_engine()
        self.after(POLL_MS, self._tick)
        if screenshot:
            self.after(int(screenshot_delay * 1000), self._take_screenshot)

    # ================================================================== layout
    def _build_menu(self) -> None:
        bar = tk.Menu(self, bg=theme.PANEL, fg=theme.FG, bd=0)
        file_m = tk.Menu(bar, tearoff=0)
        file_m.add_command(label="Save transcript...", command=self._save_text)
        file_m.add_command(label="Copy text", command=self._copy_text)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=self._on_close)
        bar.add_cascade(label="File", menu=file_m)
        src_m = tk.Menu(bar, tearoff=0)
        src_m.add_command(label="Camera", command=lambda: self._switch_source("camera"))
        src_m.add_command(label="Demo (simulated hand)   Ctrl+D", command=lambda: self._switch_source("demo"))
        src_m.add_command(label="Video file...", command=self._pick_video)
        bar.add_cascade(label="Source", menu=src_m)
        help_m = tk.Menu(bar, tearoff=0)
        help_m.add_command(label="About SignScribe", command=self._about)
        bar.add_cascade(label="Help", menu=help_m)
        self.config(menu=bar)

    def _build_layout(self) -> None:
        f = self.fonts
        root = ttk.Frame(self, padding=(14, 10, 14, 0))
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(1, weight=1)

        # --- header
        head = ttk.Frame(root)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(head, text="SignScribe", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="  ASL fingerspelling, live", foreground=theme.MUTED).pack(side="left", pady=(4, 0))
        self.pill = tk.Label(head, text="Starting...", bg=theme.PANEL2, fg=theme.MUTED, padx=12, pady=3, font=f.small)
        self.pill.pack(side="right")
        self.demo_btn = ttk.Button(head, text="Demo", command=self._toggle_demo)
        self.demo_btn.pack(side="right", padx=8)

        # --- left column: video + sentence
        left = ttk.Frame(root)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.video_holder = tk.Frame(left, bg="#000000", highlightthickness=1, highlightbackground=theme.BORDER)
        self.video_holder.grid(row=0, column=0, sticky="nsew")
        self.video_holder.pack_propagate(False)
        self.video_label = tk.Label(self.video_holder, bg="#000000", bd=0)
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self._build_error_overlay()

        sentence = ttk.Frame(left, style="Card.TFrame", padding=12)
        sentence.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        top = ttk.Frame(sentence, style="Card.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="SENTENCE", style="Heading.TLabel").pack(side="left")
        self.count_label = ttk.Label(top, text="0 words", style="Muted.TLabel")
        self.count_label.pack(side="right")
        self.text = tk.Text(
            sentence,
            height=3,
            wrap="word",
            bg=theme.PANEL2,
            fg=theme.FG,
            insertbackground=theme.FG,
            font=f.sentence,
            relief="flat",
            padx=12,
            pady=8,
            undo=True,
            selectbackground="#264f78",
        )
        self.text.pack(fill="x", pady=(6, 8))
        self.text.bind("<KeyRelease>", lambda e: self._sync_text())
        self.text.bind("<<Paste>>", lambda e: self.after(10, self._sync_text))
        self.text.bind("<<Cut>>", lambda e: self.after(10, self._sync_text))
        btns = ttk.Frame(sentence, style="Card.TFrame")
        btns.pack(fill="x")
        for label, cmd in (("Space", lambda: self._insert(" ")), ("Backspace", self._backspace), ("Clear", self._clear_text)):
            ttk.Button(btns, text=label, style="Small.TButton", command=cmd).pack(side="left", padx=(0, 5))
        for ch in (".", ",", "?", "!", "'"):
            ttk.Button(btns, text=ch, style="Small.TButton", width=2, command=lambda c=ch: self._punct(c)).pack(
                side="left", padx=(0, 3)
            )
        self.pause_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btns, text="Pause typing", variable=self.pause_var, command=self._on_pause).pack(side="right")
        ttk.Button(btns, text="Save...", style="Small.TButton", command=self._save_text).pack(side="right", padx=6)
        ttk.Button(btns, text="Copy", style="Small.TButton", command=self._copy_text).pack(side="right", padx=(6, 0))
        self.hint_label = ttk.Label(sentence, text="", style="Muted.TLabel")
        self.hint_label.pack(anchor="w", pady=(8, 0))
        self._update_hint_label()

        # --- right column: tabs
        right = ttk.Frame(root, width=450)
        right.grid(row=1, column=1, sticky="ns")
        right.pack_propagate(False)
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        live = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        train = ScrollFrame(self.tabs)
        sett = ScrollFrame(self.tabs)
        helpf = ScrollFrame(self.tabs)
        self.tabs.add(live, text="Live")
        self.tabs.add(train, text="Train")
        self.tabs.add(sett, text="Settings")
        self.tabs.add(helpf, text="Help")
        self._build_live(live)
        self._build_train(train.inner)
        self._build_settings(sett.inner)
        self._build_help(helpf.inner)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._on_tab())

        # --- status bar
        bar = ttk.Frame(self, style="Card.TFrame", padding=(14, 5))
        bar.pack(fill="x", side="bottom")
        self.status_left = ttk.Label(bar, text="", style="Status.TLabel")
        self.status_left.pack(side="left")
        self.status_right = ttk.Label(bar, text="", style="Status.TLabel")
        self.status_right.pack(side="right")

    def _build_error_overlay(self) -> None:
        self.err_frame = tk.Frame(self.video_holder, bg=theme.PANEL, padx=28, pady=22)
        tk.Label(self.err_frame, text="Camera problem", bg=theme.PANEL, fg=theme.DANGER, font=self.fonts.h2).pack()
        self.err_text = tk.Label(
            self.err_frame, text="", bg=theme.PANEL, fg=theme.FG, wraplength=420, justify="center", font=self.fonts.base
        )
        self.err_text.pack(pady=10)
        row = tk.Frame(self.err_frame, bg=theme.PANEL)
        row.pack()
        ttk.Button(row, text="Retry", style="Accent.TButton", command=lambda: self._switch_source("camera")).pack(
            side="left", padx=4
        )
        ttk.Button(row, text="Try demo instead", command=lambda: self._switch_source("demo")).pack(side="left", padx=4)

    # -------------------------------------------------------------------- Live tab
    def _build_live(self, parent: ttk.Frame) -> None:
        f = self.fonts
        top = ttk.Frame(parent, style="Card.TFrame")
        top.pack(fill="x")
        self.card = LetterCard(top, f)
        self.card.pack(side="left")
        col = ttk.Frame(top, style="Card.TFrame")
        col.pack(side="left", fill="both", expand=True, padx=(14, 0))
        ttk.Label(col, text="CLOSEST MATCHES", style="Heading.TLabel").pack(anchor="w", pady=(6, 6))
        self.top_bars = [BarMeter(col, "-", theme.ACCENT2, f, width=196, label_w=52, value_w=38) for _ in range(3)]
        for b in self.top_bars:
            b.pack(anchor="w", pady=2)
        self.conf_label = ttk.Label(col, text="", style="Muted.TLabel", wraplength=190, justify="left")
        self.conf_label.pack(anchor="w", pady=(10, 0))

        ttk.Separator(parent).pack(fill="x", pady=(10, 8))
        ttk.Label(parent, text="FINGERS  (extension)", style="Heading.TLabel").pack(anchor="w", pady=(0, 4))
        self.finger_bars = {}
        for name in ("thumb", "index", "middle", "ring", "pinky"):
            b = BarMeter(parent, name.capitalize(), theme.FINGER_HEX[name], f, width=410, label_w=64)
            b.pack(anchor="w")
            self.finger_bars[name] = b

        ttk.Separator(parent).pack(fill="x", pady=(8, 8))
        ttk.Label(parent, text="HAND POSE", style="Heading.TLabel").pack(anchor="w", pady=(0, 4))
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(fill="x")
        self.pose_vars = {}
        for i, key in enumerate(("Roll", "Pitch", "Yaw", "Speed", "Hand", "Lighting")):
            cell = ttk.Frame(grid, style="Card.TFrame")
            cell.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 22), pady=(0, 4))
            ttk.Label(cell, text=key.upper(), style="Heading.TLabel").pack(anchor="w")
            var = tk.StringVar(value="-")
            ttk.Label(cell, textvariable=var, style="Card.TLabel", font=f.bold).pack(anchor="w")
            self.pose_vars[key] = var
        self.light_hint = ttk.Label(parent, text="", style="Muted.TLabel")
        self.light_hint.pack(anchor="w")

        ttk.Separator(parent).pack(fill="x", pady=(8, 8))
        self.hint = HintCard(parent, f)
        self.hint.pack(fill="x")
        self.hint.show(None)

    # ------------------------------------------------------------------- Train tab
    def _build_train(self, parent: ttk.Frame) -> None:
        f = self.fonts
        pad = ttk.Frame(parent, style="Card.TFrame", padding=12)
        pad.pack(fill="both", expand=True)

        c = _card(pad, "Guided calibration", f)
        ttk.Label(
            c,
            style="Muted.TLabel",
            wraplength=390,
            justify="left",
            text=(
                "The built-in rules work out of the box, but recording your own hand makes recognition "
                "noticeably more accurate, especially for M, N, T, S, E and R. It takes about 3 minutes: "
                "you will be shown each letter; hold it while it records, turning your hand slightly."
            ),
        ).pack(anchor="w")
        self.wiz_start = ttk.Button(c, text="Start guided calibration", style="Accent.TButton", command=self._wizard_start)
        self.wiz_start.pack(anchor="w", pady=(10, 6))
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 4))
        self.wiz_pause = ttk.Button(row, text="Pause", style="Small.TButton", command=self._wizard_pause, state="disabled")
        self.wiz_pause.pack(side="left")
        self.wiz_skip = ttk.Button(
            row, text="Skip letter", style="Small.TButton", command=lambda: self.calibrator.skip(), state="disabled"
        )
        self.wiz_skip.pack(side="left", padx=6)
        self.wiz_stop = ttk.Button(row, text="Stop", style="Small.TButton", command=self._wizard_stop, state="disabled")
        self.wiz_stop.pack(side="left")
        self.wiz_status = ttk.Label(c, text="Not running.", style="Card.TLabel", font=f.bold, wraplength=390)
        self.wiz_status.pack(anchor="w", pady=(6, 4))
        self.wiz_bar = ttk.Progressbar(c, maximum=1.0)
        self.wiz_bar.pack(fill="x")
        self.wiz_hint = HintCard(c, f)  # shown only while the wizard is running

        c = _card(pad, "Record one sign", f)
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x")
        self.manual_label = ttk.Combobox(row, values=[*WIZARD_LETTERS, "J", "Z"], width=16)
        self.manual_label.set("A")
        self.manual_label.pack(side="left")
        self.rec_btn = ttk.Button(row, text="Record 70 frames", command=self._manual_record)
        self.rec_btn.pack(side="left", padx=8)
        self.rec_status = ttk.Label(c, text="", style="Muted.TLabel", wraplength=390)
        self.rec_status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            c,
            style="Muted.TLabel",
            wraplength=390,
            justify="left",
            text="You can also type your own label (for example THANK YOU) to teach a custom hand shape; it will be typed as a word.",
        ).pack(anchor="w", pady=(6, 0))

        c = _card(pad, "Your samples", f)
        self.sample_tree = ttk.Treeview(c, columns=("n",), height=7, selectmode="browse")
        self.sample_tree.heading("#0", text="Sign")
        self.sample_tree.heading("n", text="Samples")
        self.sample_tree.column("#0", width=200)
        self.sample_tree.column("n", width=90, anchor="e")
        self.sample_tree.pack(fill="x")
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Delete selected", command=self._delete_label).pack(side="left")
        ttk.Button(row, text="Delete all", style="Danger.TButton", command=self._delete_all_samples).pack(side="left", padx=6)

        c = _card(pad, "Model", f, pady=(0, 0))
        self.model_status = ttk.Label(c, text="", style="Card.TLabel", wraplength=390, justify="left")
        self.model_status.pack(anchor="w")
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x", pady=(10, 0))
        self.train_btn = ttk.Button(row, text="Train model", style="Accent.TButton", command=self._train)
        self.train_btn.pack(side="left")
        ttk.Button(row, text="Remove model", command=self._remove_model).pack(side="left", padx=6)
        self.train_bar = ttk.Progressbar(c, maximum=1.0)
        self.train_bar.pack(fill="x", pady=(10, 0))
        self.train_report = ttk.Label(c, text="", style="Muted.TLabel", wraplength=390, justify="left")
        self.train_report.pack(anchor="w", pady=(6, 0))

    # ----------------------------------------------------------------- Settings tab
    def _build_settings(self, parent: ttk.Frame) -> None:
        f = self.fonts
        s = self.settings
        pad = ttk.Frame(parent, style="Card.TFrame", padding=12)
        pad.pack(fill="both", expand=True)

        def combo(card, label, attr, values, apply=None):
            """Drop-down over (stored value, display name) pairs bound to ``Settings.attr``."""
            row = ttk.Frame(card, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, style="Card.TLabel").pack(side="left")
            by_name = {name: value for value, name in values}
            var = tk.StringVar(value=next((n for v, n in values if v == getattr(s, attr)), values[0][1]))
            cb = ttk.Combobox(row, textvariable=var, values=[n for _, n in values], state="readonly", width=22)

            def changed(_e=None):
                setattr(s, attr, by_name[var.get()])
                s.save()
                if apply:
                    apply()

            cb.bind("<<ComboboxSelected>>", changed)
            cb.pack(side="right")

        def slider(card, label, attr, lo, hi, fmt, apply=None):
            row = ttk.Frame(card, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            head = ttk.Frame(row, style="Card.TFrame")
            head.pack(fill="x")
            ttk.Label(head, text=label, style="Card.TLabel").pack(side="left")
            val = ttk.Label(head, text=fmt.format(getattr(s, attr)), style="Muted.TLabel")
            val.pack(side="right")
            var = tk.DoubleVar(value=getattr(s, attr))

            def changed(v):
                x = float(v)
                setattr(s, attr, x)
                val.configure(text=fmt.format(x))
                s.save()
                if apply:
                    apply()

            ttk.Scale(row, from_=lo, to=hi, variable=var, command=changed).pack(fill="x")

        def check(card, label, attr, apply=None):
            var = tk.BooleanVar(value=getattr(s, attr))

            def changed():
                setattr(s, attr, var.get())
                s.save()
                if apply:
                    apply()

            ttk.Checkbutton(card, text=label, variable=var, command=changed).pack(anchor="w", pady=2)

        c = _card(pad, "Typing", f)
        slider(c, "Hold time to type a sign", "hold_time", 0.2, 1.5, "{:.2f} s", self._update_hint_label)
        slider(c, "Minimum confidence", "min_confidence", 0.3, 0.95, "{:.0%}")
        slider(c, "Hold to repeat a letter", "repeat_delay", 0.0, 2.5, "{:.1f} s (0 = off)", self._update_hint_label)
        slider(c, "Hand out of view = space after", "space_timeout", 0.0, 3.0, "{:.1f} s (0 = off)", self._update_hint_label)
        combo(c, "Letter case", "case_mode", [("smart", "Smart (sentence case)"), ("upper", "UPPERCASE"), ("lower", "lowercase")])
        check(c, "Recognise word signs (wave, nod, I love you)", "word_gestures")

        c = _card(pad, "Recognition", f)
        combo(
            c,
            "Engine",
            "recognition_mode",
            [("auto", "Auto (hybrid if trained)"), ("rules", "Rules only"), ("model", "Trained model only")],
        )
        combo(c, "Dominant hand", "dominant_hand", [("Auto", "Auto"), ("Right", "Right"), ("Left", "Left")])
        slider(c, "Motion gate (moving hand never types)", "speed_gate", 0.5, 4.0, "{:.1f} palm/s")

        c = _card(pad, "Camera & tracking", f)
        row = ttk.Frame(c, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Camera number", style="Card.TLabel").pack(side="left")
        self.cam_var = tk.IntVar(value=s.camera_index)
        ttk.Spinbox(row, from_=0, to=8, width=4, textvariable=self.cam_var, command=self._camera_index_changed).pack(side="right")
        combo(c, "Tracking model", "model_complexity", [(0, "Fast"), (1, "Accurate")], self._restart_engine_soon)
        combo(c, "Lighting correction", "lighting", [("auto", "Auto"), ("always", "Always on"), ("off", "Off")])
        slider(c, "Smoothing (steadier <-> snappier)", "smoothing", 0.0, 1.0, "{:.2f}")
        check(c, "Mirror the picture (selfie view)", "mirror", self._restart_engine_soon)

        c = _card(pad, "Display", f)
        check(c, "Show finger extension numbers", "show_labels")
        check(c, "Show joint angles", "show_angles")
        check(c, "Show fingertip trail", "show_trail")

        c = _card(pad, "Output", f, pady=(0, 0))
        self.type_var = tk.BooleanVar(value=False)
        self.type_check = ttk.Checkbutton(
            c, text="Also type into other applications", variable=self.type_var, command=self._toggle_typer
        )
        self.type_check.pack(anchor="w")
        avail = SystemTyper.available()
        ttk.Label(
            c,
            style="Muted.TLabel",
            wraplength=390,
            justify="left",
            text=(
                "Sends every typed letter to whichever app has keyboard focus (click into it first)."
                if avail
                else "Requires the optional 'pynput' package:  pip install \"signscribe[typing]\""
            ),
        ).pack(anchor="w", pady=(4, 0))
        if not avail:
            self.type_check.state(["disabled"])

    # -------------------------------------------------------------------- Help tab
    def _build_help(self, parent: ttk.Frame) -> None:
        self._help_parent = parent
        self._help_built = False
        pad = ttk.Frame(parent, style="Card.TFrame", padding=12)
        pad.pack(fill="both", expand=True)
        self._help_pad = pad
        ttk.Label(
            pad,
            style="Card.TLabel",
            wraplength=400,
            justify="left",
            text=(
                "1. Sit where the camera sees your hand clearly, with reasonable light.\n"
                "2. Show a letter with your palm toward the camera. The big letter shows what is recognised.\n"
                "3. Hold it still until the ring fills; the letter is typed into the sentence box.\n"
                "4. Lower your hand out of view for a moment to add a space.\n"
                "5. J and Z: make the I / pointing shape, then draw the letter in the air.\n"
                "6. Word signs: wave an open hand = Hello, nod a fist = Yes, thumb+index+little finger = I love you.\n\n"
                "Shortcuts:  Space = add a space,  Ctrl+L = clear,  Ctrl+D = demo,  F11 = full screen.\n\n"
                "Below are schematic reference skeletons for every sign."
            ),
        ).pack(anchor="w")

    def _build_help_tiles(self) -> None:
        if self._help_built:
            return
        self._help_built = True
        from ..reference import render_reference

        self._help_imgs = []
        grid = ttk.Frame(self._help_pad, style="Card.TFrame")
        grid.pack(fill="x", pady=(12, 0))
        names = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "I LOVE YOU"]
        for i, name in enumerate(names):
            cell = ttk.Frame(grid, style="Card.TFrame")
            cell.grid(row=i // 3, column=i % 3, padx=4, pady=6, sticky="n")
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(render_reference(name, 118), cv2.COLOR_BGR2RGB)))
            self._help_imgs.append(img)
            tk.Label(cell, image=img, bg=theme.PANEL, bd=0).pack()
            ttk.Label(cell, text=name, style="Card.TLabel", font=self.fonts.bold).pack()
            ttk.Label(cell, text=TIPS.get(name, ""), style="Muted.TLabel", wraplength=124, justify="center").pack()

    def _on_tab(self) -> None:
        if self.tabs.tab(self.tabs.select(), "text") == "Help":
            self._build_help_tiles()

    # ============================================================ engine / sources
    def _make_source(self):
        s = self.settings
        if self.source_kind == "demo":
            return SimulatedSource(self._demo_text)
        if self.source_kind == "video" and self._video_path:
            return VideoFileSource(self._video_path)
        return CameraStream(s.camera_index, s.width, s.height, s.fps)

    def _start_engine(self) -> None:
        if self.engine is not None:
            self._drain_events()
            self.engine.stop()
        self.engine = Engine(self.settings, self._make_source(), self.model, self.store)
        self.engine.composer.set_text(self.text.get("1.0", "end-1c"))
        self.engine.paused = self.pause_var.get()
        self.err_frame.place_forget()
        self.engine.start()
        self.demo_btn.configure(text="Stop demo" if self.source_kind == "demo" else "Demo")

    def _switch_source(self, kind: str) -> None:
        self.calibrator.stop()
        self.source_kind = kind
        self._start_engine()

    def _toggle_demo(self) -> None:
        self._switch_source("camera" if self.source_kind == "demo" else "demo")

    def _pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
        )
        if path:
            self._video_path = path
            self._switch_source("video")

    def _restart_engine_soon(self) -> None:
        self.after(50, self._start_engine)

    def _camera_index_changed(self) -> None:
        try:
            self.settings.camera_index = int(self.cam_var.get())
        except (tk.TclError, ValueError):
            return
        self.settings.save()
        if self.source_kind == "camera":
            self._restart_engine_soon()

    # ====================================================================== tick
    def _tick(self) -> None:
        try:
            self._update()
        except Exception:  # never let a UI glitch kill the poll loop
            import traceback

            traceback.print_exc()
        self.after(POLL_MS, self._tick)

    def _update(self) -> None:
        eng = self.engine
        if eng is None:
            return
        self._drain_events()
        if eng.error:
            self.err_text.configure(text=eng.error)
            self.err_frame.place(relx=0.5, rely=0.5, anchor="center")
            self._set_pill("Error", theme.DANGER)
        res = eng.latest()
        if res is not None and res.t != self._last_t:
            self._last_t = res.t
            self._show_frame(res.frame)
            self._update_live(res)
        self._tick_wizard()
        self._tick_training()

    def _show_frame(self, bgr) -> None:
        w, h = self.video_holder.winfo_width(), self.video_holder.winfo_height()
        if w < 60 or h < 60 or bgr is None:
            return
        fh, fw = bgr.shape[:2]
        scale = min((w - 2) / fw, (h - 2) / fh)
        if abs(scale - 1.0) > 0.02:
            bgr = cv2.resize(
                bgr, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            )
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self._photo is not None and self._photo.width() == img.width and self._photo.height() == img.height:
            self._photo.paste(img)
        else:
            self._photo = ImageTk.PhotoImage(img)
            self.video_label.configure(image=self._photo)

    def _set_pill(self, text: str, colour: str) -> None:
        self.pill.configure(text=text, fg=colour)

    def _update_live(self, r: FrameResult) -> None:
        s = self.settings
        p = r.prediction
        confident = bool(p.label) and p.confidence >= s.min_confidence
        if r.present:
            shown = r.stable_label or p.label
            self.card.show(shown, r.progress, confident, f"{int(p.confidence * 100)}%  {r.mode}" if p.label else "no match")
        else:
            self.card.show(None, 0.0, False, "no hand")

        for bar, (lab, val) in zip(self.top_bars, (p.top + [("", 0.0)] * 3)[:3]):
            bar.set(val, label=(lab if len(lab) < 8 else lab[:7] + "."))
        if not r.present:
            self.conf_label.configure(text="Show a hand to start.")
        elif confident:
            self.conf_label.configure(text="Hold still to type it." if r.progress < 1 else "Typed.")
        else:
            self.conf_label.configure(text=f"Confidence below {int(s.min_confidence * 100)}% - adjust your hand.")
        for name, bar in self.finger_bars.items():
            bar.set(r.fingers.get(name, 0.0))

        if r.rotation is not None:
            self.pose_vars["Roll"].set(f"{r.rotation[0]:+.0f}°")
            self.pose_vars["Pitch"].set(f"{r.rotation[1]:+.0f}°")
            self.pose_vars["Yaw"].set(f"{r.rotation[2]:+.0f}°")
            self.pose_vars["Speed"].set(f"{r.speed:.1f} p/s")
            self.pose_vars["Hand"].set(f"{r.handedness} ({int(r.hand_score * 100)}%)")
        else:
            for k in ("Roll", "Pitch", "Yaw", "Speed", "Hand"):
                self.pose_vars[k].set("-")
        self.pose_vars["Lighting"].set(
            {"ok": "OK", "dark": "Dim", "bright": "Bright", "flat": "Flat"}.get(r.lighting.state, "OK")
        )
        self.light_hint.configure(text=r.lighting.describe() if r.lighting.state != "ok" else "")

        self.hint.show(p.label if (r.present and p.label in TIPS) else None)

        if r.present:
            self._set_pill("Demo" if r.demo else "Tracking", theme.ACCENT)
        elif not self.engine.error:
            self._set_pill("Demo - no hand" if r.demo else "Looking for a hand", theme.WARN)
        n_hands = f"{r.hands} hand{'s' if r.hands != 1 else ''}"
        self.status_left.configure(text=f"{r.source_name}   |   {n_hands}   |   engine: {r.mode}")
        self.status_right.configure(text=f"{r.fps:.0f} FPS   |   {r.latency_ms:.0f} ms latency")

    # ================================================================ text handling
    def _drain_events(self) -> None:
        eng = self.engine
        if eng is None:
            return
        changed = False
        while True:
            try:
                ev = eng.events.get_nowait()
            except queue.Empty:
                break
            if ev.kind == "text":
                self.text.insert("end-1c", ev.text)
            elif ev.kind == "backspace":
                self.text.delete(f"end-1c-{ev.n}c", "end-1c")
            elif ev.kind == "clear":
                self.text.delete("1.0", "end")
            if self.typer is not None and self.focus_displayof() is None:
                try:
                    self.typer.send(ev)
                except Exception:
                    self.typer = None
                    self.type_var.set(False)
            changed = True
        if changed:
            self.text.see("end")
            self._update_count()

    def _sync_text(self) -> None:
        if self.engine:
            self.engine.composer.set_text(self.text.get("1.0", "end-1c"))
        self._update_count()

    def _update_count(self) -> None:
        words = len(self.text.get("1.0", "end-1c").split())
        self.count_label.configure(text=f"{words} word{'s' if words != 1 else ''}")

    def _insert(self, s: str) -> None:
        self.text.insert("end-1c", s)
        self.text.see("end")
        self._sync_text()

    def _punct(self, ch: str) -> None:
        cur = self.text.get("1.0", "end-1c")
        if cur.endswith(" "):  # attach punctuation to the previous word
            self.text.delete("end-2c", "end-1c")
        self.text.insert("end-1c", ch + (" " if ch in ".?!," else ""))
        self.text.see("end")
        self._sync_text()

    def _backspace(self) -> None:
        self.text.delete("end-2c", "end-1c")
        self._sync_text()

    def _clear_text(self) -> None:
        self.text.delete("1.0", "end")
        self._sync_text()

    def _on_space_key(self, e) -> None:
        if isinstance(e.widget, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox)):
            return
        self._insert(" ")

    def _copy_text(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.text.get("1.0", "end-1c").strip())

    def _save_text(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")], initialfile="transcript.txt")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.text.get("1.0", "end-1c").strip() + "\n")

    def _on_pause(self) -> None:
        if self.engine:
            self.engine.paused = self.pause_var.get()
            self.engine.composer.reset_state()

    def _update_hint_label(self) -> None:
        s = self.settings
        parts = [f"Hold a sign for {s.hold_time:.2f} s to type it."]
        if s.space_timeout > 0:
            parts.append(f"Lower your hand for {s.space_timeout:.1f} s for a space.")
        if s.repeat_delay > 0:
            parts.append(f"Keep holding {s.repeat_delay:.1f} s for a double letter.")
        self.hint_label.configure(text="  ".join(parts))

    def _toggle_typer(self) -> None:
        if self.type_var.get():
            try:
                self.typer = SystemTyper()
            except Exception as exc:
                messagebox.showerror("Typing unavailable", f"Could not start keyboard output:\n{exc}")
                self.type_var.set(False)
        else:
            self.typer = None

    # ================================================================== training
    def _refresh_sample_table(self) -> None:
        for row in self.sample_tree.get_children():
            self.sample_tree.delete(row)
        for lab, n in sorted(self.store.counts().items()):
            self.sample_tree.insert("", "end", iid=lab, text=lab, values=(n,))

    def _refresh_model_status(self) -> None:
        m = self.model
        if m is None:
            self.model_status.configure(text="No personal model yet. Using the built-in geometry rules.")
        else:
            acc = m.meta.get("accuracy")
            acc_txt = f", accuracy {acc:.0%}" if isinstance(acc, float) else ""
            self.model_status.configure(
                text=f"Personal model active: {len(m.labels)} signs, {m.meta.get('samples', '?')} samples{acc_txt}. Mode: hybrid (model + rules)."
            )

    def _wizard_start(self) -> None:
        if self.source_kind == "demo":
            self._switch_source("camera")
        self.pause_var.set(False)
        self.calibrator.start()
        self._wizard_buttons(True)

    def _wizard_stop(self) -> None:
        self.calibrator.stop()
        self._wizard_buttons(False)
        self.wiz_status.configure(text="Stopped.")
        self._save_samples()

    def _wizard_pause(self) -> None:
        self.calibrator.toggle_pause()
        self.wiz_pause.configure(text="Resume" if self.calibrator.paused else "Pause")

    def _wizard_buttons(self, running: bool) -> None:
        self.wiz_start.configure(state="disabled" if running else "normal")
        for b in (self.wiz_pause, self.wiz_skip, self.wiz_stop):
            b.configure(state="normal" if running else "disabled")
        if running:
            self.wiz_hint.pack(fill="x", pady=(10, 0))
        else:
            self.wiz_hint.pack_forget()
            self.wiz_pause.configure(text="Pause")

    def _tick_wizard(self) -> None:
        cal = self.calibrator
        if cal.active:
            self.wiz_status.configure(text=cal.tick())
            self.wiz_bar.configure(value=cal.fraction)
            self.wiz_hint.show(cal.current)
        elif cal.phase == "done":
            cal.phase = "idle"
            self._wizard_buttons(False)
            self.wiz_bar.configure(value=1.0)
            self.wiz_status.configure(text=f"Done: recorded {len(cal.recorded)} signs. Now train the model.")
            self._save_samples()
            if messagebox.askyesno("Calibration complete", "All signs recorded. Train your personal model now?"):
                self._train()

    def _manual_record(self) -> None:
        label = self.manual_label.get().strip()
        if not label or self.engine is None:
            return
        if len(label) == 1:
            label = label.upper()
        self.engine.start_recording(label, 70)
        self.rec_status.configure(text=f"Recording '{label}' - hold the sign and turn your hand a little...")
        self._manual_label_active = label
        self._watch_manual()

    def _watch_manual(self) -> None:
        eng = self.engine
        rec = eng.recording_state() if eng else None
        if rec is None:
            return
        if rec["done"]:
            eng.stop_recording()
            self.rec_status.configure(text=f"Recorded {rec['count']} frames of '{rec['label']}'.")
            self._save_samples()
            return
        self.rec_status.configure(
            text=f"Recording '{rec['label']}': {rec['count']}/{rec['target']}"
            + ("" if rec["count"] else "  (waiting for a hand)")
        )
        self.after(120, self._watch_manual)

    def _save_samples(self) -> None:
        self.store.save()
        self._refresh_sample_table()

    def _delete_label(self) -> None:
        sel = self.sample_tree.selection()
        if sel:
            self.store.remove_label(sel[0])
            self._save_samples()

    def _delete_all_samples(self) -> None:
        if messagebox.askyesno("Delete all samples", "Delete every recorded sample?"):
            self.store.clear()
            self._save_samples()

    def _train(self) -> None:
        if self._training:
            return
        if len(self.store) == 0:
            messagebox.showinfo("Nothing to train on", "Record some signs first (guided calibration or 'Record one sign').")
            return
        self._training = True
        self.train_btn.configure(state="disabled")
        self.train_report.configure(text="Training...")
        self.train_bar.configure(value=0.0)

        def work() -> None:
            try:
                model, report = train_model(
                    self.store,
                    epochs=80,
                    progress=lambda e, loss, acc: self._train_q.put(("progress", e + 1, 80, acc)),
                )
                model.save(model_path())
                self._train_q.put(("done", model, report))
            except Exception as exc:
                self._train_q.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _tick_training(self) -> None:
        while True:
            try:
                msg = self._train_q.get_nowait()
            except queue.Empty:
                return
            if msg[0] == "progress":
                self.train_bar.configure(value=msg[1] / msg[2])
                self.train_report.configure(text=f"Training... epoch {msg[1]}  accuracy {msg[3]:.0%}")
            elif msg[0] == "done":
                _, model, report = msg
                self.model = model
                if self.engine:
                    self.engine.set_model(model)
                self._training = False
                self.train_btn.configure(state="normal")
                self.train_bar.configure(value=1.0)
                kind = "held-out" if report["held_out"] else "training"
                worst = sorted(report["per_label"].items(), key=lambda kv: kv[1])[:4]
                weak = ", ".join(f"{k} {v:.0%}" for k, v in worst if v < 0.98)
                self.train_report.configure(
                    text=f"Done. {kind} accuracy {report['accuracy']:.1%} over {len(report['labels'])} signs."
                    + (f" Weakest: {weak}. Record more of those." if weak else "")
                )
                self._refresh_model_status()
            elif msg[0] == "error":
                self._training = False
                self.train_btn.configure(state="normal")
                self.train_report.configure(text=f"Training failed: {msg[1]}")

    def _remove_model(self) -> None:
        with contextlib.suppress(OSError):
            model_path().unlink(missing_ok=True)
        self.model = None
        if self.engine:
            self.engine.set_model(None)
        self._refresh_model_status()

    # ==================================================================== misc
    def _about(self) -> None:
        messagebox.showinfo(
            "About SignScribe",
            f"SignScribe {__version__}\n\nReal-time ASL fingerspelling translator.\n"
            "Hand tracking: MediaPipe Hands (21 landmarks).\nRecognition: geometry rules + optional personal model.\n\n"
            "Recognises the ASL manual alphabet (J and Z by motion) and a few word signs. "
            "It is not a full sign-language translator.",
        )

    def _take_screenshot(self) -> None:
        from PIL import ImageGrab

        self.update_idletasks()
        self.attributes("-topmost", True)
        self.update()
        time.sleep(0.4)
        x, y, w, h = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(self._screenshot)
        print(f"screenshot saved to {self._screenshot}")
        self._on_close()

    def _on_close(self) -> None:
        self.calibrator.stop()
        self.settings.save()
        if self.engine:
            self.engine.stop()
        self.destroy()


def run_gui(
    settings: Settings,
    demo: bool = False,
    demo_text: str = "hello world",
    video: str | None = None,
    screenshot: str | None = None,
    screenshot_delay: float = 9.5,
    start_tab: str | None = None,
) -> int:
    if sys.platform == "win32":
        try:  # crisp text on high-DPI screens
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    app = None
    for attempt in range(2):  # Tk on Windows occasionally fails once at start-up; a retry fixes it
        try:
            app = SignScribeApp(
                settings,
                demo=demo,
                demo_text=demo_text,
                video=video,
                screenshot=screenshot,
                screenshot_delay=screenshot_delay,
                start_tab=start_tab,
            )
            break
        except tk.TclError as exc:
            if attempt == 1:
                print(
                    f"Could not start the window (Tk error: {exc}).\n"
                    "Is a display available? On Linux install python3-tk. Try `signscribe headless` for a text-only mode.",
                    file=sys.stderr,
                )
                return 1
            time.sleep(0.5)
    app.mainloop()
    return 0
