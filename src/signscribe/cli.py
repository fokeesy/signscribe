"""Command-line entry point: ``signscribe`` / ``python -m signscribe``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="signscribe",
        description="Real-time ASL fingerspelling translator with hand tracking.",
    )
    p.add_argument("--version", action="version", version=f"signscribe {__version__}")
    sub = p.add_subparsers(dest="command")

    def add_source_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--camera", type=int, default=None, help="camera index (default: from settings, usually 0)")
        sp.add_argument("--video", type=str, default=None, help="play a video file instead of using the camera")
        sp.add_argument("--demo", action="store_true", help="use a simulated signing hand (no camera needed)")
        sp.add_argument("--text", type=str, default="hello world", help="text the demo hand fingerspells")

    g = sub.add_parser("gui", help="open the translator window (default)")
    add_source_args(g)
    g.add_argument("--screenshot", type=str, default=None, help="save a screenshot of the window and exit (dev)")
    g.add_argument("--screenshot-delay", type=float, default=9.5, help="seconds to wait before the screenshot (dev)")
    g.add_argument("--tab", type=str, default=None, choices=["live", "train", "settings", "help"], help="tab to open first")
    g.add_argument("--reset-settings", action="store_true", help="ignore saved settings for this run")

    h = sub.add_parser("headless", help="print recognised text to the terminal (no window)")
    add_source_args(h)
    h.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = until Ctrl+C / video ends)")

    t = sub.add_parser("train", help="train the personal model")
    t.add_argument("--images", type=str, default=None, help="folder of labelled images (root/A/*.jpg, root/B/*.jpg ...)")
    t.add_argument("--max-per-class", type=int, default=None, help="limit images per letter when using --images")
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--out", type=str, default=None, help="model output path (default: your SignScribe data folder)")

    r = sub.add_parser("reference", help="render the reference skeleton chart to a PNG")
    r.add_argument("--out", type=str, default="alphabet_reference.png")

    sub.add_parser("doctor", help="check the installation and camera")
    return p


def _make_source(args, settings):
    from .camera import CameraStream, VideoFileSource
    from .demo import SimulatedSource

    if getattr(args, "demo", False):
        return SimulatedSource(args.text)
    if getattr(args, "video", None):
        return VideoFileSource(args.video)
    index = args.camera if getattr(args, "camera", None) is not None else settings.camera_index
    return CameraStream(index, settings.width, settings.height, settings.fps)


def _cmd_gui(args) -> int:
    from .config import Settings
    from .gui.app import run_gui

    settings = Settings() if args.reset_settings else Settings.load()
    if args.camera is not None:
        settings.camera_index = args.camera
    return run_gui(
        settings,
        demo=args.demo,
        demo_text=args.text,
        video=args.video,
        screenshot=args.screenshot,
        screenshot_delay=args.screenshot_delay,
        start_tab=args.tab,
    )


def _cmd_headless(args) -> int:
    from .config import Settings
    from .engine import Engine
    from .training import load_model

    settings = Settings.load()
    engine = Engine(settings, _make_source(args, settings), load_model())
    engine.start()
    start = time.time()
    print("SignScribe headless - press Ctrl+C to stop. Recognised text streams below.", file=sys.stderr)
    try:
        while engine.error is None:
            try:
                ev = engine.events.get(timeout=0.2)
                if ev.kind == "text":
                    print(ev.text, end="", flush=True)
                elif ev.kind == "backspace":
                    print("\b" * ev.n, end="", flush=True)
            except Exception:
                pass
            if args.seconds and time.time() - start > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    print()
    if engine.error:
        print(f"error: {engine.error}", file=sys.stderr)
        return 1
    return 0


def _cmd_train(args) -> int:
    from .config import model_path
    from .training import SampleStore, dataset_from_images, train_model

    out = Path(args.out) if args.out else model_path()
    if args.images:
        from .tracker import HandTracker

        tracker = HandTracker(max_hands=1, complexity=1, min_detection_conf=0.4, smoothing=0.0, static=True)

        def progress(ci, n, label, i, total):
            print(f"\r[{ci + 1}/{n}] {label}: {i}/{total}   ", end="", flush=True)

        store = dataset_from_images(Path(args.images), tracker, args.max_per_class, progress)
        tracker.close()
        print(f"\nextracted {len(store)} hand samples from {len(store.counts())} classes")
    else:
        store = SampleStore.load()
        print(f"loaded {len(store)} recorded samples: {store.counts()}")
    if len(store) == 0:
        print("no samples to train on. Record some in the GUI (Train tab) or pass --images.", file=sys.stderr)
        return 1
    model, report = train_model(
        store,
        epochs=args.epochs,
        progress=lambda e, loss, acc: print(f"\repoch {e + 1:3d} loss {loss:.3f} acc {acc:.3f}", end=""),
    )
    print()
    model.save(out)
    kind = "held-out" if report["held_out"] else "training"
    print(f"saved {out}  ({kind} accuracy {report['accuracy']:.1%}, {len(report['labels'])} classes)")
    worst = sorted(report["per_label"].items(), key=lambda kv: kv[1])[:5]
    print("weakest:", ", ".join(f"{k} {v:.0%}" for k, v in worst))
    return 0


def _cmd_reference(args) -> int:
    import cv2

    from .reference import contact_sheet

    cv2.imwrite(args.out, contact_sheet())
    print(f"wrote {args.out}")
    return 0


def _cmd_doctor(_args) -> int:
    import platform

    print(f"SignScribe {__version__}  Python {platform.python_version()}  {platform.platform()}")
    ok = True
    for mod in ("numpy", "cv2", "mediapipe", "PIL", "tkinter"):
        try:
            m = __import__(mod)
            print(f"  ok   {mod:10s} {getattr(m, '__version__', '')}")
        except Exception as exc:
            ok = False
            print(f"  FAIL {mod:10s} {exc}")
    try:
        import mediapipe as mp

        print(f"  {'ok  ' if hasattr(mp, 'solutions') else 'FAIL'} mediapipe.solutions.hands available")
    except Exception:
        pass
    from .typing_out import SystemTyper

    print(
        f"  {'ok  ' if SystemTyper.available() else 'info'} pynput (optional: type into other apps): {'installed' if SystemTyper.available() else 'not installed'}"
    )
    from .camera import CameraStream

    for idx in range(3):
        cam = CameraStream(idx)
        try:
            cam.start()
            frame = cam.read(2.0)
            print(
                f"  ok   camera {idx}: {frame.frame.shape[1]}x{frame.frame.shape[0]}"
                if frame
                else f"  warn camera {idx}: opened but no frames"
            )
        except Exception:
            print(f"  --   camera {idx}: not available")
        finally:
            cam.stop()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"gui", "headless", "train", "reference", "doctor", "-h", "--help", "--version"}:
        argv = ["gui", *argv]  # bare `signscribe` and `signscribe --demo` open the window
    args = parser.parse_args(argv)
    command = args.command
    handlers = {
        "gui": _cmd_gui,
        "headless": _cmd_headless,
        "train": _cmd_train,
        "reference": _cmd_reference,
        "doctor": _cmd_doctor,
    }
    return handlers[command](args)


if __name__ == "__main__":
    raise SystemExit(main())
