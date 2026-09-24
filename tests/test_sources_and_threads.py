"""Worker thread lifecycle, frame sources, error paths and optional keyboard output."""

import sys
import time
import types

import cv2
import numpy as np
import pytest

from signscribe import cli
from signscribe.camera import CameraError, CameraStream, FramePacket, VideoFileSource
from signscribe.composer import TextEvent
from signscribe.config import Settings
from signscribe.demo import SimulatedSource
from signscribe.engine import Engine


def wait_for(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def collect(engine, seconds):
    out = ""
    end = time.time() + seconds
    while time.time() < end:
        try:
            ev = engine.events.get(timeout=0.1)
        except Exception:
            continue
        if ev.kind == "text":
            out += ev.text
        elif ev.kind == "backspace":
            out = out[: len(out) - ev.n]
    return out


def test_engine_thread_types_from_a_simulated_source_and_stops_cleanly():
    engine = Engine(Settings(), SimulatedSource("hi", loop=False), None)
    engine.start()
    try:
        assert wait_for(lambda: engine.latest() is not None, 5)
        text = collect(engine, 6.0)
    finally:
        engine.stop()
    assert text == "Hi "
    assert engine.error in (None, "Demo finished.")  # a non-looping demo may already have ended
    assert not engine._thread.is_alive()


def test_engine_reports_a_finished_source_as_an_error_message():
    engine = Engine(Settings(), SimulatedSource("a", loop=False, speed=40.0), None)
    engine.start()
    assert wait_for(lambda: engine.error is not None, 10)
    assert "finished" in engine.error.lower()
    engine.stop()


def test_camera_that_cannot_open_raises_a_helpful_error():
    with pytest.raises(CameraError, match="Could not open camera"):
        CameraStream(index=97).start()


def test_engine_surfaces_camera_failure_instead_of_crashing():
    engine = Engine(Settings(), CameraStream(index=97), None)
    engine.start()
    assert wait_for(lambda: engine.error is not None, 10)
    assert "camera" in engine.error.lower()
    engine.stop()


@pytest.fixture()
def video_file(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(45):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


def test_video_file_source_plays_and_loops(video_file):
    src = VideoFileSource(video_file, loop=True)
    src.start()
    try:
        frames = [src.read(1.0) for _ in range(60)]  # more than the clip length: must loop
    finally:
        src.stop()
    assert all(isinstance(f, FramePacket) and f.frame.shape == (240, 320, 3) for f in frames)
    with pytest.raises(CameraError):
        VideoFileSource("/definitely/not/here.mp4").start()


def test_engine_runs_real_tracking_on_a_video_without_hands(video_file):
    mp = pytest.importorskip("mediapipe")
    if not hasattr(mp, "solutions"):
        pytest.skip("needs mp.solutions")
    engine = Engine(Settings(), VideoFileSource(video_file), None)
    engine.start()
    try:
        assert wait_for(lambda: engine.latest() is not None, 20)
        res = engine.latest()
        assert res.frame is not None and res.hands == 0 and not res.present
        assert res.fps >= 0 and res.lighting is not None
    finally:
        engine.stop()
    assert engine.error is None


def test_system_typer_sends_text_and_backspaces(monkeypatch):
    calls = []

    class FakeKey:
        backspace = "<bs>"

    class FakeController:
        def type(self, s):
            calls.append(("type", s))

        def press(self, k):
            calls.append(("press", k))

        def release(self, k):
            calls.append(("release", k))

    keyboard = types.ModuleType("pynput.keyboard")
    keyboard.Controller, keyboard.Key = FakeController, FakeKey
    pynput = types.ModuleType("pynput")
    pynput.keyboard = keyboard
    monkeypatch.setitem(sys.modules, "pynput", pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)

    from signscribe.typing_out import SystemTyper

    assert SystemTyper.available()
    typer = SystemTyper()
    typer.send(TextEvent("text", "Hi "))
    typer.send(TextEvent("backspace", n=2))
    typer.send(TextEvent("clear"))
    assert calls == [("type", "Hi "), ("press", "<bs>"), ("release", "<bs>"), ("press", "<bs>"), ("release", "<bs>")]


def test_system_typer_reports_when_pynput_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynput", None)  # makes `import pynput` raise ImportError
    from signscribe.typing_out import SystemTyper

    assert SystemTyper.available() is False


def test_doctor_command_runs_without_touching_a_real_camera(monkeypatch, capsys):
    class FakeCam:
        def __init__(self, idx):
            self.idx = idx

        def start(self):
            if self.idx != 0:
                raise CameraError("nope")

        def read(self, timeout=0):
            return FramePacket(np.zeros((480, 640, 3), np.uint8), 0.0)

        def stop(self):
            pass

    monkeypatch.setattr("signscribe.camera.CameraStream", FakeCam)
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "mediapipe" in out and "camera 0: 640x480" in out and "camera 1: not available" in out
