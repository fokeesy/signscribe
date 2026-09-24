"""End-to-end: a virtual signer -> detection-free pipeline -> typed text, in virtual time."""

import numpy as np
import pytest

from signscribe.camera import FramePacket
from signscribe.config import Settings
from signscribe.demo import VirtualSigner, make_background
from signscribe.engine import Engine

BG = make_background()


def transcribe(text, settings=None, model=None, **signer_kw):
    eng = Engine(settings or Settings(), source=None, model=model)
    out = ""
    last = None
    for tau, obs in VirtualSigner(text, **signer_kw).frames(30.0):
        last = eng.process_packet(FramePacket(BG, tau, hands=[obs] if obs is not None else [], info={"demo": True}))
        while not eng.events.empty():
            e = eng.events.get()
            if e.kind == "text":
                out += e.text
            elif e.kind == "backspace":
                out = out[: len(out) - e.n]
    return out, last


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello world", "Hello world "),
        ("abc", "Abc "),
        ("i am ok", "I am ok "),
        ("jazz", "Jazz "),
        ("the quick brown fox jumps over the lazy dog", "The quick brown fox jumps over the lazy dog "),
        ("puzzle jazz", "Puzzle jazz "),
    ],
)
def test_fingerspelling_round_trip(text, expected):
    got, _ = transcribe(text)
    assert got == expected


def test_left_handed_signer():
    got, _ = transcribe("hello world jazz", mirror=True)
    assert got == "Hello world jazz "


def test_noisy_landmarks_still_type_correctly():
    got, _ = transcribe("hello world", noise=0.015)
    assert got == "Hello world "


def test_wave_types_hello_word_sign():
    got, _ = transcribe("~ my name is zed")
    assert got == "Hello my name is zed "


def test_double_letters_need_a_brief_release():
    got, _ = transcribe("bookkeeper")
    assert got == "Bookkeeper "


def test_upper_case_setting():
    got, _ = transcribe("hi", Settings(case_mode="upper"))
    assert got == "HI "


def test_paused_engine_types_nothing_but_still_reports_state():
    eng = Engine(Settings(), None, None)
    eng.paused = True
    label = None
    for tau, obs in VirtualSigner("ab").frames(30.0):
        res = eng.process_packet(FramePacket(BG, tau, hands=[obs] if obs is not None else []))
        label = res.prediction.label or label
    assert eng.events.empty()
    assert label in ("A", "B")


def test_frame_result_contents():
    eng = Engine(Settings(), None, None)
    res = None
    for tau, obs in VirtualSigner("w").frames(30.0):
        res = eng.process_packet(FramePacket(BG, tau, hands=[obs] if obs is not None else [], info={"demo": True}))
        if obs is not None and res.prediction.label == "W" and res.speed < 0.2:
            break
    assert res.present and res.frame.shape == (480, 640, 3)
    assert res.prediction.label == "W" and res.prediction.confidence > 0.8
    assert set(res.fingers) == {"thumb", "index", "middle", "ring", "pinky"}
    assert res.fingers["index"] > 0.8 and res.fingers["pinky"] < 0.3
    roll, pitch, yaw = res.rotation
    assert abs(roll) < 15 and abs(pitch) < 20 and abs(yaw) < 25
    assert res.mode == "rules" and res.demo


def test_recording_collects_samples_and_suspends_typing():
    eng = Engine(Settings(), None, None)
    eng.store.clear()
    eng.start_recording("W", target=25)
    for tau, obs in VirtualSigner("w").frames(30.0):
        eng.process_packet(FramePacket(BG, tau, hands=[obs] if obs is not None else []))
        if eng.recording_state() and eng.recording_state()["done"]:
            break
    assert eng.recording_state()["done"]
    assert eng.store.counts() == {"W": 25}
    assert eng.events.empty()  # nothing typed while recording
    eng.stop_recording()
    assert eng.recording_state() is None


def test_two_hands_dominant_selection():
    from signscribe.synthetic import letter_observation

    eng = Engine(Settings(dominant_hand="Left"), None, None)
    right = letter_observation("A", center=(0.3, 0.5), t=1.0)
    left = letter_observation("B", center=(0.7, 0.5), mirror=True, t=1.0)
    res = eng.process_packet(FramePacket(BG, 1.0, hands=[right, left]))
    assert res.handedness == "Left" and res.hands == 2
    eng2 = Engine(Settings(dominant_hand="Right"), None, None)
    res2 = eng2.process_packet(FramePacket(BG, 1.0, hands=[right, left]))
    assert res2.handedness == "Right"


def test_processing_is_fast_enough_for_realtime():
    import time

    eng = Engine(Settings(), None, None)
    frames = [(t, o) for t, o in VirtualSigner("abc").frames(30.0)][:90]
    t0 = time.perf_counter()
    for tau, obs in frames:
        eng.process_packet(FramePacket(BG, tau, hands=[obs] if obs is not None else []))
    per_frame = (time.perf_counter() - t0) / len(frames) * 1000
    assert per_frame < 12.0, f"{per_frame:.1f} ms per frame (excluding the camera and MediaPipe)"


def test_random_frames_never_crash_the_engine():
    rng = np.random.default_rng(0)
    eng = Engine(Settings(), None, None)
    from signscribe.synthetic import random_observation

    for k in range(60):
        hands = [random_observation("ABCDEFG"[k % 7], rng, noise=0.05)] if k % 3 else []
        eng.process_packet(FramePacket(BG, k / 30, hands=hands))
