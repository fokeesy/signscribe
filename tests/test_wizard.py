"""The guided-calibration state machine, driven with a fake clock and a virtual signer."""

import numpy as np

from signscribe.camera import FramePacket
from signscribe.config import Settings
from signscribe.demo import make_background
from signscribe.engine import Engine
from signscribe.gui.wizard import COUNTDOWN, Calibrator
from signscribe.synthetic import letter_observation
from signscribe.training import train_model

BG = make_background()


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_wizard_records_every_letter_then_trains_a_working_model():
    engine = Engine(Settings(), None, None)
    engine.store.clear()
    clock = Clock()
    cal = Calibrator(lambda: engine, clock=clock)
    letters = ["A", "B", "L", "Y"]
    cal.start(letters)
    rng = np.random.default_rng(0)
    seen_status = set()

    for _step in range(4000):
        clock.t += 1 / 30
        status = cal.tick()
        seen_status.add(status.split(" ")[0])
        if cal.phase == "done":
            break
        letter = cal.current or "A"
        # a hand held reasonably still, with a little joint jitter and per-frame sensor noise
        obs = letter_observation(letter, noise=0.02, rng=rng, t=clock.t, yaw=rng.normal(0, 8), pitch=rng.normal(0, 8))
        engine.process_packet(FramePacket(BG, clock.t, hands=[obs]))
    else:
        raise AssertionError("wizard never finished")

    assert cal.recorded == letters
    assert set(engine.store.counts()) == set(letters)
    assert all(n >= 60 for n in engine.store.counts().values())
    assert {"Make", "Recording"} <= seen_status
    assert engine.events.empty()  # nothing was typed while calibrating

    model, report = train_model(engine.store, epochs=30)
    assert report["accuracy"] > 0.9


def test_wizard_countdown_pause_skip_and_stop():
    engine = Engine(Settings(), None, None)
    clock = Clock()
    cal = Calibrator(lambda: engine, clock=clock)
    cal.start(["A", "B", "C"])
    assert cal.phase == "countdown" and cal.current == "A"
    clock.t += COUNTDOWN / 2
    assert "starting in" in cal.tick()
    cal.toggle_pause()
    clock.t += 30
    assert cal.tick().startswith("Paused") and cal.phase == "countdown"
    cal.toggle_pause()
    cal.skip()
    assert cal.current == "B"
    cal.stop()
    assert not cal.active and engine.recording_state() is None


def test_wizard_skips_a_letter_when_no_hand_is_ever_seen():
    engine = Engine(Settings(), None, None)
    clock = Clock()
    cal = Calibrator(lambda: engine, clock=clock)
    cal.start(["A", "B"])
    for _ in range(2000):
        clock.t += 0.1
        cal.tick()
        engine.process_packet(FramePacket(BG, clock.t, hands=[]))
        if cal.current == "B" or cal.phase == "done":
            break
    assert cal.current == "B" or cal.phase == "done"
    assert cal.recorded == []
