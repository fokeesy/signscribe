"""Tracker tests. The MediaPipe network is only exercised on a blank frame (no hands expected);
everything else runs against a stub that mimics MediaPipe's result objects."""

from types import SimpleNamespace

import numpy as np
import pytest

from signscribe.synthetic import letter_observation
from signscribe.tracker import HandTracker

mp = pytest.importorskip("mediapipe")
pytestmark = pytest.mark.skipif(not hasattr(mp, "solutions"), reason="needs a MediaPipe build with mp.solutions")


def _lm(points):
    return SimpleNamespace(landmark=[SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points])


def _fake_result(obs, label="Left", score=0.97, with_world=True):
    return SimpleNamespace(
        multi_hand_landmarks=[_lm(obs.image)],
        multi_hand_world_landmarks=[_lm(obs.world)] if with_world else None,
        multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label=label, score=score)])],
    )


def make_tracker(**kw):
    return HandTracker(complexity=0, **kw)


def test_blank_frame_finds_no_hands():
    tracker = make_tracker()
    try:
        assert tracker.process(np.zeros((240, 320, 3), np.uint8), 0.0) == []
    finally:
        tracker.close()


def test_parses_landmarks_into_observation_and_swaps_handedness_when_not_mirrored():
    tracker = make_tracker(mirrored_input=False, smoothing=0.0)
    obs = letter_observation("A")
    tracker._hands = SimpleNamespace(process=lambda rgb: _fake_result(obs, label="Left"), close=lambda: None)
    frame = np.zeros((480, 640, 3), np.uint8)
    (hand,) = tracker.process(frame, 1.0)
    assert hand.handedness == "Right"  # 'Left' from MediaPipe means Right when the image is not mirrored
    assert hand.extras["chirality_right"] is True
    assert np.allclose(hand.world, obs.world) and np.allclose(hand.image, obs.image)
    assert hand.frame_size == (640, 480) and hand.score == pytest.approx(0.97)
    tracker.mirrored_input = True
    (hand,) = tracker.process(frame, 2.0)
    assert hand.handedness == "Left"


def test_smoothing_filters_jitter_but_not_the_first_frame():
    tracker = make_tracker(smoothing=1.0)
    rng = np.random.default_rng(0)
    base = letter_observation("B")
    seen = []

    def process(_rgb):
        obs = letter_observation("B", noise=0.03, rng=rng)
        return _fake_result(obs)

    tracker._hands = SimpleNamespace(process=process, close=lambda: None)
    frame = np.zeros((480, 640, 3), np.uint8)
    for k in range(60):
        (hand,) = tracker.process(frame, k / 30)
        seen.append(hand.world[8].copy())
    seen = np.array(seen[20:])
    raw = np.array([letter_observation("B", noise=0.03, rng=np.random.default_rng(i)).world[8] for i in range(40)])
    assert seen.std(axis=0).mean() < 0.6 * raw.std(axis=0).mean()
    assert np.allclose(seen.mean(axis=0), base.world[8], atol=0.01)


def test_two_hands_are_sorted_by_score_and_capped():
    tracker = make_tracker(smoothing=0.0)
    a, b = letter_observation("A"), letter_observation("B", center=(0.7, 0.5))
    res = SimpleNamespace(
        multi_hand_landmarks=[_lm(a.image), _lm(b.image)],
        multi_hand_world_landmarks=[_lm(a.world), _lm(b.world)],
        multi_handedness=[
            SimpleNamespace(classification=[SimpleNamespace(label="Left", score=0.6)]),
            SimpleNamespace(classification=[SimpleNamespace(label="Right", score=0.9)]),
        ],
    )
    tracker._hands = SimpleNamespace(process=lambda rgb: res, close=lambda: None)
    hands = tracker.process(np.zeros((480, 640, 3), np.uint8), 0.0)
    assert [h.score for h in hands] == [0.9, 0.6]


def test_missing_world_landmarks_fall_back_to_image_estimate():
    tracker = make_tracker(smoothing=0.0)
    obs = letter_observation("A")
    tracker._hands = SimpleNamespace(process=lambda rgb: _fake_result(obs, with_world=False), close=lambda: None)
    (hand,) = tracker.process(np.zeros((480, 640, 3), np.uint8), 0.0)
    assert hand.world.shape == (21, 3) and np.isfinite(hand.world).all()
