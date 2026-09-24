import numpy as np

from signscribe import dynamic as D
from signscribe import features as F
from signscribe.demo import VirtualSigner
from signscribe.synthetic import letter_observation


def j_path(n=40, mirror=False):
    stem = np.stack([np.zeros(28), np.linspace(0, 1.7, 28)], axis=1)
    th = np.radians(np.linspace(0, 160, 12))
    hook = np.stack([-0.5 + 0.5 * np.cos(th), 1.7 + 0.5 * np.sin(th)], axis=1)
    pts = np.vstack([stem, hook])
    if mirror:
        pts[:, 0] *= -1
    return pts


def z_path(mirror=False):
    pts = np.array([[0, 0], [1.3, 0], [0, 1.3], [1.3, 1.3]], dtype=float)
    dense = np.vstack([np.linspace(a, b, 15) for a, b in zip(pts[:-1], pts[1:])])
    if mirror:
        dense[:, 0] *= -1
    return dense


def test_j_recognised_both_hook_directions():
    assert D.match_j(j_path()) > 0.6
    assert D.match_j(j_path(mirror=True)) > 0.6


def test_j_rejects_plain_strokes():
    down = np.stack([np.zeros(40), np.linspace(0, 1.8, 40)], axis=1)
    up = down[::-1]
    across = np.stack([np.linspace(0, 1.8, 40), np.zeros(40)], axis=1)
    for stroke in (down, up, across, z_path()):
        assert D.match_j(stroke) == 0.0


def test_z_recognised_both_directions_and_rejects_others():
    assert D.match_z(z_path()) > 0.6
    assert D.match_z(z_path(mirror=True)) > 0.6
    down = np.stack([np.zeros(40), np.linspace(0, 1.8, 40)], axis=1)
    circle = np.stack([np.cos(np.linspace(0, 6.28, 60)), np.sin(np.linspace(0, 6.28, 60))], axis=1)
    for stroke in (down, circle, j_path()):
        assert D.match_z(stroke) == 0.0


def test_best_suffix_ignores_a_preceding_movement():
    approach = np.stack([np.linspace(-1.0, 0.0, 12), np.linspace(2.0, 0.0, 12)], axis=1)  # hand swoops in
    stroke = np.vstack([approach, j_path()])
    assert D.match_j(stroke) == 0.0  # whole thing is not a J...
    assert D.best_suffix(D.match_j, stroke) > 0.6  # ...but its tail is


def test_count_swings():
    t = np.linspace(0, 4 * np.pi, 120)
    assert D.count_swings(np.sin(t), 0.5) >= 3
    assert D.count_swings(np.linspace(0, 3, 50), 0.5) == 0
    assert D.count_swings(0.05 * np.sin(t), 0.5) == 0  # jitter below the amplitude threshold


def test_rdp_and_resample():
    line = np.stack([np.linspace(0, 1, 50), np.zeros(50)], axis=1)
    assert len(D.rdp(line, 0.01)) == 2
    r = D.resample(np.array([[0, 0], [1, 0], [1, 1]], float), 9)
    assert r.shape == (9, 2)
    assert np.allclose(r[0], [0, 0]) and np.allclose(r[-1], [1, 1])


def run_signer(text, **kw):
    signer = VirtualSigner(text, **kw)
    rec = D.MotionRecognizer()
    events = []
    for _tau, obs in signer.frames(30):
        if obs is None:
            ev = rec.lost(_tau)
        else:
            ev = rec.update(obs, F.extract(obs.world, obs.roll))
        if ev:
            events.append(ev.label)
    return events


def test_motion_recogniser_finds_j_and_z_from_a_signer():
    assert run_signer("j") == ["J"]
    assert run_signer("z") == ["Z"]


def test_motion_recogniser_works_for_left_hand():
    assert run_signer("jz", mirror=True) == ["J", "Z"]


def test_wave_means_hello_and_static_letters_trigger_nothing():
    assert run_signer("~") == ["Hello"]
    assert run_signer("hello world") == []


def test_nod_means_yes():
    rec = D.MotionRecognizer()
    events = []
    for k in range(60):
        t = k / 30
        y = 0.42 + 0.08 * np.sin(t * 2 * np.pi * 2.5)  # ~0.7 palm-length nods
        obs = letter_observation("S", center=(0.5, y), t=t)
        ev = rec.update(obs, F.extract(obs.world, obs.roll))
        if ev:
            events.append(ev.label)
    assert events == ["Yes"]


def test_hand_dynamics_speed():
    dyn = D.HandDynamics()
    speeds = []
    for k in range(30):
        obs = letter_observation("B", center=(0.3 + k * 0.01, 0.5), t=k / 30)
        speeds.append(dyn.update(obs))
    assert speeds[-1] > 1.0
    still = D.HandDynamics()
    for k in range(30):
        s = still.update(letter_observation("B", t=k / 30))
    assert s < 0.05
