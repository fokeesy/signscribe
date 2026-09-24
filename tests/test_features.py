import numpy as np
import pytest

from signscribe import features as F
from signscribe.synthetic import POSES, letter_observation, make_observation, resolved


def feats_of(letter, **kw):
    obs = letter_observation(letter, **kw)
    return F.extract(obs.world, obs.roll)


def test_vector_shape_and_names():
    obs = letter_observation("A")
    vec = F.to_vector(F.extract_batch(obs.world, obs.roll))
    assert vec.shape == (1, F.vector_dim())
    assert np.isfinite(vec).all()
    assert len(set(F.VECTOR_KEYS)) == len(F.VECTOR_KEYS)


def test_open_hand_vs_fist_extension():
    open_hand = feats_of("B")
    fist = feats_of("S")
    for s in "imrp":
        assert open_hand[f"ext_{s}"] > 0.85
        assert fist[f"ext_{s}"] < 0.15


@pytest.mark.parametrize("letter", ["A", "K", "V", "O", "Y"])
def test_invariant_to_scale_position_and_mirror(letter):
    base = feats_of(letter)
    moved = feats_of(letter, center=(0.2, 0.7), px_per_cm=20.0, mirror=True)
    for key in F.VECTOR_KEYS:
        if key in ("up", "side"):  # orientation features are allowed to depend on roll only
            continue
        assert moved[key] == pytest.approx(base[key], abs=1e-6), key


def test_in_plane_rotation_only_changes_orientation_features():
    pose = resolved("V")
    upright = make_observation(pose, roll=0.0)
    turned = make_observation(pose, roll=90.0)
    fu = F.extract(upright.world, upright.roll)
    ft = F.extract(turned.world, turned.roll)
    # (the model hand's knuckle axis is ~4 degrees off its own vertical, hence the tolerance)
    assert fu["up"] == pytest.approx(1.0, abs=0.01) and fu["side"] == pytest.approx(0.0, abs=0.08)
    assert ft["up"] == pytest.approx(0.0, abs=0.08) and ft["side"] == pytest.approx(1.0, abs=0.01)
    for key in ("ext_i", "ext_m", "spread_im", "t_pc", "d_ti"):
        assert ft[key] == pytest.approx(fu[key], abs=1e-6)


def test_roll_is_measured_from_image_pixels():
    # The wrist -> middle-knuckle axis of the model hand leans ~3.7 degrees toward the thumb.
    obs = letter_observation("B", roll=30.0)
    assert np.degrees(obs.roll) == pytest.approx(30.0 + 3.7, abs=1.0)
    obs = letter_observation("B", roll=-75.0)
    assert np.degrees(obs.roll) == pytest.approx(-75.0 + 3.7, abs=1.0)


def test_angle_deadzone_keeps_noisy_straight_fingers_straight():
    rng = np.random.default_rng(0)
    exts = []
    for _ in range(50):
        obs = letter_observation("B", noise=0.03, rng=rng)
        exts.append(F.extract(obs.world, obs.roll)["ext_i"])
    assert np.mean(exts) > 0.8


def test_batch_matches_single():
    obs = [letter_observation(n) for n in ("A", "B", "C")]
    world = np.stack([o.world for o in obs])
    roll = np.array([o.roll for o in obs])
    batch = F.extract_batch(world, roll)
    for i, o in enumerate(obs):
        single = F.extract(o.world, o.roll)
        for key in F.VECTOR_KEYS:
            assert batch[key][i] == pytest.approx(single[key])


def test_all_poses_produce_finite_features():
    for name in POSES:
        f = feats_of(name)
        assert all(np.isfinite(v) for v in f.values()), name
