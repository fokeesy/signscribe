import numpy as np

from signscribe.filters import OneEuroFilter, ScalarEMA


def run(filt, signal, dt=1 / 30):
    return np.array([filt(np.array([v]), i * dt)[0] for i, v in enumerate(signal)])


def test_reduces_jitter_at_rest():
    rng = np.random.default_rng(0)
    noisy = 1.0 + rng.normal(0, 0.01, 300)
    out = run(OneEuroFilter(min_cutoff=1.5, beta=5.0), noisy)
    assert out[30:].std() < 0.5 * noisy[30:].std()


def test_follows_fast_motion_with_little_lag():
    t = np.arange(120) / 30
    ramp = np.clip((t - 1.0) * 2.0, 0, 1.0)  # a fast 0 -> 1 move
    out = run(OneEuroFilter(min_cutoff=1.5, beta=30.0), ramp)
    settled = np.argmax(out > 0.95)
    reference = np.argmax(ramp > 0.95)
    assert settled - reference <= 4  # within ~130 ms


def test_resets_after_a_gap():
    f = OneEuroFilter(reset_gap=0.4)
    f(np.array([0.0]), 0.0)
    f(np.array([0.0]), 0.03)
    jump = f(np.array([5.0]), 2.0)  # hand reappeared elsewhere: no smoothing across the gap
    assert jump[0] == 5.0


def test_handles_arrays_and_non_monotonic_time():
    f = OneEuroFilter()
    a = f(np.zeros((21, 3)), 1.0)
    b = f(np.ones((21, 3)), 1.0)  # same timestamp: treated as a restart, must not divide by zero
    assert a.shape == b.shape == (21, 3)
    assert np.isfinite(b).all()


def test_scalar_ema():
    ema = ScalarEMA(0.5)
    assert ema(10.0) == 10.0
    assert ema(0.0) == 5.0
    ema.reset()
    assert ema(3.0) == 3.0
