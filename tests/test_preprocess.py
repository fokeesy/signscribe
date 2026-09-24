import numpy as np

from signscribe.preprocess import LightingAdapter


def frame(level, noise=6, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(level + rng.normal(0, noise, (240, 320, 3)), 0, 255).astype(np.uint8)


def test_off_mode_is_a_no_op():
    img = frame(20)
    assert LightingAdapter("off").process(img) is img


def test_well_lit_scene_is_left_alone_when_contrast_is_healthy():
    ramp = np.linspace(20, 235, 320, dtype=np.float32)  # smooth wide-range image, mid mean
    img = np.repeat(np.repeat(ramp[None, :, None], 240, axis=0), 3, axis=2).astype(np.uint8)
    ad = LightingAdapter("auto")
    out = ad.process(img)
    assert out is img and ad.info.state == "ok"


def test_dark_scene_is_brightened_toward_mid_grey():
    ad = LightingAdapter("auto")
    img = frame(22)
    out = img
    for _ in range(6):  # let the exposure estimate settle
        out = ad.process(img)
    assert ad.info.state == "dark" and ad.info.gamma < 1.0
    assert out.mean() > img.mean() * 2.0
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_bright_scene_is_toned_down():
    ad = LightingAdapter("auto")
    img = frame(238, noise=4)
    for _ in range(6):
        out = ad.process(img)
    assert ad.info.state == "bright" and ad.info.gamma > 1.0
    assert out.mean() < img.mean()


def test_meters_on_the_hand_region_when_given_one():
    """A dark hand on a bright backdrop should be judged by the hand, not the background."""
    img = frame(215, noise=3)
    img[100:200, 100:220] = 30  # dark hand
    roi = (100, 100, 220, 200)
    whole = LightingAdapter("auto")
    for _ in range(8):
        whole.process(img)
    hand = LightingAdapter("auto")
    for _ in range(8):
        hand.process(img, roi)
    assert hand.info.mean < whole.info.mean
    assert hand.info.gamma < whole.info.gamma  # boosts more because the hand is dark


def test_hysteresis_prevents_flicker_around_the_threshold():
    ad = LightingAdapter("auto")
    states = []
    for level in [70, 76, 74, 79, 82, 77, 80, 75, 79, 84, 78]:
        ad.process(frame(level, noise=20))
        states.append(ad.info.state)
    assert len(set(states[3:])) <= 2  # settles rather than toggling every frame


def test_force_enhance_keeps_shape_and_type():
    ad = LightingAdapter("auto")
    out = ad.force_enhance(frame(40))
    assert out.shape == (240, 320, 3) and out.dtype == np.uint8
