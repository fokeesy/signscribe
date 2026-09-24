from collections import defaultdict

import numpy as np
import pytest

from signscribe import features as F
from signscribe import rules
from signscribe.synthetic import POSES, letter_observation, random_observation

STATIC = [n for n in POSES if n not in ("OPEN", "J", "Z")]


def classify(obs):
    f = F.extract(obs.world, obs.roll)
    probs = rules.to_probabilities(rules.score_all(f))
    top = max(probs, key=probs.get)
    return top, probs[top]


def truth(name):
    return "I LOVE YOU" if name == "ILY" else name


@pytest.mark.parametrize("name", STATIC)
def test_canonical_pose_is_recognised(name):
    top, conf = classify(letter_observation(name))
    assert top == truth(name)
    assert conf > 0.5


def test_motion_letters_are_never_reported_statically():
    assert "J" not in rules.LETTER_RULES and "Z" not in rules.LETTER_RULES


def test_randomised_accuracy_left_and_right_hands():
    rng = np.random.default_rng(7)
    ok = defaultdict(int)
    n = 40
    for name in STATIC:
        for _ in range(n):
            top, conf = classify(random_observation(name, rng, noise=0.02))
            if top == truth(name) and conf >= 0.35:
                ok[name] += 1
    overall = sum(ok.values()) / (n * len(STATIC))
    assert overall > 0.93, f"overall {overall:.3f}: {dict(ok)}"
    # No single letter may collapse. The tightest pairs (R vs U, E vs M) still need > 60 %.
    worst = min(ok.values()) / n
    assert worst > 0.6, dict(ok)


def test_rotation_does_not_change_the_answer_for_orientation_free_letters():
    for name in ("A", "B", "D", "F", "L", "O", "W", "Y"):
        for roll in (-40, -15, 0, 20, 45):
            obs = letter_observation(name, roll=roll)
            top, _ = classify(obs)
            assert top == truth(name), (name, roll)


def test_orientation_separates_g_from_q_and_k_from_p():
    assert classify(letter_observation("G"))[0] == "G"
    assert classify(letter_observation("Q"))[0] == "Q"
    assert classify(letter_observation("K"))[0] == "K"
    assert classify(letter_observation("P"))[0] == "P"


def test_relaxed_open_palm_is_not_confidently_a_letter():
    _, conf = classify(letter_observation("OPEN"))
    assert conf < 0.5


def test_probabilities_are_bounded_and_ambiguity_lowers_confidence():
    f = F.extract(*(lambda o: (o.world, o.roll))(letter_observation("A")))
    probs = rules.to_probabilities(rules.score_all(f))
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert rules.to_probabilities({}) == {}
    assert max(rules.to_probabilities({"X": 0.0, "Y": 0.0}).values()) == 0.0
    sharp = max(rules.to_probabilities({"X": 0.9, "Y": 0.1}).values())
    tie = max(rules.to_probabilities({"X": 0.9, "Y": 0.9}).values())
    assert tie < sharp


def test_membership_helpers():
    assert rules.up(0.5, 0, 1) == 0.5 and rules.up(-1, 0, 1) == 0.0 and rules.up(2, 0, 1) == 1.0
    assert rules.dn(0.25, 0, 1) == 0.75
    assert rules.band(0.5, 0, 0.2, 0.8, 1.0) == 1.0 and rules.band(0.9, 0, 0.2, 0.8, 1.0) == pytest.approx(0.5)
