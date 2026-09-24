"""Zero-training ASL fingerspelling classifier based on hand geometry.

Each letter is described the way a signing guide describes it: "index straight, other
fingers closed, thumb out at right angles" and so on. The description is turned into fuzzy
memberships over the invariant features from :mod:`signscribe.features`:

* ``crit`` - conditions that must all hold (finger states). The weakest one gates the score.
* ``soft`` - supporting evidence with weights (thumb placement, spread, orientation...).

The result is a score in 0..1 for every letter. J and Z are *motion* letters, so this static
classifier never reports them; see :mod:`signscribe.dynamic`.

The rules are a good starting point for clearly distinct shapes. Letters that differ only in
where the thumb tucks (A, S, T, N, M, E) are inherently the hardest to separate from landmarks
alone; recording a few samples of your own hand with the built-in trainer fixes that.
"""

from __future__ import annotations

import math
from collections.abc import Callable

Feat = dict[str, float]
Rule = tuple[list[float], list[tuple[float, float]]]  # (critical memberships, (membership, weight))

_FLOOR = 0.04


def up(x: float, a: float, b: float) -> float:
    """0 at or below ``a``, 1 at or above ``b``, linear between."""
    if b == a:
        return 1.0 if x >= b else 0.0
    return min(1.0, max(0.0, (x - a) / (b - a)))


def dn(x: float, a: float, b: float) -> float:
    """1 at or below ``a``, 0 at or above ``b``, linear between."""
    return 1.0 - up(x, a, b)


def band(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid: rises a->b, flat b..c, falls c->d."""
    return min(up(x, a, b), dn(x, c, d))


def straight(e: float) -> float:
    return up(e, 0.60, 0.85)


def closed(e: float) -> float:
    return dn(e, 0.12, 0.38)


def _spread(f: Feat) -> float:
    return max(f["spread_im"], f["spread_mr"], f["spread_rp"])


def _fist4(f: Feat) -> list[float]:
    return [closed(f["ext_i"]), closed(f["ext_m"]), closed(f["ext_r"]), closed(f["ext_p"])]


def _thumb_tip_max(f: Feat) -> float:
    return max(f["d_ti"], f["d_tm"], f["d_tr"], f["d_tp"])


def _rule_a(f: Feat) -> Rule:
    return _fist4(f), [(dn(f["t_along"], -0.02, 0.20), 3.0), (up(f["t_height"], 0.85, 1.05), 1.5)]


def _rule_s(f: Feat) -> Rule:
    return _fist4(f), [
        (band(f["t_along"], 0.15, 0.30, 0.60, 0.75), 2.0),
        (up(f["t_pc"], 0.40, 0.55), 2.0),
        (up(f["t_pip_i"], 0.22, 0.32), 1.0),
    ]


def _rule_t(f: Feat) -> Rule:
    return _fist4(f), [
        (band(f["t_along"], -0.02, 0.08, 0.28, 0.40), 2.5),
        (dn(f["t_pip_i"], 0.24, 0.50), 2.0),
        (up(f["t_pc"], 0.40, 0.55), 1.0),
    ]


def _rule_n(f: Feat) -> Rule:
    return _fist4(f), [
        (band(f["t_along"], 0.32, 0.42, 0.56, 0.66), 2.5),
        (dn(f["t_pc"], 0.38, 0.55), 2.0),
    ]


def _rule_m(f: Feat) -> Rule:
    return _fist4(f), [(up(f["t_along"], 0.62, 0.78), 2.5), (dn(f["t_pc"], 0.42, 0.60), 1.5)]


def _rule_e(f: Feat) -> Rule:
    mean_ext = (f["ext_i"] + f["ext_m"] + f["ext_r"] + f["ext_p"]) / 4.0
    return [band(mean_ext, 0.04, 0.10, 0.50, 0.68)], [
        (dn(_thumb_tip_max(f), 0.40, 0.75), 2.0),
        (dn(f["t_height"], 0.70, 0.90), 1.5),
        (up(f["t_along"], 0.45, 0.65), 1.0),
        (dn(f["t_pc"], 0.4, 0.7), 1.0),
    ]


def _rule_b(f: Feat) -> Rule:
    return [straight(f["ext_i"]), straight(f["ext_m"]), straight(f["ext_r"]), straight(f["ext_p"])], [
        (dn(_spread(f), 10.0, 26.0), 2.0),
        (dn(f["t_pc"], 0.55, 0.95), 2.0),
    ]


def _rule_c(f: Feat) -> Rule:
    crit = [band(f[k], 0.25, 0.38, 0.75, 0.88) for k in ("ext_i", "ext_m", "ext_r", "ext_p")]
    return crit, [
        (band(f["d_ti"], 0.30, 0.50, 1.30, 1.70), 2.0),
        (dn(_spread(f), 15.0, 35.0), 1.0),
        (up(f["side"], 0.15, 0.55), 1.0),
    ]


def _rule_d(f: Feat) -> Rule:
    return [straight(f["ext_i"]), dn(f["ext_m"], 0.45, 0.75), dn(f["ext_r"], 0.50, 0.80), dn(f["ext_p"], 0.50, 0.80)], [
        (dn(f["d_tm"], 0.25, 0.60), 3.0),
        (dn(f["d_tr"], 0.40, 0.90), 1.0),
    ]


def _rule_f(f: Feat) -> Rule:
    return [
        straight(f["ext_m"]),
        straight(f["ext_r"]),
        straight(f["ext_p"]),
        band(f["ext_i"], 0.15, 0.30, 0.75, 0.90),
    ], [(dn(f["d_ti"], 0.20, 0.50), 3.0)]


def _g_like(f: Feat) -> list[float]:
    return [straight(f["ext_i"]), closed(f["ext_m"]), closed(f["ext_r"]), closed(f["ext_p"])]


def _rule_g(f: Feat) -> Rule:
    return _g_like(f), [
        (dn(f["ang_ti"], 30.0, 60.0), 2.5),
        (up(f["th_reach"], 0.70, 0.90), 2.0),
        (up(f["side"], 0.30, 0.70), 2.0),
        (dn(f["t_pc"], 1.3, 1.8), 0.5),
    ]


def _rule_q(f: Feat) -> Rule:
    return _g_like(f), [
        (dn(f["ang_ti"], 30.0, 60.0), 2.5),
        (up(f["th_reach"], 0.70, 0.90), 2.0),
        (up(-f["up"], 0.30, 0.70), 2.5),
    ]


def _two_up(f: Feat) -> list[float]:
    return [straight(f["ext_i"]), straight(f["ext_m"]), closed(f["ext_r"]), closed(f["ext_p"])]


def _rule_h(f: Feat) -> Rule:
    return _two_up(f), [
        (dn(f["spread_im"], 10.0, 22.0), 2.0),
        (up(f["side"], 0.35, 0.70), 3.0),
        (dn(f["cross_im"], -0.12, 0.02), 1.0),
        (up(f["t_pip_m"], 0.40, 0.75), 1.0),
    ]


def _rule_u(f: Feat) -> Rule:
    return _two_up(f), [
        (dn(f["spread_im"], 10.0, 22.0), 2.0),
        (up(f["up"], 0.0, 0.5), 3.0),
        (dn(f["cross_im"], -0.12, 0.02), 1.5),
        (dn(f["t_pc"], 0.85, 1.25), 1.0),
        (up(f["t_pip_m"], 0.40, 0.75), 1.5),
    ]


def _rule_v(f: Feat) -> Rule:
    return _two_up(f), [
        (up(f["spread_im"], 10.0, 22.0), 3.0),
        (up(f["t_pip_m"], 0.50, 0.80), 1.5),
        (dn(f["cross_im"], -0.10, 0.05), 1.0),
    ]


def _rule_k(f: Feat) -> Rule:
    return _two_up(f), [
        (up(f["spread_im"], 6.0, 16.0), 2.0),
        (dn(f["t_pip_m"], 0.45, 0.85), 3.0),
        (up(f["up"], -0.1, 0.4), 1.5),
    ]


def _rule_p(f: Feat) -> Rule:
    return _two_up(f), [
        (up(f["spread_im"], 6.0, 16.0), 2.0),
        (dn(f["t_pip_m"], 0.45, 0.85), 3.0),
        (up(-f["up"], 0.2, 0.6), 2.5),
    ]


def _rule_r(f: Feat) -> Rule:
    return _two_up(f), [
        (up(f["cross_im"], -0.10, 0.03), 3.5),
        (dn(f["spread_im"], 10.0, 24.0), 1.0),
        (up(f["up"], 0.0, 0.5), 1.5),
        (up(f["t_pip_m"], 0.40, 0.75), 1.0),
    ]


def _rule_w(f: Feat) -> Rule:
    return [straight(f["ext_i"]), straight(f["ext_m"]), straight(f["ext_r"]), closed(f["ext_p"])], [
        (up(min(f["spread_im"], f["spread_mr"]), 4.0, 13.0), 2.5),
    ]


def _rule_x(f: Feat) -> Rule:
    return [band(f["ext_i"], 0.20, 0.30, 0.65, 0.78), closed(f["ext_m"]), closed(f["ext_r"]), closed(f["ext_p"])], [
        (up(f["pip_i"], 40.0, 65.0), 3.0),
        (dn(f["mcp_i"], 45.0, 75.0), 1.5),
    ]


def _pinky_only(f: Feat) -> list[float]:
    return [straight(f["ext_p"]), closed(f["ext_i"]), closed(f["ext_m"]), closed(f["ext_r"])]


def _rule_i(f: Feat) -> Rule:
    return _pinky_only(f), [(dn(f["t_pc"], 0.85, 1.20), 3.0)]


def _rule_y(f: Feat) -> Rule:
    return _pinky_only(f), [(up(f["t_pc"], 0.90, 1.25), 3.0), (up(f["th_reach"], 0.70, 0.90), 1.5)]


def _rule_ily(f: Feat) -> Rule:
    return [straight(f["ext_i"]), straight(f["ext_p"]), closed(f["ext_m"]), closed(f["ext_r"])], [
        (up(f["t_pc"], 0.90, 1.25), 2.5),
        (up(f["th_reach"], 0.70, 0.90), 1.0),
    ]


def _rule_l(f: Feat) -> Rule:
    return _g_like(f), [
        (up(f["t_pc"], 0.90, 1.25), 2.5),
        (band(f["ang_ti"], 45.0, 65.0, 115.0, 140.0), 2.5),
        (up(f["th_reach"], 0.70, 0.90), 1.0),
    ]


def _rule_o(f: Feat) -> Rule:
    mean_ext = (f["ext_i"] + f["ext_m"] + f["ext_r"] + f["ext_p"]) / 4.0
    return [dn(f["d_ti"], 0.25, 0.50), dn(f["d_tm"], 0.30, 0.60)], [
        (band(mean_ext, 0.08, 0.18, 0.62, 0.80), 2.0),
        (dn(f["d_tr"], 0.50, 1.00), 1.0),
        (dn(f["d_tp"], 0.70, 1.20), 1.0),
    ]


#: Static letters (J and Z are motion letters). Order = tie-break order.
LETTER_RULES: dict[str, Callable[[Feat], Rule]] = {
    "A": _rule_a, "B": _rule_b, "C": _rule_c, "D": _rule_d, "E": _rule_e, "F": _rule_f,
    "G": _rule_g, "H": _rule_h, "I": _rule_i, "K": _rule_k, "L": _rule_l, "M": _rule_m,
    "N": _rule_n, "O": _rule_o, "P": _rule_p, "Q": _rule_q, "R": _rule_r, "S": _rule_s,
    "T": _rule_t, "U": _rule_u, "V": _rule_v, "W": _rule_w, "X": _rule_x, "Y": _rule_y,
}  # fmt: skip

#: Whole-word handshapes recognised statically.
WORD_RULES: dict[str, Callable[[Feat], Rule]] = {"I LOVE YOU": _rule_ily}


def score_rule(rule: Rule) -> float:
    """Combine critical (min-gated) and supporting (weighted geometric mean) evidence."""
    crit, soft = rule
    gate = min(crit) if crit else 1.0
    if soft:
        wsum = sum(w for _, w in soft)
        geo = math.exp(sum(w * math.log(max(m, _FLOOR)) for m, w in soft) / wsum)
    else:
        geo = 1.0
    return float(gate**0.7 * geo)


def score_all(f: Feat, include_words: bool = True) -> dict[str, float]:
    """Score every static letter (and word handshape) for one hand."""
    out = {name: score_rule(fn(f)) for name, fn in LETTER_RULES.items()}
    if include_words:
        out.update({name: score_rule(fn(f)) for name, fn in WORD_RULES.items()})
    return out


def to_probabilities(scores: dict[str, float], sharpness: float = 3.0) -> dict[str, float]:
    """Turn raw rule scores into a confidence-style distribution.

    The best score sets how well *any* letter fits; the sharpened share sets how much better
    it fits than its rivals. Genuinely ambiguous shapes (R vs U, M vs N) therefore report a
    lower confidence instead of a false certainty.
    """
    if not scores:
        return {}
    best = max(scores.values())
    if best <= 1e-6:
        return dict.fromkeys(scores, 0.0)
    powered = {k: v**sharpness for k, v in scores.items()}
    total = sum(powered.values()) + 1e-12
    return {k: (p / total) * best for k, p in powered.items()}
