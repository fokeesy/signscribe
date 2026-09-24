"""Per-frame sign recognition: geometry rules, optional personal model, or both."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import features as F
from . import rules
from .constants import ALPHABET, MOTION_LETTERS
from .hand import HandObservation
from .model import MLPClassifier

STATIC_LETTERS = tuple(c for c in ALPHABET if c not in MOTION_LETTERS)


@dataclass
class Prediction:
    """What the recogniser thinks one frame shows."""

    label: str | None = None
    confidence: float = 0.0
    top: list[tuple[str, float]] = field(default_factory=list)
    source: str = "none"  # rules | model | hybrid
    features: dict[str, float] = field(default_factory=dict)


class Recognizer:
    """Classifies a hand observation into a letter or word handshape.

    Modes:
        ``rules``  geometry rules only (works out of the box)
        ``model``  the trained personal model only
        ``auto``   hybrid when a model is loaded, otherwise rules
    """

    MODEL_WEIGHT = 0.8  # trust in a model that covers the whole alphabet

    def __init__(self, mode: str = "auto", model: MLPClassifier | None = None, include_words: bool = True) -> None:
        self.mode = mode
        self.model = model
        self.include_words = include_words

    def set_model(self, model: MLPClassifier | None) -> None:
        self.model = model

    @property
    def effective_mode(self) -> str:
        if self.mode == "rules" or self.model is None:
            return "rules"
        return "model" if self.mode == "model" else "hybrid"

    def _model_weight(self) -> float:
        if self.model is None:
            return 0.0
        covered = len(set(self.model.labels) & set(STATIC_LETTERS))
        return self.MODEL_WEIGHT * min(1.0, covered / max(len(STATIC_LETTERS) * 0.85, 1))

    def predict_features(self, feats: dict[str, np.ndarray]) -> Prediction:
        scalars = {k: float(v[0]) for k, v in feats.items()}
        mode = self.effective_mode
        scores: dict[str, float] = {}

        rule_p = rules.to_probabilities(rules.score_all(scalars, self.include_words)) if mode != "model" else {}
        model_p: dict[str, float] = {}
        if mode != "rules" and self.model is not None:
            probs = self.model.predict_proba_features(F.to_vector(feats))[0]
            model_p = {lab: float(p) for lab, p in zip(self.model.labels, probs)}

        if mode == "rules":
            scores = rule_p
        elif mode == "model":
            scores = model_p
        else:
            w = self._model_weight()
            pm_max = max(model_p.values()) if model_p else 0.0
            for lab in set(rule_p) | set(model_p):
                pr = rule_p.get(lab, 0.0)
                if lab in model_p:
                    scores[lab] = w * model_p[lab] + (1.0 - w) * pr
                else:
                    scores[lab] = pr * (1.0 - w * pm_max)

        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        label, conf = top[0] if top else (None, 0.0)
        if conf < 0.25:
            label = None
        return Prediction(label, float(conf), [(k, float(v)) for k, v in top], mode, scalars)

    def predict(self, obs: HandObservation) -> Prediction:
        feats = F.extract_batch(obs.world, obs.roll)
        return self.predict_features(feats)
