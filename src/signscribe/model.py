"""Small NumPy neural network for personalised hand-shape classification.

Why NumPy and not PyTorch/TensorFlow: the network is tiny (a few thousand parameters), trains
in seconds on a laptop CPU, and this keeps installation light. Models are stored as plain
``.npz`` archives (no pickle), so loading a model file can never execute code.

Training data is landmark-level so it can be *augmented* (jitter, scale, small rotations,
mirroring) before the invariant features are computed. See :mod:`signscribe.training`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as F

MODEL_FORMAT_VERSION = 1


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class MLPClassifier:
    """Two-hidden-layer MLP: features -> 96 -> 48 -> classes, softmax output."""

    labels: list[str]
    mean: np.ndarray
    std: np.ndarray
    weights: list[np.ndarray] = field(default_factory=list)
    biases: list[np.ndarray] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def init(
        cls,
        labels: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        hidden: tuple[int, ...] = (96, 48),
        seed: int = 0,
    ) -> MLPClassifier:
        rng = np.random.default_rng(seed)
        sizes = [len(mean), *hidden, len(labels)]
        weights, biases = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            weights.append(rng.normal(0.0, np.sqrt(2.0 / a), (a, b)).astype(np.float32))
            biases.append(np.zeros(b, dtype=np.float32))
        return cls(list(labels), mean.astype(np.float32), std.astype(np.float32), weights, biases)

    # ---------------------------------------------------------------------- inference
    def _forward(self, x: np.ndarray, keep: bool = False):
        acts = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            h = z if i == len(self.weights) - 1 else np.maximum(z, 0.0)
            acts.append(h)
        return (h, acts) if keep else h

    def predict_proba_features(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities for an (N, D) matrix of raw feature vectors."""
        z = (np.atleast_2d(x).astype(np.float32) - self.mean) / self.std
        return _softmax(self._forward(z))

    def predict_proba(self, world: np.ndarray, roll: float | np.ndarray) -> np.ndarray:
        feats = F.extract_batch(world, roll)
        return self.predict_proba_features(F.to_vector(feats))

    def predict_dict(self, world: np.ndarray, roll: float) -> dict[str, float]:
        p = self.predict_proba(world, roll)[0]
        return {label: float(v) for label, v in zip(self.labels, p)}

    # ---------------------------------------------------------------------- training
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 60,
        lr: float = 3e-3,
        batch: int = 128,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        seed: int = 0,
        make_batch=None,
        patience: int = 12,
        progress=None,
    ) -> dict:
        """Train with Adam. ``x`` is either raw features or ignored when ``make_batch`` is set.

        Args:
            x, y: raw (un-normalised) features and integer labels.
            make_batch: optional callable ``(epoch) -> (x, y)`` that regenerates an augmented
                training set every epoch (feature-space noise alone is a poor substitute for
                landmark-space augmentation).
            progress: optional callable ``(epoch, loss, val_acc)``.
        """
        rng = np.random.default_rng(seed)
        n_classes = len(self.labels)
        params = [*self.weights, *self.biases]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        best = (-1.0, None)
        stale = 0
        history = {"loss": [], "val_acc": []}

        for epoch in range(epochs):
            xe, ye = make_batch(epoch) if make_batch else (x, y)
            xn = ((xe - self.mean) / self.std).astype(np.float32)
            order = rng.permutation(len(xn))
            losses = []
            for start in range(0, len(order), batch):
                idx = order[start : start + batch]
                xb, yb = xn[idx], ye[idx]
                out, acts = self._forward(xb, keep=True)
                p = _softmax(out)
                target = np.full_like(p, label_smoothing / max(n_classes - 1, 1))
                target[np.arange(len(yb)), yb] = 1.0 - label_smoothing
                losses.append(float(-(target * np.log(p + 1e-9)).sum(axis=1).mean()))
                grad = (p - target) / len(yb)
                gw, gb = [None] * len(self.weights), [None] * len(self.weights)
                for layer in range(len(self.weights) - 1, -1, -1):
                    gw[layer] = acts[layer].T @ grad + weight_decay * self.weights[layer]
                    gb[layer] = grad.sum(axis=0)
                    if layer > 0:
                        grad = (grad @ self.weights[layer].T) * (acts[layer] > 0)
                step += 1
                grads = [*gw, *gb]
                for i, (p_, g) in enumerate(zip(params, grads)):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    mh = m[i] / (1 - 0.9**step)
                    vh = v[i] / (1 - 0.999**step)
                    p_ -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(p_.dtype)
            loss = float(np.mean(losses))
            history["loss"].append(loss)
            if x_val is not None and y_val is not None and len(x_val):
                acc = float((self.predict_proba_features(x_val).argmax(axis=1) == y_val).mean())
            else:
                acc = float((self.predict_proba_features(xe).argmax(axis=1) == ye).mean())
            history["val_acc"].append(acc)
            if progress:
                progress(epoch, loss, acc)
            if acc > best[0] + 1e-4:
                best = (acc, ([w.copy() for w in self.weights], [b.copy() for b in self.biases]))
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best[1] is not None:
            self.weights, self.biases = best[1]
        history["best_val_acc"] = best[0]
        return history

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "labels": np.array(self.labels),
            "mean": self.mean,
            "std": self.std,
            "n_layers": np.array(len(self.weights)),
            "meta": np.array(json.dumps({**self.meta, "format": MODEL_FORMAT_VERSION, "features": list(F.VECTOR_KEYS)})),
        }
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> MLPClassifier:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("features") != list(F.VECTOR_KEYS):
                raise ValueError("model was trained with a different feature set; please retrain")
            n = int(z["n_layers"])
            return cls(
                labels=[str(s) for s in z["labels"]],
                mean=z["mean"],
                std=z["std"],
                weights=[z[f"w{i}"] for i in range(n)],
                biases=[z[f"b{i}"] for i in range(n)],
                meta=meta,
            )
