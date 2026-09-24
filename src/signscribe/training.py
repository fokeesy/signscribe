"""Recording, storing and training on your own hand samples.

Samples are stored at the *landmark* level (21x3 world coordinates + in-plane roll) so they can
be augmented before features are computed. Recording ~60 frames per letter (a few seconds of
holding the sign, slightly moving and rotating the hand) is enough to get a personal model
that beats the built-in rules for the hard letters.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from . import features as F
from .config import model_path, samples_path
from .hand import HandObservation
from .model import MLPClassifier


class SampleStore:
    """Growing set of labelled hand samples, persisted as an ``.npz`` file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or samples_path()
        self.world = np.zeros((0, 21, 3), dtype=np.float32)
        self.roll = np.zeros((0,), dtype=np.float32)
        self.labels: list[str] = []

    # -------------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.labels)

    def counts(self) -> dict[str, int]:
        return dict(Counter(self.labels))

    def add(self, label: str, obs: HandObservation) -> None:
        self.add_raw(label, obs.world, obs.roll)

    def add_raw(self, label: str, world: np.ndarray, roll: float) -> None:
        self.world = np.concatenate([self.world, np.asarray(world, dtype=np.float32)[None]], axis=0)
        self.roll = np.concatenate([self.roll, np.array([roll], dtype=np.float32)])
        self.labels.append(label)

    def remove_label(self, label: str) -> int:
        keep = np.array([lab != label for lab in self.labels], dtype=bool)
        removed = int((~keep).sum())
        self.world, self.roll = self.world[keep], self.roll[keep]
        self.labels = [lab for lab, k in zip(self.labels, keep) if k]
        return removed

    def clear(self) -> None:
        self.__init__(self.path)

    # --------------------------------------------------------------- persistence
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, world=self.world, roll=self.roll, labels=np.array(self.labels))

    @classmethod
    def load(cls, path: Path | None = None) -> SampleStore:
        store = cls(path)
        if store.path.exists():
            with np.load(store.path, allow_pickle=False) as z:
                store.world = z["world"].astype(np.float32)
                store.roll = z["roll"].astype(np.float32)
                store.labels = [str(s) for s in z["labels"]]
        return store


def augment(world: np.ndarray, roll: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Landmark-space augmentation: sensor jitter, per-axis stretch and in-plane roll wobble.

    Rotation and uniform scale are *not* applied to the landmarks because the feature set is
    already invariant to them; they would be no-ops.
    """
    n = len(world)
    w = world.astype(np.float64).copy()
    stretch = rng.uniform(0.93, 1.07, (n, 1, 3))
    w = w * stretch
    w = w + rng.normal(0.0, 0.0016, w.shape)  # ~1.6 mm of landmark jitter
    r = roll.astype(np.float64) + rng.normal(0.0, np.radians(10.0), n)
    return w, r


def _blocked_split(labels: list[str], every: int = 5) -> np.ndarray:
    """Validation mask that holds out ~20 % of each label as contiguous blocks.

    Consecutive frames are near-duplicates, so a random split would leak training data into
    validation and report optimistic accuracy.
    """
    totals = Counter(labels)
    is_val = np.zeros(len(labels), dtype=bool)
    seen: Counter = Counter()
    for i, lab in enumerate(labels):
        k = seen[lab]
        seen[lab] += 1
        block = max(3, totals[lab] // (2 * every))
        if (k // block) % every == every - 1:
            is_val[i] = True
    return is_val


def train_model(
    store: SampleStore,
    epochs: int = 80,
    augment_factor: int = 6,
    seed: int = 0,
    progress=None,
    min_per_label: int = 8,
) -> tuple[MLPClassifier, dict]:
    """Train a personal classifier on the recorded samples.

    Returns the model and a report dict with per-label validation accuracy.
    """
    counts = store.counts()
    labels = sorted(lab for lab, c in counts.items() if c >= min_per_label)
    if len(labels) < 2:
        raise ValueError(f"need at least 2 labels with {min_per_label}+ samples each (have {counts})")
    keep = np.array([lab in labels for lab in store.labels])
    world, roll = store.world[keep], store.roll[keep]
    y_names = [lab for lab in store.labels if lab in labels]
    y = np.array([labels.index(lab) for lab in y_names])

    is_val = _blocked_split(y_names) if min(counts[lab] for lab in labels) >= 25 else np.zeros(len(y), bool)
    tr, va = ~is_val, is_val

    rng = np.random.default_rng(seed)

    def batch(epoch: int):
        ws, rs, ys = [world[tr]], [roll[tr]], [y[tr]]
        for _ in range(augment_factor):
            aw, ar = augment(world[tr], roll[tr], rng)
            ws.append(aw), rs.append(ar), ys.append(y[tr])
        w_all, r_all, y_all = np.concatenate(ws), np.concatenate(rs), np.concatenate(ys)
        return F.to_vector(F.extract_batch(w_all, r_all)), y_all

    x0, _ = batch(0)
    mean, std = x0.mean(axis=0), x0.std(axis=0) + 1e-3
    model = MLPClassifier.init(labels, mean, std, seed=seed)
    x_val = F.to_vector(F.extract_batch(world[va], roll[va])) if va.any() else None
    y_val = y[va] if va.any() else None
    hist = model.fit(None, None, x_val, y_val, epochs=epochs, make_batch=batch, seed=seed, progress=progress)

    # Report: accuracy on held-out frames (or training frames when too little data).
    xe = x_val if x_val is not None else F.to_vector(F.extract_batch(world, roll))
    ye = y_val if y_val is not None else y
    pred = model.predict_proba_features(xe).argmax(axis=1)
    per_label = {lab: float((pred[ye == i] == i).mean()) for i, lab in enumerate(labels) if (ye == i).any()}
    report = {
        "labels": labels,
        "samples": int(len(y)),
        "held_out": bool(x_val is not None),
        "accuracy": float((pred == ye).mean()),
        "per_label": per_label,
        "epochs_run": len(hist["loss"]),
    }
    model.meta.update({"samples": int(len(y)), "accuracy": report["accuracy"]})
    return model, report


def load_model(path: Path | None = None) -> MLPClassifier | None:
    """Load the user's trained model if present and compatible; otherwise ``None``."""
    path = path or model_path()
    if not path.exists():
        return None
    try:
        return MLPClassifier.load(path)
    except (OSError, ValueError, KeyError):
        return None


def dataset_from_images(root: Path, tracker, max_per_class: int | None = None, progress=None) -> SampleStore:
    """Build a sample store from a folder of labelled images (``root/A/*.jpg`` ...).

    Works with the common Kaggle "ASL Alphabet" layout. Images where no hand is found are
    skipped. ``tracker`` needs a ``process(frame_bgr, t)`` method (see :class:`HandTracker`
    created with ``static_image_mode=True``).
    """
    import cv2

    store = SampleStore(path=Path(root) / "_samples_tmp.npz")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = sorted(p for p in Path(root).iterdir() if p.is_dir())
    for ci, cls_dir in enumerate(classes):
        label = cls_dir.name.upper() if len(cls_dir.name) == 1 else cls_dir.name
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in exts)
        if max_per_class:
            files = files[:max_per_class]
        for fi, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None:
                continue
            hands = tracker.process(img, float(fi))
            if hands:
                best = max(hands, key=lambda h: h.score)
                store.add(label, best)
            if progress:
                progress(ci, len(classes), label, fi + 1, len(files))
    return store
