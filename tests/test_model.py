from pathlib import Path

import numpy as np
import pytest

from signscribe import features as F
from signscribe.config import Settings
from signscribe.model import MLPClassifier
from signscribe.recognizer import Recognizer
from signscribe.synthetic import POSES, random_observation
from signscribe.training import SampleStore, augment, load_model, train_model

LETTERS = [n for n in POSES if n not in ("OPEN", "J", "Z")]


def label_of(name):
    return "I LOVE YOU" if name == "ILY" else name


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(3)
    store = SampleStore(path=Path("unused.npz"))  # never saved
    for name in LETTERS:
        for _ in range(45):
            store.add(label_of(name), random_observation(name, rng, noise=0.02))
    model, report = train_model(store, epochs=40)
    return store, model, report


def test_sample_store_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    store = SampleStore(tmp_path / "s.npz")
    for name in ("A", "B", "A"):
        store.add(name, random_observation(name, rng))
    store.save()
    loaded = SampleStore.load(tmp_path / "s.npz")
    assert len(loaded) == 3 and loaded.counts() == {"A": 2, "B": 1}
    assert loaded.world.shape == (3, 21, 3)
    assert loaded.remove_label("A") == 2 and len(loaded) == 1
    loaded.clear()
    assert len(loaded) == 0


def test_missing_sample_file_gives_empty_store(tmp_path):
    assert len(SampleStore.load(tmp_path / "nope.npz")) == 0


def test_augment_changes_landmarks_but_keeps_shape():
    rng = np.random.default_rng(0)
    w = np.stack([random_observation("A", rng).world for _ in range(5)])
    aw, ar = augment(w, np.zeros(5, np.float32), rng)
    assert aw.shape == w.shape and not np.allclose(aw, w)
    assert ar.shape == (5,)


def test_training_learns_and_reports(trained):
    _store, model, report = trained
    assert report["held_out"] is True
    assert report["accuracy"] > 0.95
    assert set(report["labels"]) == {label_of(n) for n in LETTERS}
    assert model.meta["samples"] > 0


def test_model_beats_or_matches_rules_under_heavy_noise(trained):
    _s, model, _r = trained
    rng = np.random.default_rng(99)
    hybrid = Recognizer("auto", model)
    rules_only = Recognizer("rules")
    ok_h = ok_r = total = 0
    for name in LETTERS:
        for _ in range(15):
            obs = random_observation(name, rng, noise=0.035)
            total += 1
            ok_h += hybrid.predict(obs).label == label_of(name)
            ok_r += rules_only.predict(obs).label == label_of(name)
    assert ok_h / total >= ok_r / total
    assert ok_h / total > 0.9


def test_save_load_roundtrip_and_feature_guard(trained, tmp_path):
    _s, model, _r = trained
    path = tmp_path / "m.npz"
    model.save(path)
    again = MLPClassifier.load(path)
    rng = np.random.default_rng(0)
    obs = random_observation("B", rng)
    a, b = model.predict_dict(obs.world, obs.roll), again.predict_dict(obs.world, obs.roll)
    assert all(abs(a[k] - b[k]) < 1e-6 for k in a)
    assert abs(sum(a.values()) - 1.0) < 1e-4

    # A model trained with a different feature set must be refused, not silently misused.
    import json

    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data["meta"]))
    meta["features"] = meta["features"][:-1]
    data["meta"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "old.npz", **data)
    with pytest.raises(ValueError):
        MLPClassifier.load(tmp_path / "old.npz")


def test_load_model_is_safe_on_garbage(tmp_path):
    bad = tmp_path / "model.npz"
    bad.write_bytes(b"not a model")
    assert load_model(bad) is None
    assert load_model(tmp_path / "missing.npz") is None


def test_train_needs_enough_data():
    store = SampleStore(path=Path("x.npz"))
    rng = np.random.default_rng(0)
    store.add("A", random_observation("A", rng))
    with pytest.raises(ValueError):
        train_model(store)


def test_partial_model_does_not_veto_untrained_letters(trained):
    """A model trained on a few letters must not override rules for letters it never saw."""
    rng = np.random.default_rng(5)
    small = SampleStore(path=Path("y.npz"))
    for name in ("A", "B", "C"):
        for _ in range(45):
            small.add(name, random_observation(name, rng, noise=0.02))
    model, _ = train_model(small, epochs=30)
    rec = Recognizer("auto", model)
    ok = 0
    for _ in range(20):
        ok += rec.predict(random_observation("L", rng, noise=0.02)).label == "L"
    assert ok >= 14


def test_recognizer_modes_and_settings_wiring(trained):
    _s, model, _ = trained
    obs = random_observation("Y", np.random.default_rng(1), noise=0.0)
    assert Recognizer("rules", model).effective_mode == "rules"
    assert Recognizer("model", model).effective_mode == "model"
    assert Recognizer("auto", model).effective_mode == "hybrid"
    assert Recognizer("auto", None).effective_mode == "rules"
    for mode in ("rules", "model", "auto"):
        assert Recognizer(mode, model).predict(obs).label == "Y"
    assert Settings().recognition_mode == "auto"
    assert F.vector_dim() == model.mean.shape[0]
