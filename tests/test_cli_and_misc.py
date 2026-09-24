import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from signscribe import __version__, cli
from signscribe import reference as ref
from signscribe.config import Settings, model_path, samples_path, user_dir
from signscribe.constants import ALPHABET, BONES, FINGER_CHAINS
from signscribe.hand import rotation_angles
from signscribe.synthetic import POSES, build_local, letter_observation, local_to_camera, resolved


def test_every_letter_has_a_pose_a_tip_and_a_rendering():
    for letter in ALPHABET:
        assert letter in POSES and letter in ref.TIPS
        img = ref.render_reference(letter, 96)
        assert img.shape == (96, 96, 3) and img.std() > 5  # something is drawn


def test_contact_sheet_dimensions():
    sheet = ref.contact_sheet(cols=7, size=64)
    assert sheet.shape == (64 * 4, 64 * 7, 3)


def test_skeleton_topology_is_sane():
    assert len(BONES) == 21
    assert {i for _, a, b in BONES for i in (a, b)} == set(range(21))
    assert all(len(chain) == 4 for chain in FINGER_CHAINS.values())


def test_synthetic_hand_has_plausible_proportions():
    local = build_local(resolved("B"))
    # Wrist -> middle fingertip of an open hand is roughly 17-20 cm.
    length = np.linalg.norm(local[12] - local[0])
    assert 16.0 < length < 21.0
    # Bone lengths never change with pose.
    fist = build_local(resolved("S"))
    for chain in FINGER_CHAINS.values():
        for a, b in zip(chain[:-1], chain[1:]):
            assert np.linalg.norm(local[a] - local[b]) == pytest.approx(np.linalg.norm(fist[a] - fist[b]), abs=1e-6)


def test_rotation_angles_recover_synthetic_pose():
    pose = resolved("B")
    for roll in (-50, 0, 35):
        obs = letter_observation("B", roll=roll, yaw=0, pitch=0)
        r, p, y = rotation_angles(obs.world, True)
        # Roll is the angle of the wrist -> middle-knuckle axis, which sits ~4 degrees off the
        # model hand's "vertical", so allow that offset but require the world and image
        # estimates of roll to agree exactly.
        assert r == pytest.approx(roll, abs=6.0)
        assert r == pytest.approx(np.degrees(obs.roll), abs=0.5)
        assert abs(p) < 3 and abs(y) < 3
    yaws = [rotation_angles(letter_observation("B", roll=0, yaw=a, pitch=0).world, True)[2] for a in (-30, 0, 30)]
    assert yaws[0] < -20 < 20 < yaws[2] or yaws[0] > 20 > -20 > yaws[2]  # monotonic and roughly the right size
    pitched = rotation_angles(letter_observation("B", roll=0, yaw=0, pitch=35).world, True)[1]
    assert abs(pitched) > 25
    assert local_to_camera(build_local(pose), 0, 0, 40).shape == (21, 3)


def test_settings_roundtrip_ignores_unknown_keys(tmp_path):
    path = tmp_path / "s.json"
    s = Settings(hold_time=0.9, dominant_hand="Left")
    s.save(path)
    path.write_text(path.read_text().replace("{", '{"from_the_future": 1,', 1))
    loaded = Settings.load(path)
    assert loaded.hold_time == 0.9 and loaded.dominant_hand == "Left"
    assert Settings.load(tmp_path / "missing.json") == Settings()
    (tmp_path / "bad.json").write_text("not json")
    assert Settings.load(tmp_path / "bad.json") == Settings()


def test_user_dir_honours_override(tmp_home):
    assert user_dir() == tmp_home
    assert samples_path().parent == tmp_home and model_path().parent == tmp_home


def test_cli_reference_and_version(tmp_path, capsys):
    out = tmp_path / "sheet.png"
    assert cli.main(["reference", "--out", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 1000
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_cli_train_without_samples_fails_cleanly(tmp_home, capsys):
    assert cli.main(["train"]) == 1
    assert "no samples" in capsys.readouterr().err


def test_bare_options_route_to_gui(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_cmd_gui", lambda args: seen.update(demo=args.demo, text=args.text) or 0)
    assert cli.main(["--demo", "--text", "abc"]) == 0
    assert seen == {"demo": True, "text": "abc"}


def test_headless_demo_prints_recognised_text():
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, "-m", "signscribe", "headless", "--demo", "--text", "hi", "--seconds", "6"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().lower() == "hi"
