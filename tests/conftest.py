import os
import tempfile
from pathlib import Path

import pytest

# Keep tests away from the developer's real settings, samples and model.
_HOME = tempfile.mkdtemp(prefix="signscribe_tests_")
os.environ["SIGNSCRIBE_HOME"] = _HOME
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture()
def tmp_home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("SIGNSCRIBE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="session")
def rng_seed() -> int:
    return 1234
