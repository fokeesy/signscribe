"""Smoke-test the Tk window in demo mode. Skipped when there is no display (e.g. headless CI).

One window is shared by every test in this module: repeatedly creating and destroying Tk roots
in a single process is slow and, on Windows, occasionally trips a Tcl start-up race.
"""

import time
import tkinter as tk

import pytest

from signscribe.config import Settings


def _make(factory, attempts=3):
    last = None
    for _ in range(attempts):
        try:
            return factory()
        except tk.TclError as exc:  # transient Tcl init failures or no display
            last = exc
            time.sleep(0.3)
    raise last


def _display_available() -> bool:
    try:
        _make(lambda: tk.Tk().destroy())
    except tk.TclError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _display_available(), reason="no display available for Tk")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    import os

    os.environ["SIGNSCRIBE_HOME"] = str(tmp_path_factory.mktemp("gui_home"))
    from signscribe.gui.app import SignScribeApp

    a = _make(lambda: SignScribeApp(Settings(), demo=True, demo_text="hi"))
    yield a
    if a.winfo_exists():
        a._on_close()


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.01)


def test_window_types_the_demo_sentence(app):
    pump(app, 5.5)
    assert app.text.get("1.0", "end-1c").strip().lower() == "hi"
    assert app.engine.error is None
    assert app.pill.cget("text") in ("Demo", "Demo - no hand")


def test_buttons_edit_the_sentence_and_sync_the_composer(app):
    app.text.delete("1.0", "end")
    app._insert("hello")
    app._punct(".")
    assert app.text.get("1.0", "end-1c") == "hello. "
    app._backspace()
    assert app.text.get("1.0", "end-1c") == "hello."
    app._clear_text()
    assert app.text.get("1.0", "end-1c") == ""
    assert app.engine.composer.text == ""


def test_tabs_and_help_tiles_build(app):
    for i in range(4):
        app.tabs.select(i)
        app.update()
    assert app._help_built
    app.tabs.select(0)


def test_pausing_survives_a_source_switch(app):
    app.pause_var.set(True)
    app._on_pause()
    assert app.engine.paused
    app._switch_source("demo")
    assert app.engine.paused is True
    assert app.engine.source.name.startswith("Demo")
    app.pause_var.set(False)
    app._on_pause()


def test_calibration_wizard_manual_recording_and_training_in_the_window(app, monkeypatch):
    """Drive the Train tab with the demo hand: wizard -> samples -> train -> model in use."""
    from signscribe.gui import app as app_module
    from signscribe.gui import wizard

    monkeypatch.setattr(wizard, "COUNTDOWN", 0.15)
    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *a, **k: False)
    monkeypatch.setattr(app_module.messagebox, "showinfo", lambda *a, **k: None)
    app._switch_source("demo")
    app.store.clear()
    app.tabs.select(1)
    pump(app, 1.2)  # let the demo hand appear

    app.calibrator.start(["A", "B"])
    app._wizard_buttons(True)
    assert app.wiz_hint.winfo_manager() == "pack" and str(app.wiz_start.cget("state")) == "disabled"
    deadline = time.time() + 25
    while app.calibrator.active and time.time() < deadline:
        pump(app, 0.2)
    pump(app, 0.5)
    assert app.calibrator.recorded == ["A", "B"]
    assert app.engine.calibrating is False
    assert set(app.store.counts()) == {"A", "B"}
    assert app.text.get("1.0", "end-1c") == ""  # nothing typed during calibration
    assert not app.wiz_hint.winfo_manager()
    assert set(app.sample_tree.get_children()) == {"A", "B"}

    app.manual_label.set("Q")
    app._manual_record()
    deadline = time.time() + 15
    while app.engine.recording_state() and time.time() < deadline:
        pump(app, 0.2)
    assert app.store.counts().get("Q", 0) >= 60

    app._train()
    deadline = time.time() + 40
    while app._training and time.time() < deadline:
        pump(app, 0.2)
    pump(app, 0.3)
    assert app.model is not None and set(app.model.labels) == {"A", "B", "Q"}
    assert app.engine.model is app.model
    assert "Personal model active" in app.model_status.cget("text")
    app._remove_model()
    assert app.model is None and app.engine.model is None


def test_camera_error_is_shown_not_raised(app):
    app._switch_source("camera")  # no camera in CI; a real one on a developer machine
    pump(app, 1.5)
    if app.engine.error:  # only asserted when the camera really failed to open
        assert app.err_frame.winfo_ismapped()
    app._switch_source("demo")
