from signscribe.composer import Composer
from signscribe.config import Settings
from signscribe.dynamic import MotionEvent


def make(**kw) -> Composer:
    s = Settings(hold_time=0.5, repeat_delay=1.0, space_timeout=1.0, min_confidence=0.5, speed_gate=1.4, **kw)
    return Composer(s)


def feed(c: Composer, label, seconds, start=0.0, conf=0.9, speed=0.05, present=True, fps=30):
    """Feed identical frames; returns (events, end_time)."""
    out = []
    n = int(seconds * fps)
    for k in range(n):
        out += c.update(start + k / fps, present, label, conf, speed)
    return out, start + n / fps


def text_of(events):
    return "".join(e.text for e in events if e.kind == "text")


def test_holding_a_sign_types_it_once_with_smart_caps():
    c = make()
    events, _ = feed(c, "A", 1.0)
    assert text_of(events) == "A"  # first letter of a sentence is capitalised
    assert c.text == "A"


def test_sign_shorter_than_hold_time_does_not_type():
    c = make()
    events, _ = feed(c, "A", 0.3)
    assert events == []


def test_progress_ramps_up_while_holding():
    c = make()
    seen = []
    for k in range(14):  # 0.47 s: long enough to be well under way, short of the 0.5 s hold
        c.update(k / 30, True, "B", 0.9, 0.0)
        seen.append(c.progress)
    assert seen[0] == 0.0 and max(seen) > 0.6 and max(seen) < 1.0
    assert seen == sorted(seen)


def test_low_confidence_or_fast_motion_never_types():
    c = make()
    events, _ = feed(c, "A", 1.5, conf=0.3)
    assert events == []
    events, _ = feed(make(), "A", 1.5, speed=3.0)
    assert events == []


def test_changing_sign_types_both_letters_in_order():
    c = make()
    e1, t = feed(c, "H", 0.9)
    e2, t = feed(c, "I", 0.9, start=t)
    assert text_of(e1 + e2) == "Hi"


def test_same_letter_needs_release_or_long_hold():
    c = make()
    e1, t = feed(c, "L", 0.8)
    assert text_of(e1) == "L"
    # still holding L: no repeat before repeat_delay...
    e2, t = feed(c, "L", 0.5, start=t)
    assert text_of(e2) == ""
    # ...and a double letter once it is exceeded
    e3, t = feed(c, "L", 1.0, start=t)
    assert text_of(e3) == "l"


def test_dropping_the_hand_rearms_the_same_letter():
    c = make()
    e1, t = feed(c, "L", 0.8)
    _, t = feed(c, None, 0.2, start=t, present=False)
    e2, t = feed(c, "L", 0.8, start=t)
    assert text_of(e1) + text_of(e2) == "Ll"


def test_hand_out_of_view_adds_exactly_one_space():
    c = make()
    e1, t = feed(c, "A", 0.8)
    e2, t = feed(c, None, 3.0, start=t, present=False)
    assert text_of(e2) == " "
    assert c.text == "A "
    e3, t = feed(c, None, 3.0, start=t, present=False)
    assert e3 == []


def test_no_leading_space_when_nothing_typed():
    c = make()
    events, _ = feed(c, None, 3.0, present=False)
    assert events == [] and c.text == ""


def test_smart_case_after_sentence_end_and_pronoun_fix():
    c = make()
    c.set_text("Ok. ")
    e, _ = feed(c, "B", 0.8)
    assert text_of(e) == "B"
    c2 = make()
    c2.set_text("")
    events, t = feed(c2, "I", 0.8)
    events2, t = feed(c2, None, 2.0, start=t, present=False)
    assert c2.text == "I "
    c3 = make()
    c3.set_text("so ")
    e1, t = feed(c3, "I", 0.8)
    e2, t = feed(c3, None, 2.0, start=t, present=False)
    assert c3.text == "so I "


def test_upper_and_lower_modes():
    up = Composer(Settings(hold_time=0.3, case_mode="upper", speed_gate=9))
    feed(up, "a", 0.8)
    up.set_text("x")
    e, _ = feed(up, "b", 0.8, start=5.0)
    assert text_of(e) == "B"
    lo = Composer(Settings(hold_time=0.3, case_mode="lower", speed_gate=9))
    e, _ = feed(lo, "A", 0.8)
    assert text_of(e) == "a"


def test_motion_event_types_immediately_and_locks_static_signs():
    c = make()
    c.set_text("ha")
    ev = c.on_motion(MotionEvent("Z", "letter", 0.9, 1.0))
    assert text_of(ev) == "z"
    # still holding the shape recognised as 'D': must not type for a while
    events, _ = feed(c, "D", 0.8, start=1.0)
    assert events == []


def test_motion_event_suppresses_the_held_static_shape_until_released():
    c = make()
    feed(c, "I", 0.2)  # hand is in the I shape...
    ev = c.on_motion(MotionEvent("J", "letter", 0.9, 0.3))
    assert text_of(ev) == "J"
    late, t = feed(c, "I", 2.5, start=1.5)  # ...and stays there long after the lock
    assert late == []
    released, _ = feed(c, "B", 0.8, start=t)
    assert text_of(released) == "b"


def test_word_events_get_spacing_and_case():
    c = make()
    ev = c.on_motion(MotionEvent("Hello", "word", 0.9, 1.0))
    assert text_of(ev) == "Hello "
    ev = c.on_motion(MotionEvent("Yes", "word", 0.9, 5.0))
    assert text_of(ev) == "yes "
    assert c.text == "Hello yes "


def test_custom_word_labels_are_typed_naturally():
    c = make()
    events, t = feed(c, "THANK YOU", 1.0)
    assert text_of(events) == "Thank you "
    events, t = feed(c, "PLEASE", 1.0, start=t + 1.0)
    assert text_of(events) == "please "
    upper = Composer(Settings(hold_time=0.3, case_mode="upper", speed_gate=9))
    events, _ = feed(upper, "Thank you", 0.8)
    assert text_of(events) == "THANK YOU "


def test_static_word_handshape_ily():
    c = make()
    events, _ = feed(c, "I LOVE YOU", 1.0)
    assert text_of(events) == "I love you "


def test_multiple_disagreeing_frames_do_not_type_noise():
    c = make()
    out = []
    for k in range(90):  # alternates every frame: no stable majority
        out += c.update(k / 30, True, "AB"[k % 2], 0.9, 0.0)
    assert out == []
