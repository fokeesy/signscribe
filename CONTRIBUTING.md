# Contributing

Thanks for helping! Bug reports, accuracy reports from real signers, docs fixes and code are all welcome.

## Setup

```bash
git clone https://github.com/fokeesy/signscribe.git
cd signscribe
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Python 3.10-3.12. You do not need a camera: `signscribe --demo` runs the whole stack with a simulated hand,
and the tests drive the real pipeline with `signscribe.demo.VirtualSigner`.

## Before you open a PR

```bash
ruff check . && ruff format .
pytest
```

CI runs the same on Linux and Windows for Python 3.10, 3.11 and 3.12.

## Where things live

Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The short version:

* `features.py` is the single definition of the hand-shape features; change it and the rules *and* the
  trained-model format (`VECTOR_KEYS`) are affected. Old model files are rejected automatically.
* `rules.py` - one small function per letter. Keep thresholds as trapezoid memberships and
  explain non-obvious numbers in a comment.
* `composer.py` - timing logic. It is fed virtual timestamps in tests, so it must not read the clock.
* `synthetic.py` - the procedural hand. Poses are approximations; if you improve one, say what
  reference you used.

## Changing recognition behaviour

Please include evidence. The most useful contribution this project can get is **real-signer data**:

* Run the guided calibration on your hand and report per-letter accuracy, camera, lighting, and which
  letters were confused, or
* open a PR that adds a small, properly licensed landmark dataset plus a script to evaluate the rules
  against it.

For rule tweaks, `tests/test_rules.py` (randomised accuracy on synthetic hands) and
`tests/test_pipeline.py` (end-to-end typing) must still pass, and a change that helps one letter but
hurts another needs a note on why the trade is worthwhile.

## Style

* Type hints, short docstrings that explain *why*, no dead code.
* Keep the GUI thread free of heavy work; anything slow goes on a worker thread and reports back through a queue.
* New user-facing settings go in `config.Settings` (so they persist) and get a control in the Settings tab.

## Code of conduct

Be kind and assume good faith. Sign language belongs to the Deaf community: if you have lived
experience of ASL, please speak up when something here is wrong or unhelpful. That feedback is a gift.
