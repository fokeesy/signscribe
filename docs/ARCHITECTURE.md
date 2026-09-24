# Architecture

SignScribe is a pipeline of small, separately testable stages. This document explains each
stage and the reasoning behind the non-obvious choices.

## Data flow

```
CameraStream ─► Engine.process_packet ─► FrameResult ─► GUI
 (thread)         │
                  ├─ LightingAdapter          exposure correction (only fed to the tracker)
                  ├─ HandTracker              MediaPipe Hands + One Euro smoothing
                  ├─ Recognizer               features → rules / model → Prediction
                  ├─ MotionRecognizer         J, Z, wave, nod
                  ├─ Composer                 votes, hold timer, re-arming, spaces → TextEvent
                  └─ overlay                  skeleton, chip, ring, trail, dial
```

`Engine.process_packet(FramePacket)` is a plain, thread-free method. The worker thread only loops
`source.read() → process_packet → publish`. Tests and `signscribe headless` call `process_packet`
directly with *virtual timestamps*, so timing logic is deterministic and a minute-long signing
session is simulated in seconds instead of in real time.

Threads:

| Thread | Work |
|---|---|
| camera reader | `cv2.VideoCapture.read()` in a loop, keeps **only the newest frame** (no backlog, so latency does not grow when processing is slow) |
| engine | detection, recognition, composition, drawing |
| Tk main | polls `engine.latest()` every 12 ms, drains `engine.events`, updates widgets |
| training | started on demand; posts progress through a queue |

Only the Tk thread touches widgets. The engine communicates through two thread-safe channels: the
latest immutable-ish `FrameResult`, and a `queue.Queue` of `TextEvent`s (so no typed letter is ever
missed if the GUI is momentarily slow).

## Perception

### `HandObservation`

Two coordinate sets per hand, both from MediaPipe:

* `image` - normalised image coordinates (used for drawing, palm size in pixels and the in-plane
  **roll**, which is only defined in the image).
* `world` - metric 3D landmarks in metres, centred on the hand and aligned with the camera axes.
  Every *shape* feature uses these because they are independent of how far the hand is from the lens.

Handedness: MediaPipe assumes a mirrored ("selfie") image. SignScribe flips the frame when *mirror*
is on, so the label is anatomically correct; if you turn mirroring off the label is swapped. Notably,
**recognition does not use handedness at all** (see below); it is only shown in the UI and used to
pick the dominant hand.

### Lighting adapter (`preprocess.py`)

1. Meter luminance on the previous hand box (70 %) blended with the whole frame (30 %).
2. Exponential moving average, then a hysteresis state machine: engage at mean < 78 (dim) or > 172
   (bright), release inside 100 / 150, so it never flickers.
3. Auto-gamma toward mid-grey (via a 256-entry LUT), clamped to 0.45-2.2.
4. CLAHE on the L channel of LAB when the scene is dim/bright/flat.
5. If nothing is detected for 4+ frames, every third frame is retried on a CLAHE-enhanced copy.

### Smoothing (`filters.py`)

A One Euro filter per landmark set: low cutoff at rest (kills jitter), cutoff opening with speed
(no lag when you move). The *Smoothing* slider maps to the cutoff. Filters restart after a 0.4 s
gap so a hand reappearing elsewhere is not smeared.

## Features (`features.py`)

The single source of truth for both the rules and the learned model. Everything is computed from
`world` landmarks divided by the *palm length* (mean wrist-to-knuckle distance), so it is invariant
to scale, translation and (because only angles and distances are used) **rotation and mirroring**.
A left hand's "A" produces the same numbers as a right hand's "A".

| Feature | Meaning |
|---|---|
| `ext_i/m/r/p` | finger extension 0 (fist) .. 1 (ruler-straight), from the summed MCP+PIP+DIP flexion |
| `mcp_/pip_/dip_` | joint flexion angles in degrees |
| `th_ext`, `th_reach`, `th_mcp`, `th_ip` | thumb straightness and joint angles |
| `d_ti … d_rp`, `t_pip_*`, `t_mcp_*`, `t_dip_*`, `tip_pc_*`, `t_pc` | fingertip and thumb-tip distances (palm units) |
| `spread_im/mr/rp`, `ang_ti` | angle between adjacent fingers / between thumb and index |
| `t_along`, `t_height` | where the thumb tip sits across (0 = index side .. 1 = pinky side) and up the hand |
| `cross_im` | > 0 when index and middle fingertips have swapped sides (crossed fingers, R) |
| `up`, `side` | orientation from the in-plane roll: `cos(roll)`, `|sin(roll)|` |

Two details worth knowing:

* **Angle dead-zone.** Joint angles are magnitudes, so landmark noise always adds bend to a straight
  finger. A 7° dead-zone per joint removes that bias without affecting real curls. On the
  synthetic sweep, dropping it costs accuracy: 98 % → 95 % at realistic landmark jitter and
  82 % → 64 % at double that jitter.
* **Orientation** is the only non-invariant input, and it is mirror-safe (`|sin|`), which is what
  lets G/H (sideways) be told apart from Q/P (down) and U/K from H/P without any handedness logic.

## Recognition

### Geometry rules (`rules.py`)

Each letter is written the way a signing guide describes it:

```python
def _rule_l(f):  # index straight, other fingers closed, thumb out at ~90 degrees
    return _g_like(f), [
        (up(f["t_pc"], 0.90, 1.25), 2.5),
        (band(f["ang_ti"], 45.0, 65.0, 115.0, 140.0), 2.5),
        (up(f["th_reach"], 0.70, 0.90), 1.0),
    ]
```

* `crit` - memberships that must all hold; the **weakest gates** the score (`min`, then `^0.7`).
* `soft` - `(membership, weight)` pairs combined as a **weighted geometric mean** with a floor, so
  any clearly violated cue pulls the score down.

Memberships are trapezoids (`up`, `dn`, `band`), so thresholds degrade gracefully instead of
flipping. `to_probabilities` sharpens the scores and multiplies by the best raw score, so a shape
that fits *nothing* well, or two letters equally well (R vs U), reports a **low confidence** instead
of a false certainty. J and Z never appear in the static rules; they are motion letters.

### Personal model (`model.py`, `training.py`)

A NumPy MLP (features → 96 → 48 → classes, ReLU, softmax, Adam, label smoothing, L2). No deep
learning framework is needed and models are plain `.npz` archives loaded with `allow_pickle=False`,
so a model file cannot run code.

Samples are stored as **landmarks**, not features, so they can be augmented before feature
extraction: per-axis stretch, ~1.6 mm jitter and roll wobble. (Rotation and uniform scale would be
no-ops because the features are already invariant to them.) Validation uses held-out *contiguous
blocks* of frames per sign; a random split would leak near-duplicate consecutive frames into
validation and inflate the score.

### Fusion (`recognizer.py`)

`auto` mode blends model and rules: `w · p_model + (1 - w) · p_rules` for signs the model knows.
`w` scales with how much of the alphabet the model covers (max 0.8), and for signs the model has
*never seen* the rule score is discounted by how sure the model is about something else. A model
trained on three letters therefore cannot veto an L.

## Motion (`dynamic.py`)

Everything is measured in palm lengths, so distance to the camera does not matter, and strokes are
accepted in either direction so left-handed signers need no setting.

* **Stroke segmentation.** A fingertip stroke starts when its speed exceeds 1.1 palm/s and ends
  after 0.15 s of stillness, when the handshape stops matching, when it lasts > 2.6 s, or when the
  hand leaves view. (The last one matters: people often drop the hand straight after tracing.)
* **J**: pinky-only handshape; downward stem of ≥ 0.8 palm that hooks back up by ≥ 0.18 and sideways
  by ≥ 0.22. **Z**: index-only handshape; Douglas-Peucker simplification must yield exactly three
  segments: across, diagonal back, across.
* **Tail matching.** Strokes are judged on their best-fitting *tail*, so a letter drawn straight out
  of the hand sweeping into view is still recognised. Motion is ignored for 0.3 s after the hand
  appears.
* **Hello / Yes**: zig-zag reversal counting on the palm centre (≥ 3 reversals of ≥ 0.32 palm) with
  an open / closed handshape held for ≥ 70 % of the last 1.7 s.

## Composer (`composer.py`)

Turns per-frame guesses into deliberate typing. State per frame:

1. **Vote.** The last 0.28 s of frames; a label needs ≥ 60 % of the votes (and ≥ 3 frames) to become
   the *stable* label. Frames below the confidence threshold or moving faster than the *motion gate*
   vote for "nothing".
2. **Hold.** `progress = (now − stable_since) / hold_time`; at 1.0 the label is typed.
3. **Re-arm.** After typing, the same letter is blocked until the stable label changes, the hand
   leaves, or the hold continues for `repeat_delay` (double letters).
4. **Space.** Hand absent for `space_timeout` after something was typed → exactly one space (and a
   lone lowercase `i` becomes `I`).
5. **Motion events** (J, Z, words) type immediately, clear the votes, and lock static signs for a
   second. The static shape you were *still holding* must be released before it can type, so
   finishing a J does not also type an "I".

Casing is context-aware (`smart` mode capitalises after `. ! ?` and at the start).

## Synthetic hand (`synthetic.py`) and the demo signer (`demo.py`)

A small forward-kinematics model: rest knuckle positions and bone lengths from typical adult
proportions, three flexion angles + a spread angle per finger, and a FABRIK IK solve for the
thumb to a target tip position (e.g. "touching the index tip", "across the front of the fist").
It yields exactly the 21 landmarks MediaPipe produces. Each letter has a pose. It powers the tests,
the demo mode and the reference charts. The poses are approximations built from anatomical
descriptions, not measurements of real signers.

## Extending

* **A new static sign.** Add a pose to `synthetic.py` (for tests/reference), a rule function to
  `rules.py`, and a tip to `reference.TIPS`. The randomised accuracy test will tell you whether it
  collides with existing letters.
* **A new motion.** Add a matcher next to `match_j` / `match_z` and route it in
  `MotionRecognizer.update`.
* **A different tracker.** Anything that returns `HandObservation`s works; see `tracker.py`.
