# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-09-24

First release.

### Added
- Live hand tracking with MediaPipe Hands: 21 image + 21 metric world landmarks per hand, up to two hands,
  left/right detection, One Euro smoothing, dominant-hand selection.
- Adaptive exposure (`LightingAdapter`): metering on the hand, hysteretic auto-gamma, CLAHE, and a periodic
  enhanced-frame retry when no hand is found.
- Rotation-, scale- and handedness-invariant pose features: joint angles, finger extension, spread, thumb
  placement, roll / pitch / yaw, palm speed.
- Rule-based recogniser for A-Z (J and Z by motion) and the "I love you" handshape.
- Motion recogniser for J, Z, wave (Hello) and fist nod (Yes).
- Personal model: NumPy MLP trained on your own recorded landmarks, blended with the rules; guided calibration
  wizard; custom labels; training from a labelled image folder (`signscribe train --images`).
- Composer: majority voting, hold-to-type, motion gate, re-arming for double letters, spaces on hand-out,
  smart capitalisation, pronoun "I" fix.
- Tkinter GUI: live video with skeleton overlay, big letter readout with progress ring, finger bars,
  rotation, lighting status, sentence box with editing / copy / save, Train, Settings and Help tabs.
- Optional typing into other applications (`pip install "signscribe[typing]"`).
- Camera-free demo mode with a procedural virtual signer, also used for the automated tests.
- CLI: `gui`, `headless`, `train`, `reference`, `doctor`.
- 130+ automated tests, GitHub Actions CI (Linux + Windows, Python 3.10-3.12).
