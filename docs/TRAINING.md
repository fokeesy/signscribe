# Training on your own hand

The built-in geometry rules work without any setup. A **personal model** learns from *your* hand
and camera and is blended with the rules. It is the single most effective way to raise accuracy,
especially for the letters that differ only by where the thumb tucks.

## Guided calibration (recommended)

1. Open the **Train** tab and press **Start guided calibration**.
2. For each letter you will see its reference skeleton and a 3-second countdown. Make the sign,
   hold it, and **turn your hand slightly** while about 70 frames are recorded (roughly two seconds).
3. Nothing is typed into your sentence while calibrating.
4. When the run ends, press **Train model** (you are offered it automatically). Training takes a
   few seconds on a CPU.
5. The Train tab reports accuracy on frames held out from training and lists the weakest signs.
   Re-record those with **Record one sign** and train again.

J and Z reuse the I and pointing shapes, so they are not recorded separately. Their motion is
recognised geometrically.

Tips:

* Sit and light the scene as you will when using it.
* Vary what you can vary: distance, small rotations, slight finger tension. Do not vary the sign.
* Recording both hands is not needed; the features are mirror-invariant, so a model trained on one
  hand works for the other.
* More data beats more epochs. 70 frames × 25 signs is plenty to start.

## Custom signs

Type your own label (for example `THANK YOU`) in *Record one sign*. Multi-character labels are typed
as **words** ("Thank you "). They are static hand shapes: the model matches the handshape, not a
movement.

## From a folder of images

If you have a labelled image dataset in the common layout

```
dataset/
  A/  img001.jpg  img002.jpg ...
  B/  ...
```

run

```bash
signscribe train --images dataset --max-per-class 300 --epochs 80 --out my_model.npz
```

Hands are detected with MediaPipe in still-image mode and their landmarks extracted (images with no
detected hand are skipped), then the same trainer is used. Copy the resulting `.npz` to your data
folder as `model.npz` to use it. Check the licence of any dataset before using or redistributing a
model trained on it.

## Where things are stored

| File | Contents |
|---|---|
| `samples.npz` | your recorded landmarks (21 × 3 numbers per frame + label), no images |
| `model.npz` | the trained network (plain arrays, no pickle) |
| `settings.json` | your settings |

in `%APPDATA%\SignScribe` (Windows), `~/Library/Application Support/SignScribe` (macOS) or
`~/.local/share/signscribe` (Linux). Set `SIGNSCRIBE_HOME` to use another folder. Delete samples or
the model from the Train tab at any time.

## How accuracy is reported

Each sign's frames are split into contiguous blocks and about 20 % of the blocks are held out.
Consecutive frames are near-duplicates, so a random split would leak; block splitting is far more
honest but is still optimistic compared with a completely new session. If accuracy looks great in
the tab but the live result is poor, record again in your real conditions.
