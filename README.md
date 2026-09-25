# webcam-teleop

<p align="center">
  <img src="docs/artifacts/move.gif" width="600" alt="Webcam hand tracking controlling the simulated SO-101 arm">
</p>

Control a simulated [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot
arm in MuJoCo by moving your hand in front of a plain laptop webcam. No depth
camera, no real hardware, no GPU required.

```
webcam (OpenCV) -> MediaPipe hand landmarks -> metric hand pose
                 -> clutch + smoothing + mirror retargeting
                 -> differential IK (mink)
                 -> MuJoCo simulated SO-101
```

This is a from-scratch, simplified take on the same idea as
[guptabhishekumar/handrobot](https://github.com/guptabhishekumar/handrobot)
and [ReenaCatherine/SO101-Gesture-Teleoperation](https://github.com/ReenaCatherine/SO101-Gesture-Teleoperation):
webcam-driven hand tracking teleoperating a simulated SO-101 arm. The SO-101
MJCF model and meshes in `assets/so101/` are vendored from handrobot (MIT) /
[TheRobotStudio's SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache 2.0) — see `assets/so101/LICENSE` and `assets/so101/README.md`.

## Setup

Requires Python 3.12 (MediaPipe does not publish wheels for 3.13+).

```bash
uv venv --python 3.12 .venv
uv pip install -e .
./scripts/fetch_models.sh   # downloads the MediaPipe hand landmarker model
```

## Run

```bash
.venv/bin/python -m webcam_teleop.teleop        # uses the first detected camera
.venv/bin/python -m webcam_teleop.teleop 1      # or pick a specific camera index (0, 1, 2, ...)
```

The number is a camera device index — the same one the UI's own **Camera**
dropdown lists, and the same one `scripts/list_cameras.py` reports. You can
also switch cameras later from the dropdown without restarting.

This opens a [Dear PyGui](https://github.com/hoffstadt/DearPyGui) window: the
webcam feed (with hand landmarks) and the simulated arm side by side, with
controls underneath:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  webcam-teleop                                                   [ ✕ ]   │
├──────────────────────────────────────┬───────────────────────────────────┤
│                                      │                                   │
│           WEBCAM FEED                │           SIMULATED ARM           │
│        (hand landmarks drawn         │       (drag to orbit, scroll      │
│            on top)                   │             to zoom)              │
│                                      │.                                  │
├──────────────────────────────────────┴───────────────────────────────────┤
│  clutch released                            no hand in frame             │
│                                                                          │
│  Camera: [ 1 ▾ ]     [ Engage Clutch (C) ]  [ Reset View ] [ Quit (Q) ]  │
│                                                                          │
│  Sensitivity: ├─────────●───────────┤                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Camera** dropdown — lists every detected camera index; pick whichever one
  actually shows you (see the placeholder-camera note below).
- **Engage/Disengage Clutch** button, or press **c** — like lifting a mouse:
  disengage to reposition your hand without dragging the arm, then re-engage
  wherever's convenient.
- **Sensitivity** slider — how far the arm moves per centimetre of hand
  movement.
- **Reset View** — snaps the sim camera back to its default angle.
- **Quit**, or **q** / **Esc**.
- **Left-drag** the sim panel to orbit the camera around the arm; **right-drag**
  or scroll to zoom.

On macOS, the first run will prompt for camera permission; if nothing shows up
in the webcam panel, check System Settings > Privacy & Security > Camera and
fully quit/reopen the terminal. If the webcam panel shows a static gear icon
instead of a real picture, that camera index is a virtual/placeholder device,
not your built-in one — try another entry in the **Camera** dropdown (on many
Macs, device `0` is the placeholder and `1` is the real camera). If you need
to inspect what each device actually sees before picking, run
`scripts/list_cameras.py`, which saves a snapshot from each.

A red ball sits on the floor within reach — it's a real physics object (has
gravity, unlike the translucent green target marker, which just shows where
your hand is currently mapping to), so you can pick it up and it'll actually
fall if you let go, roll if you knock it, etc.

## How the mapping works

- **Position** is relative: moving your hand right/down/toward the camera
  moves the gripper right/down/away, anchored to wherever you engaged the
  clutch. This sidesteps the fact that depth from one webcam is the noisiest
  signal available — it doesn't need to be *accurate*, only *relatively*
  accurate frame to frame.
- **Depth** is recovered from MediaPipe's paired normalized-image and
  metric-world landmarks: a known rigid part of the hand (the palm) has a
  known size in both, so the ratio gives an absolute scale (see
  `src/webcam_teleop/hand_pose.py`).
- **Every channel is smoothed with a One Euro filter** (`src/webcam_teleop/filters.py`),
  which relaxes when the hand is still and tightens when it moves — a fixed
  exponential filter can't do both.
- **Gripper open/close** maps directly from thumb-to-index pinch distance.
- **IK** (`src/webcam_teleop/ik.py`) is a warm-started differential solver (via
  [`mink`](https://github.com/kevinzakka/mink)) that only ever takes a small
  step from the arm's current pose per frame, so a bad detection can't fling
  the arm across the workspace.

## Project layout

```
src/webcam_teleop/
  hand_pose.py   camera geometry, metric hand pose from MediaPipe landmarks
  tracker.py     webcam capture + MediaPipe HandLandmarker wrapper
  filters.py     One Euro filter, angle filter
  retarget.py    clutch + mirror mapping + smoothing -> gripper command
  ik.py          differential IK for the SO-101 (mink)
  sim.py         MuJoCo simulation wrapper
  ui.py          Dear PyGui window: video panels, camera/clutch/gain controls
  teleop.py      main loop
assets/so101/    vendored SO-101 MJCF model + meshes, plus our scene.xml
assets/models/   MediaPipe hand landmarker model (fetched, not committed)
scripts/         fetch_models.sh, list_cameras.py
```

## Not included (possible next steps)

- Recording demonstrations (images + joint states + actions) and training an
  imitation-learning policy (e.g. ACT), so the arm can act without a hand in
  front of the camera.
- Driving a real SO-101 instead of the simulation, e.g. via
  [LeRobot](https://github.com/huggingface/lerobot)'s `Robot` interface.

## License

MIT (see `LICENSE`) for the code in this repo. The vendored SO-101 model in
`assets/so101/` carries its own license — see `assets/so101/LICENSE`.
