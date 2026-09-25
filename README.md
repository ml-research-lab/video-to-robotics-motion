# webcam-teleop

<p align="center">
  <img src="docs/artifacts/move.gif" width="600" alt="Webcam hand tracking controlling the simulated SO-101 arm">
</p>

Control a simulated robot arm — or a dexterous hand, finger by finger — in
MuJoCo by moving your hand in front of a plain laptop webcam. No depth
camera, no real hardware, no GPU required.

```
webcam (OpenCV) -> MediaPipe hand landmarks -> metric hand pose
                 -> clutch + smoothing + mirror retargeting
                 -> differential IK (mink), one target per gripper/fingertip
                 -> MuJoCo simulated robot (any in webcam_teleop.robots.ROBOTS)
```

This started as a from-scratch, simplified take on the same idea as
[guptabhishekumar/handrobot](https://github.com/guptabhishekumar/handrobot)
and [ReenaCatherine/SO101-Gesture-Teleoperation](https://github.com/ReenaCatherine/SO101-Gesture-Teleoperation):
webcam-driven hand tracking teleoperating a simulated SO-101 arm. It now
drives several more arms and dexterous hands too (below) — each fetched on
first use from Google DeepMind's [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
via the [`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py)
package (Apache-2.0), cached locally after that. Only the SO-101's MJCF model
and meshes are vendored in this repo, from handrobot (MIT) /
[TheRobotStudio's SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache 2.0) — see `assets/so101/LICENSE` and `assets/so101/README.md`.

## Robots

All free to use; see each one's license before any commercial use.

| Robot | `--robot` name | Kind | DoF | License |
|---|---|---|---|---|
| SO-101 (default) | `so101` | arm + gripper | 6 | Apache-2.0 |
| Franka Panda | `panda` | arm + gripper | 7+1 | Apache-2.0 |
| UFactory xArm7 | `xarm7` | arm + gripper | 7+1 | BSD-3-Clause |
| Trossen ViperX 300s | `viperx` | arm + gripper | 6+1 | BSD-3-Clause |
| Shadow Hand E3M5 | `shadow_hand` | dexterous hand | 24 | Apache-2.0 |
| LEAP Hand | `leap_hand` | dexterous hand | 16 | MIT |
| Allegro Hand V3 | `allegro_hand` | dexterous hand | 16 | BSD-2-Clause |

An **arm** is driven the way the SO-101 always was: hand position moves the
gripper, pinch opens/closes it. A **dexterous hand** is mounted at a fixed
point in the scene and driven finger by finger — each of your fingertips
(relative to your wrist) drives the matching robot fingertip directly, live,
no clutch-anchoring needed, since that mapping only needs *relative*
position (see `src/webcam_teleop/hand_retarget.py`).

Not every free Menagerie arm made the cut here: UR5e ships with no gripper in
its own model (would need attaching Menagerie's separate Robotiq 2F-85), and
PiPER's model has no ready-made TCP site, both solvable but out of scope for
now. Also free and MuJoCo-ready, for a future pass: the Sharpa Wave hand (22
DoF, Apache-2.0), Aero Hand Open (16 DoF, Apache-2.0), and full humanoids
(Unitree G1, PNDbotics Adam-lite) — a humanoid would need deciding which one
arm a single tracked hand should drive.

## Setup

Requires Python 3.12 (MediaPipe does not publish wheels for 3.13+).

```bash
uv venv --python 3.12 .venv
uv pip install -e .
./scripts/fetch_models.sh   # downloads the MediaPipe hand landmarker model
```

## Run

```bash
.venv/bin/python -m webcam_teleop.teleop                        # SO-101, first detected camera
.venv/bin/python -m webcam_teleop.teleop 1                      # a specific camera index (0, 1, 2, ...)
.venv/bin/python -m webcam_teleop.teleop --robot shadow_hand 1  # a different robot + camera
```

`--robot` accepts any name from the table above. The first time you pick a
given robot it's fetched from MuJoCo Menagerie (needs internet); after that
it's cached and works offline. The camera device number is the same one the
UI's own **Camera** dropdown lists, and the same one `scripts/list_cameras.py`
reports. Both the robot and the camera can also be switched later from their
dropdowns without restarting.

This opens a [Dear PyGui](https://github.com/hoffstadt/DearPyGui) window: the
webcam feed (with hand landmarks) and the simulated robot side by side, with
controls underneath:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  webcam-teleop                                                   [ ✕ ]   │
├──────────────────────────────────────┬───────────────────────────────────┤
│                                      │                                   │
│           WEBCAM FEED                │          SIMULATED ROBOT          │
│        (hand landmarks drawn         │       (drag to orbit, scroll      │
│            on top)                   │             to zoom)              │
│                                      │.                                  │
├──────────────────────────────────────┴───────────────────────────────────┤
│  clutch released                            no hand in frame             │
│                                                                          │
│  Robot: [ so101 ▾ ]  Camera: [ 1 ▾ ]  [ Engage Clutch (C) ]  [ Reset ]  │
│                                                          [ Quit (Q) ]   │
│  Sensitivity: ├─────────●───────────┤                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Robot** dropdown — switches robots on the fly (any name from the table
  above); the sim rebuilds, the clutch disengages.
- **Camera** dropdown — lists every detected camera index; pick whichever one
  actually shows you (see the placeholder-camera note below).
- **Engage/Disengage Clutch** button, or press **c** — like lifting a mouse:
  disengage to reposition your hand without dragging the robot, then
  re-engage wherever's convenient. (A dexterous hand doesn't need this to
  avoid drift the way an arm does, but it still gates when tracking is live.)
- **Sensitivity** slider — how far the gripper/fingertips move per centimetre
  of hand movement.
- **Reset View** — snaps the sim camera back to its default angle.
- **Quit**, or **q** / **Esc**.
- **Left-drag** the sim panel to orbit the camera around the robot; **right-drag**
  or scroll to zoom.

On macOS, the first run will prompt for camera permission; if nothing shows up
in the webcam panel, check System Settings > Privacy & Security > Camera and
fully quit/reopen the terminal. If the webcam panel shows a static gear icon
instead of a real picture, that camera index is a virtual/placeholder device,
not your built-in one — try another entry in the **Camera** dropdown (on many
Macs, device `0` is the placeholder and `1` is the real camera). If you need
to inspect what each device actually sees before picking, run
`scripts/list_cameras.py`, which saves a snapshot from each.

For an arm, a red ball sits on the floor within reach — a real physics
object (has gravity, friction, collisions), so you can pick it up and it'll
actually fall if you let go, roll if you knock it, etc.

## How the mapping works

- **An arm's position mapping is relative**: moving your hand right/down/
  toward the camera moves the gripper right/down/away, anchored to wherever
  you engaged the clutch. This sidesteps the fact that depth from one webcam
  is the noisiest signal available — it doesn't need to be *accurate*, only
  *relatively* accurate frame to frame. **A dexterous hand's fingertip
  mapping needs no such anchor**: each fingertip's position *relative to the
  wrist* already cancels out the unknown absolute translation, so it maps
  straight through live (see `src/webcam_teleop/hand_retarget.py`).
- **Depth** (used only by arms) is recovered from MediaPipe's paired
  normalized-image and metric-world landmarks: a known rigid part of the
  hand (the palm) has a known size in both, so the ratio gives an absolute
  scale (see `src/webcam_teleop/hand_pose.py`).
- **Every channel is smoothed with a One Euro filter** (`src/webcam_teleop/filters.py`),
  which relaxes when the hand is still and tightens when it moves — a fixed
  exponential filter can't do both.
- **An arm's gripper open/close** maps directly from thumb-to-index pinch
  distance.
- **IK** (`src/webcam_teleop/ik.py`) is a warm-started differential solver
  (via [`mink`](https://github.com/kevinzakka/mink)) that only ever takes a
  small step from the current pose per frame, so a bad detection can't fling
  the robot across the workspace. An arm solves one target (the gripper
  site); a dexterous hand solves one per tracked fingertip at once.

## Project layout

```
src/webcam_teleop/
  robots.py         the robot registry: RobotSpec per arm/hand, how each is
                     fetched (vendored for SO-101, MuJoCo Menagerie for the
                     rest) and mounted into the scene
  hand_pose.py       camera geometry, metric hand pose from MediaPipe landmarks
  tracker.py         webcam capture + MediaPipe HandLandmarker wrapper
  filters.py         One Euro filter, angle filter
  retarget.py        arm: clutch + mirror mapping + smoothing -> gripper command
  hand_retarget.py   dexterous hand: per-fingertip mirror mapping + smoothing
  ik.py              differential IK over one or more frame targets (mink)
  controllers.py     ArmController / HandController: wire retarget + IK + ctrl
  sim.py             assembles + simulates the scene for any RobotSpec
  ui.py              Dear PyGui window: video panels, robot/camera/clutch/gain
  teleop.py          main loop
assets/so101/        vendored SO-101 MJCF model + meshes (everything else
                     fetched from MuJoCo Menagerie into robot_descriptions'
                     own cache, not committed here)
assets/models/       MediaPipe hand landmarker model (fetched, not committed)
scripts/             fetch_models.sh, list_cameras.py
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
