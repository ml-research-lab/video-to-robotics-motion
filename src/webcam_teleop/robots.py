"""Which robot to drive, and everything the rest of the code needs to know.

Two kinds:

* **arm** -- an arm with a gripper, driven the way the SO-101 already was:
  hand position -> gripper position (via ``retarget.HandToGripper`` and
  ``ik.MultiFrameIK`` solving one site), pinch -> gripper open/close.
* **hand** -- a dexterous hand mounted at a fixed point in the scene, driven
  finger by finger (via ``hand_retarget.HandToFingers`` and the same
  ``ik.MultiFrameIK`` solving one body frame per fingertip).

Every non-SO-101 model is fetched and cached by the ``robot_descriptions``
package (Apache-2.0) from Google DeepMind's MuJoCo Menagerie, so nothing but
SO-101 needs to be vendored in this repo. First use of a given robot needs
internet access to populate that cache; every use after that is offline.

Gripper open/closed ctrl values are given explicitly, not inferred from
``actuator_ctrlrange``, because which end of that range is "open" is not
consistent across manufacturers -- xArm7's is the reverse of the others.
Confirmed for each robot here by measuring the actual fingertip separation at
both ends of the range.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

from webcam_teleop.config import WorkspaceBox
from webcam_teleop.paths import SO101_ARM_XML

_DEFAULT_CAMERA = dict(lookat=(0.2, 0.0, 0.08), distance=0.7, azimuth=140.0, elevation=-20.0)


def _so101_spec():
    import mujoco

    return mujoco.MjSpec.from_file(str(SO101_ARM_XML))


def _menagerie_spec(module_name: str, filename: str | None = None):
    """A loader for a Menagerie robot: fetches/caches it, returns a factory.

    ``module_name`` is the ``robot_descriptions`` package name (e.g.
    ``"panda_mj_description"``); ``filename`` overrides which MJCF in that
    package to load, for models with several variants (left/right hand,
    with/without gripper, ...).
    """

    def _load():
        import mujoco
        from robot_descriptions.loaders.mujoco import load_robot_description

        load_robot_description(module_name)  # ensures the file is cached
        module = importlib.import_module(f"robot_descriptions.{module_name}")
        path = Path(module.PACKAGE_PATH) / filename if filename else Path(module.MJCF_PATH)
        return mujoco.MjSpec.from_file(str(path))

    return _load


@dataclass(frozen=True)
class ArmSpec:
    """Everything an arm-with-gripper controller needs."""

    tcp_site: str
    n_arm_joints: int
    gripper_open_ctrl: float
    gripper_closed_ctrl: float
    workspace: WorkspaceBox
    home_qpos: tuple[float, ...]
    """Arm joint angles (radians), not including the gripper."""
    tcp_offset_body: str | None = None
    """If the MJCF has no TCP site, add one at ``tcp_offset_pos`` on this body."""
    tcp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class HandSpec:
    """Everything a dexterous-hand controller needs."""

    #: Human finger name -> the robot body whose origin the fingertip should track.
    fingertip_bodies: dict[str, str]
    mount_pos: tuple[float, float, float]
    """Where the hand's wrist/forearm body is fixed in the scene."""
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: float = 1.0
    """Robot fingertip-to-wrist reach divided by a typical human hand's."""
    #: Menagerie models where one actuator drives a tendon spanning several
    #: joints (Shadow Hand's coupled distal joints); ctrl for that actuator is
    #: the sum of those joints' solved angles. Empty for fully-actuated hands.
    tendon_actuators: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotSpec:
    name: str
    display_name: str
    kind: str  # "arm" | "hand"
    license: str
    load_spec: object  # Callable[[], mujoco.MjSpec]
    arm: ArmSpec | None = None
    hand: HandSpec | None = None
    camera: dict = field(default_factory=lambda: dict(_DEFAULT_CAMERA))


#: Name prefix MjSpec.attach() applies to everything from the attached
#: robot (joints, bodies, sites, actuators). Any code that looks up a
#: frame by name on a *scene or mounted-model* level -- as opposed to
#: through RobotSim's own site_position()/body_position(), which already
#: apply it -- needs to prepend this.
ROBOT_PREFIX = "robot_"


def _mount_pose(spec: "RobotSpec") -> tuple[list[float], list[float]]:
    if spec.kind == "hand" and spec.hand is not None:
        return list(spec.hand.mount_pos), list(spec.hand.mount_quat)
    return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]


def load_and_prepare(spec: "RobotSpec"):
    """Load a fresh copy of the robot's bare MJCF and add anything it lacks.

    Adds the TCP site to models that don't ship one (Panda). Each call
    returns an independent MjSpec, so this is safe to call once per user
    (an IK solver's bare model, the copy attached into the full scene) --
    they never share or fight over the same spec object.
    """
    robot = spec.load_spec()
    if spec.kind == "arm" and spec.arm is not None and spec.arm.tcp_offset_body:
        body = robot.body(spec.arm.tcp_offset_body)
        body.add_site(name=spec.arm.tcp_site, pos=list(spec.arm.tcp_offset_pos))
    return robot


def attach_robot(base_spec, robot_mjspec, spec: "RobotSpec") -> None:
    """Attach an already-``load_and_prepare``'d robot into ``base_spec``.

    Copies the robot's own tuned physics options onto ``base_spec`` (attach()
    otherwise leaves MjSpec's generic defaults in place, silently), and
    mounts it at the spec's fixed pose -- identity for an arm, a hand's own
    ``mount_pos``/``mount_quat`` for a dexterous hand.
    """
    for field_name in ("timestep", "integrator", "cone", "iterations", "ls_iterations", "impratio"):
        setattr(base_spec.option, field_name, getattr(robot_mjspec.option, field_name))

    mount = base_spec.worldbody.add_body(name="mount")
    pos, quat = _mount_pose(spec)
    attach_site = mount.add_site(name="attach", pos=pos, quat=quat)
    base_spec.attach(robot_mjspec, prefix=ROBOT_PREFIX, site=attach_site)


def mounted_bare_model(spec: "RobotSpec"):
    """Compile just the robot, mounted at its scene pose, with no floor/ball.

    Used by IK solvers, which must reason in the same world frame the full
    scene (built separately in sim.py, sharing this mount logic) places the
    robot in -- for an arm, mounted at identity, this is moot; for a hand
    fixed away from the origin, it is not.
    """
    import mujoco

    robot = load_and_prepare(spec)
    base = mujoco.MjSpec()
    attach_robot(base, robot, spec)
    return base.compile()


SO101 = RobotSpec(
    name="so101",
    display_name="SO-101",
    kind="arm",
    license="Apache-2.0 (TheRobotStudio SO-ARM100)",
    load_spec=_so101_spec,
    arm=ArmSpec(
        tcp_site="gripperframe",
        n_arm_joints=5,
        gripper_open_ctrl=1.745329,
        gripper_closed_ctrl=-0.174533,
        workspace=WorkspaceBox(x=(0.14, 0.30), y=(-0.16, 0.16), z=(0.02, 0.20)),
        home_qpos=(0.0, -1.57, 1.57, 0.9, 0.0),
    ),
    camera=dict(_DEFAULT_CAMERA),
)

PANDA = RobotSpec(
    name="panda",
    display_name="Franka Panda",
    kind="arm",
    license="Apache-2.0 (Franka Emika, via MuJoCo Menagerie)",
    load_spec=_menagerie_spec("panda_mj_description"),
    arm=ArmSpec(
        tcp_site="attached_tcp",
        n_arm_joints=7,
        gripper_open_ctrl=255.0,
        gripper_closed_ctrl=0.0,
        workspace=WorkspaceBox(x=(0.32, 0.62), y=(-0.32, 0.32), z=(0.04, 0.40)),
        home_qpos=(0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853),
        tcp_offset_body="hand",
        tcp_offset_pos=(0.0, 0.0, 0.1034),  # measured: finger-pad depth at this model's home pose
    ),
    camera=dict(lookat=(0.4, 0.0, 0.15), distance=1.1, azimuth=140.0, elevation=-20.0),
)

XARM7 = RobotSpec(
    name="xarm7",
    display_name="UFactory xArm7",
    kind="arm",
    license="BSD-3-Clause (UFactory, via MuJoCo Menagerie)",
    load_spec=_menagerie_spec("xarm7_mj_description"),
    arm=ArmSpec(
        tcp_site="link_tcp",
        n_arm_joints=7,
        gripper_open_ctrl=0.0,  # reversed vs. the others: measured 141mm gap at ctrl=0
        gripper_closed_ctrl=255.0,  # vs. 58mm gap at ctrl=255
        workspace=WorkspaceBox(x=(0.25, 0.55), y=(-0.25, 0.25), z=(0.05, 0.45)),
        home_qpos=(0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0),
    ),
    camera=dict(lookat=(0.35, 0.0, 0.15), distance=1.0, azimuth=140.0, elevation=-20.0),
)

VIPERX = RobotSpec(
    name="viperx",
    display_name="Trossen ViperX 300s",
    kind="arm",
    license="BSD-3-Clause (Trossen Robotics, via MuJoCo Menagerie)",
    load_spec=_menagerie_spec("viper_mj_description"),
    arm=ArmSpec(
        tcp_site="pinch",
        n_arm_joints=6,
        gripper_open_ctrl=0.057,
        gripper_closed_ctrl=0.021,
        workspace=WorkspaceBox(x=(0.20, 0.45), y=(-0.20, 0.20), z=(0.05, 0.35)),
        home_qpos=(0.0, -0.96, 1.16, 0.0, -0.3, 0.0),
    ),
    camera=dict(lookat=(0.25, 0.0, 0.12), distance=0.85, azimuth=140.0, elevation=-20.0),
)

SHADOW_HAND = RobotSpec(
    name="shadow_hand",
    display_name="Shadow Hand E3M5 (24 DoF)",
    kind="hand",
    license="Apache-2.0 (Shadow Robot, via MuJoCo Menagerie)",
    load_spec=_menagerie_spec("shadow_hand_mj_description", "right_hand.xml"),
    hand=HandSpec(
        fingertip_bodies={
            "thumb": "rh_thdistal",
            "index": "rh_ffdistal",
            "middle": "rh_mfdistal",
            "ring": "rh_rfdistal",
            "pinky": "rh_lfdistal",
        },
        # Higher than the other hands: Shadow Hand's forearm+wrist chain
        # before the palm is much longer, and this mount height is measured
        # to clear the floor with it.
        mount_pos=(0.2, 0.0, 0.45),
        mount_quat=(0.7071, 0.0, 0.7071, 0.0),  # palm facing the operator
        scale=1.35,
        tendon_actuators={
            "rh_A_FFJ0": ("rh_FFJ1", "rh_FFJ2"),
            "rh_A_MFJ0": ("rh_MFJ1", "rh_MFJ2"),
            "rh_A_RFJ0": ("rh_RFJ1", "rh_RFJ2"),
            "rh_A_LFJ0": ("rh_LFJ1", "rh_LFJ2"),
        },
    ),
    camera=dict(lookat=(0.2, 0.0, 0.35), distance=0.55, azimuth=140.0, elevation=-10.0),
)

LEAP_HAND = RobotSpec(
    name="leap_hand",
    display_name="LEAP Hand (16 DoF)",
    kind="hand",
    license="MIT (via MuJoCo Menagerie)",
    load_spec=_menagerie_spec("leap_hand_mj_description", "right_hand.xml"),
    hand=HandSpec(
        fingertip_bodies={
            "thumb": "th_ds",
            "index": "if_ds",
            "middle": "mf_ds",
            "ring": "rf_ds",
        },
        mount_pos=(0.2, 0.0, 0.25),
        mount_quat=(0.7071, 0.0, 0.7071, 0.0),
        scale=1.1,
    ),
    camera=dict(lookat=(0.2, 0.0, 0.25), distance=0.4, azimuth=140.0, elevation=-10.0),
)

ALLEGRO_HAND = RobotSpec(
    name="allegro_hand",
    display_name="Allegro Hand V3 (16 DoF)",
    kind="hand",
    license="BSD-2-Clause (Wonik Robotics, via MuJoCo Menagerie)",
    load_spec=_menagerie_spec("allegro_hand_mj_description", "right_hand.xml"),
    hand=HandSpec(
        fingertip_bodies={
            "thumb": "th_tip",
            "index": "ff_tip",
            "middle": "mf_tip",
            "ring": "rf_tip",
        },
        mount_pos=(0.2, 0.0, 0.25),
        mount_quat=(0.7071, 0.0, 0.7071, 0.0),
        scale=1.3,
    ),
    camera=dict(lookat=(0.2, 0.0, 0.25), distance=0.45, azimuth=140.0, elevation=-10.0),
)

ROBOTS: dict[str, RobotSpec] = {
    spec.name: spec
    for spec in (SO101, PANDA, XARM7, VIPERX, SHADOW_HAND, LEAP_HAND, ALLEGRO_HAND)
}
DEFAULT_ROBOT = "so101"


def get_robot(name: str | None = None) -> RobotSpec:
    key = (name or DEFAULT_ROBOT).lower()
    if key not in ROBOTS:
        raise KeyError(f"unknown robot {name!r}; choose from {sorted(ROBOTS)}")
    return ROBOTS[key]
