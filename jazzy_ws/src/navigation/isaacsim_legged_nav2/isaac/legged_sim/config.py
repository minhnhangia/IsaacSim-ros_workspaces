# SPDX-License-Identifier: Apache-2.0
"""Command-line options. Parsed before Isaac Sim starts, so this module must not import it.

Options whose default is None take the selected robot's value (robots/<robot>.py).
"""

import argparse
import shlex

ROBOTS = ("spot", "anymal_d")  # keep in sync with robots.ROBOTS (not imported: it pulls in Isaac Sim)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A legged robot in Isaac Sim, walking under an Isaac Lab locomotion policy, driven by ROS 2")
    parser.add_argument("--robot", choices=ROBOTS, default="spot",
                        help="spot: flat ground only; anymal_d: also climbs and descends stairs")

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--headless", action="store_true")
    sim.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                     help="physics device (cpu is faster for a single robot)")
    sim.add_argument("--engine", choices=["physx", "newton"], default="physx")
    sim.add_argument("--actuators", choices=["fast", "isaac"], default="fast",
                     help="spot only. fast: its actuator model in one NumPy pass (same model as isaac, much "
                          "cheaper); isaac: Isaac Sim's per-joint ArticulationActuators")
    sim.add_argument("--scene", default="office", help="'office', 'grid', or a USD file to load as the environment")
    sim.add_argument("--deactivate", default=None,
                     help="comma-separated prims to switch off in the scene (default: the scene preset's)")
    sim.add_argument("--spawn", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "YAW_DEG"))
    sim.add_argument("--floor-z", type=float, default=0.0, help="height of the floor the robot spawns on")
    sim.add_argument("--floor-friction", type=float, default=1.0,
                     help="friction of the scene's own surfaces (they have no physics material otherwise, i.e. "
                          "0.5); combined by multiplication with the feet, as in Isaac Lab's training terrain")
    sim.add_argument("--open-stair-doors", action=argparse.BooleanOptionalAction, default=None,
                     help="open the stairwell doors (default: only for robots that climb stairs)")

    ros = parser.add_argument_group("ROS 2 and sensors")
    ros.add_argument("--namespace", default=None, help="ROS namespace (default: the robot name)")
    ros.add_argument("--no-lidar", action="store_true")
    ros.add_argument("--no-camera", action="store_true", help="no D455 stereo cameras (Nav2 does not use them)")
    ros.add_argument("--lidar-config", default="OS0_REV6_128ch10hz512res",
                     help="OS0 sensor variant (the asset's own default, 128x1024, doubles the cloud size)")
    ros.add_argument("--lidar-mount", type=float, nargs=3, default=None, metavar=("X", "Y", "DZ"),
                     help="OS0 position on the body: x, y, and mast height above the body's top surface")
    ros.add_argument("--camera-resolution", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    ros.add_argument("--camera-rate", type=float, default=0.0,
                     help="render and publish the cameras at this rate in Hz (0 = every frame)")

    cmd = parser.add_argument_group("cmd_vel handling")
    cmd.add_argument("--cmd-timeout", type=float, default=0.5, help="sim seconds without cmd_vel before stopping")
    cmd.add_argument("--max-vx", type=float, default=None)
    cmd.add_argument("--min-vx", type=float, default=None)
    cmd.add_argument("--max-vy", type=float, default=None)
    cmd.add_argument("--max-wz", type=float, default=None)
    cmd.add_argument("--deadband-comp", action=argparse.BooleanOptionalAction, default=None,
                     help="scale small commands up to where the policy starts walking (from a stand)")
    cmd.add_argument("--min-cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"),
                     help="deadband compensation target per axis")
    cmd.add_argument("--hold-heading", action=argparse.BooleanOptionalAction, default=None,
                     help="when the command is zero, steer back to the heading held at stop")
    cmd.add_argument("--hold-gain", type=float, default=1.0, help="rad/s of yaw rate per rad of heading error")
    cmd.add_argument("--hold-max-wz", type=float, default=0.3)
    cmd.add_argument("--hold-engage-deg", type=float, default=3.0)
    cmd.add_argument("--hold-release-deg", type=float, default=1.0)

    view = parser.add_argument_group("viewing and testing")
    view.add_argument("--view-offset", type=float, nargs=3, default=None, metavar=("FWD", "LEFT", "UP"),
                      help="initial GUI camera relative to the robot (default: a nearby view with a clear line of sight)")
    view.add_argument("--snapshot-dir", default="", help="save follow-camera and close-up PNGs of the robot here")
    view.add_argument("--snapshot-every", type=float, default=5.0, help="sim seconds between snapshots")
    view.add_argument("--add-box", action="append", default=[], metavar="X,Y,SX,SY,SZ",
                      help="add a static obstacle box standing on the floor")
    view.add_argument("--duration", type=float, default=0.0, help="stop after this many sim seconds (0 = run forever)")
    view.add_argument("--log", default="", help="CSV of per-frame commands and pose")
    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    """Undo two ways ``ros2 launch`` and argparse mangle the arguments.

    ``sim_args:="--headless --spawn 1 2 0"`` arrives as ONE argv entry; split such entries the way a
    shell would. ``--add-box -1.4,5.6,...`` would be read as two options; bind the value to its flag.
    """
    parts = [p for a in argv for p in (shlex.split(a) if a.startswith("--") and " " in a else [a])]
    out = []
    i = 0
    while i < len(parts):
        if parts[i] == "--add-box" and i + 1 < len(parts):
            out.append(f"--add-box={parts[i + 1]}")
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return out


def apply_robot_defaults(args: argparse.Namespace, robot) -> argparse.Namespace:
    """Fills options left at None from the robot's profile (``robot.commands``, mounts, stairs)."""
    profile = robot.commands
    defaults = {"min_vx": profile.min_vx, "max_vx": profile.max_vx, "max_vy": profile.max_vy,
                "max_wz": profile.max_wz, "min_cmd": list(profile.min_command),
                "deadband_comp": profile.deadband_compensation, "hold_heading": profile.hold_heading,
                "lidar_mount": list(robot.lidar_mount), "open_stair_doors": robot.climbs_stairs,
                "namespace": robot.name}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the options; unknown ones (Kit's own, for example) are reported and ignored."""
    args, unknown = build_parser().parse_known_args(normalize_argv(argv))
    if unknown:
        print(f"LEGGED_SIM: ignoring unknown arguments {unknown}", flush=True)
    return args
