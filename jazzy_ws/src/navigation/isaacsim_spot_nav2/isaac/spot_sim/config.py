# SPDX-License-Identifier: Apache-2.0
"""Command-line options. Parsed before Isaac Sim starts, so this module must not import it."""

import argparse
import shlex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spot in Isaac Sim, walking under the bundled Isaac Lab locomotion policy, driven by ROS 2")

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--headless", action="store_true")
    sim.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                     help="physics device (cpu is faster for a single robot)")
    sim.add_argument("--engine", choices=["physx", "newton"], default="physx")
    sim.add_argument("--actuators", choices=["fast", "isaac"], default="fast",
                     help="fast: Spot's actuator model in one NumPy pass (same model as isaac, much cheaper); "
                          "isaac: Isaac Sim's per-joint ArticulationActuators")
    sim.add_argument("--scene", default="office", help="'office', 'grid', or a USD file to load as the environment")
    sim.add_argument("--deactivate", default=None,
                     help="comma-separated prims to switch off in the scene (default: the scene preset's)")
    sim.add_argument("--spawn", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "YAW_DEG"))
    sim.add_argument("--floor-z", type=float, default=0.0, help="height of the floor Spot stands on")

    ros = parser.add_argument_group("ROS 2 and sensors")
    ros.add_argument("--namespace", default="spot")
    ros.add_argument("--no-lidar", action="store_true")
    ros.add_argument("--no-camera", action="store_true", help="no D455 stereo cameras (Nav2 does not use them)")
    ros.add_argument("--lidar-config", default="OS0_REV6_128ch10hz512res",
                     help="OS0 sensor variant (the asset's own default, 128x1024, doubles the cloud size)")
    ros.add_argument("--lidar-mount", type=float, nargs=3, default=[0.0, 0.0, 0.10], metavar=("X", "Y", "DZ"),
                     help="OS0 position on the body: x, y, and mast height above the body's top surface")
    ros.add_argument("--camera-resolution", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    ros.add_argument("--camera-rate", type=float, default=0.0,
                     help="render and publish the cameras at this rate in Hz (0 = every frame)")

    cmd = parser.add_argument_group("cmd_vel handling")
    cmd.add_argument("--cmd-timeout", type=float, default=0.5, help="sim seconds without cmd_vel before stopping")
    cmd.add_argument("--max-vx", type=float, default=1.5)
    cmd.add_argument("--min-vx", type=float, default=-1.0)
    cmd.add_argument("--max-vy", type=float, default=0.8)
    cmd.add_argument("--max-wz", type=float, default=1.2)
    cmd.add_argument("--deadband-comp", action=argparse.BooleanOptionalAction, default=True,
                     help="scale small commands up to where the policy starts walking (from a stand)")
    cmd.add_argument("--min-cmd", type=float, nargs=3, default=[0.30, 0.22, 0.32], metavar=("VX", "VY", "WZ"),
                     help="deadband compensation target per axis")
    cmd.add_argument("--hold-heading", action=argparse.BooleanOptionalAction, default=True,
                     help="when the command is zero, steer back to the heading held at stop")
    cmd.add_argument("--hold-gain", type=float, default=1.0, help="rad/s of yaw rate per rad of heading error")
    cmd.add_argument("--hold-max-wz", type=float, default=0.3)
    cmd.add_argument("--hold-engage-deg", type=float, default=3.0)
    cmd.add_argument("--hold-release-deg", type=float, default=1.0)

    view = parser.add_argument_group("viewing and testing")
    view.add_argument("--view-offset", type=float, nargs=3, default=None, metavar=("FWD", "LEFT", "UP"),
                      help="initial GUI camera relative to Spot (default: a nearby view with a clear line of sight)")
    view.add_argument("--snapshot-dir", default="", help="save follow-camera and close-up PNGs of Spot here")
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the options; unknown ones (Kit's own, for example) are reported and ignored."""
    args, unknown = build_parser().parse_known_args(normalize_argv(argv))
    if unknown:
        print(f"SPOTSIM: ignoring unknown arguments {unknown}", flush=True)
    return args
