# SPDX-License-Identifier: Apache-2.0
from legged_sim.config import apply_robot_defaults, normalize_argv, parse_args
from legged_sim.robots.base import CommandProfile


def test_ros2_launch_single_token_is_split():
    assert normalize_argv(["--headless --spawn 1 2 90"]) == ["--headless", "--spawn", "1", "2", "90"]


def test_add_box_value_bound_to_its_flag():
    assert normalize_argv(["--add-box", "-1.4,5.6,0.6,0.6,0.8"]) == ["--add-box=-1.4,5.6,0.6,0.6,0.8"]


def test_parse_from_launch_style_argv():
    args = parse_args(["--headless --add-box -1.4,5.6,0.6,0.6,0.8 --spawn 1 2 90"])
    assert args.headless
    assert args.add_box == ["-1.4,5.6,0.6,0.6,0.8"]
    assert args.spawn == [1.0, 2.0, 90.0]


def test_defaults():
    args = parse_args([])
    assert (args.robot, args.device, args.actuators, args.scene) == ("spot", "cpu", "fast", "office")
    assert args.camera_rate == 0.0
    # robot-dependent options stay unset until the robot is known
    assert args.namespace is None and args.max_vx is None and args.hold_heading is None


class FakeRobot:
    name = "fake"
    lidar_mount = (0.1, 0.0, 0.2)
    climbs_stairs = True
    commands = CommandProfile(min_vx=-0.5, max_vx=1.0, max_vy=0.6, max_wz=0.9, min_command=(0.2, 0.2, 0.3),
                              deadband_compensation=True, hold_heading=False)


def test_robot_defaults_fill_only_unset_options():
    args = apply_robot_defaults(parse_args(["--max-vx", "0.7", "--no-open-stair-doors"]), FakeRobot)
    assert args.max_vx == 0.7  # set on the command line: kept
    assert (args.min_vx, args.max_vy, args.max_wz) == (-0.5, 0.6, 0.9)
    assert args.min_cmd == [0.2, 0.2, 0.3] and args.lidar_mount == [0.1, 0.0, 0.2]
    assert args.deadband_comp is True and args.hold_heading is False
    assert args.open_stair_doors is False and args.namespace == "fake"
