# SPDX-License-Identifier: Apache-2.0
from spot_sim.config import normalize_argv, parse_args


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
    assert (args.device, args.actuators, args.scene, args.namespace) == ("cpu", "fast", "office", "spot")
    assert args.deadband_comp and args.hold_heading
    assert args.camera_rate == 0.0
