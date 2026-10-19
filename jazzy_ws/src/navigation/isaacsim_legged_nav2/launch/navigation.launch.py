# SPDX-License-Identifier: Apache-2.0
"""Nav2 for a legged robot in Isaac Sim: ``ros2 launch isaacsim_legged_nav2 navigation.launch.py robot:=anymal_d map:=...``.

Looks up the robot's footprint, self-filter ranges and controller tuning in params/robot_profiles.yaml and starts
nav2_stack.launch.xml with them. Every other argument is passed through to that file (launch
configurations are shared with included files, so the defaults here must be real values).
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

PACKAGE = "isaacsim_legged_nav2"
TUNING = ("xy_goal_tolerance", "yaw_goal_tolerance", "min_approach_velocity", "rotate_to_heading_angular_vel")
PASSED_THROUGH = ("map", "localization", "params_file", "use_sim_time", "rviz", "initial_x", "initial_y", "initial_yaw")


def load_profiles() -> dict:
    with open(os.path.join(get_package_share_directory(PACKAGE), "params", "robot_profiles.yaml")) as f:
        return yaml.safe_load(f)


def include_stack(context):
    robot = LaunchConfiguration("robot").perform(context)
    profile = load_profiles()[robot]
    namespace = LaunchConfiguration("namespace").perform(context) or robot
    # Nav2's bringup rewrites the params YAML (dropping quotes) before substituting $(var ...), so a
    # bare "[[...]]" would arrive as a list, not the string Nav2 expects: quote it for YAML here.
    arguments = {"namespace": namespace, "footprint": f"'{profile['footprint']}'",
                 "scan_range_min": str(profile["scan_range_min"]),
                 "self_filter_range": str(profile["self_filter_range"])}
    arguments.update({key: str(profile[key]) for key in TUNING})
    arguments.update({name: LaunchConfiguration(name).perform(context) for name in PASSED_THROUGH})
    stack = os.path.join(get_package_share_directory(PACKAGE), "launch", "nav2_stack.launch.xml")
    return [IncludeLaunchDescription(XMLLaunchDescriptionSource(stack), launch_arguments=arguments.items())]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="spot", choices=sorted(load_profiles()),
                              description="Robot profile in params/robot_profiles.yaml"),
        DeclareLaunchArgument("namespace", default_value="", description="Default: the robot name"),
        DeclareLaunchArgument("map", description="Full path to the occupancy-map yaml file"),
        DeclareLaunchArgument("localization", default_value="ground_truth", choices=["ground_truth", "amcl"],
                              description="ground_truth: exact, on every floor; amcl: against the (one-floor) map"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(get_package_share_directory(PACKAGE), "params", "nav2.yaml")),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("initial_x", default_value="0.0", description="AMCL initial pose (map frame)"),
        DeclareLaunchArgument("initial_y", default_value="0.0"),
        DeclareLaunchArgument("initial_yaw", default_value="0.0", description="radians"),
        OpaqueFunction(function=include_stack),
    ])
