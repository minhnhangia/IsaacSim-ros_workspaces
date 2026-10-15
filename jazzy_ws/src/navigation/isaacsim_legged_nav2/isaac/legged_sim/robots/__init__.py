# SPDX-License-Identifier: Apache-2.0
"""Robots the simulator can run. Each module imports Isaac Sim, so they are imported on demand."""

ROBOTS = ("spot", "anymal_d")


def create_robot(name: str, args):
    if name == "spot":
        from .spot import SpotRobot

        return SpotRobot(engine=args.engine, fast_actuators=args.actuators == "fast")
    if name == "anymal_d":
        from .anymal_d import AnymalDRobot

        return AnymalDRobot()
    raise ValueError(f"unknown robot {name!r}; choose from {ROBOTS}")
