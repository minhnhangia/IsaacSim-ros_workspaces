# SPDX-License-Identifier: Apache-2.0
"""A legged robot (Spot or ANYmal-D) in Isaac Sim, walking under an Isaac Lab policy, driven by ROS 2.

Run with Isaac Sim's Python (usually through sim.launch.xml):

    ~/isaacsim/python.sh legged_ros2_sim.py [--robot spot|anymal_d] [--headless] [--help]
"""

import sys

from legged_sim.app import start_app
from legged_sim.config import parse_args


def main() -> None:
    args = parse_args(sys.argv[1:])
    app = start_app(args)
    from legged_sim.simulation import LeggedSimulation  # imports Isaac Sim, so only once the app runs

    try:
        LeggedSimulation(app, args).run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
