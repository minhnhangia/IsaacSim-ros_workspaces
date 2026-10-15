# SPDX-License-Identifier: Apache-2.0
"""Spot in Isaac Sim, walking under the bundled Isaac Lab locomotion policy, driven by ROS 2.

Run with Isaac Sim's Python (usually through spot_sim.launch.xml):

    ~/isaacsim/python.sh spot_ros2_sim.py [--headless] [--scene office|grid|<file.usd>] [--help]
"""

import sys

from spot_sim.app import start_app
from spot_sim.config import parse_args


def main() -> None:
    args = parse_args(sys.argv[1:])
    app = start_app(args)
    from spot_sim.simulation import SpotSimulation  # imports Isaac Sim, so only once the app runs

    try:
        SpotSimulation(app, args).run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
