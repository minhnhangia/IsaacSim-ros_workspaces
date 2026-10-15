# SPDX-License-Identifier: Apache-2.0
"""Spot in Isaac Sim, walking under the bundled Isaac Lab locomotion policy, driven by ROS 2.

Modules that import Isaac Sim (``omni``, ``isaacsim``, ``pxr``) may only be imported after
:func:`spot_sim.app.start_app` has started the application. ``config``, ``geometry``, ``commands``,
``odometry`` and ``actuators`` are plain Python and can be imported (and tested) anywhere.
"""


def log(message: str) -> None:
    """Print a status line; the ``SPOTSIM:`` prefix makes it easy to find in Isaac Sim's output."""
    print(f"SPOTSIM: {message}", flush=True)
