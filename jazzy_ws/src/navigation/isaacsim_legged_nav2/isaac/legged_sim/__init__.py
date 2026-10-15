# SPDX-License-Identifier: Apache-2.0
"""Legged robots in Isaac Sim, walking under Isaac Lab locomotion policies, driven by ROS 2.

Modules that import Isaac Sim (``omni``, ``isaacsim``, ``pxr``) may only be imported after
:func:`legged_sim.app.start_app` has started the application. ``config``, ``geometry``, ``commands``,
``odometry``, ``actuators``, ``networks``, ``rough_policy`` and ``policy_assets`` are plain Python and
can be imported (and tested) anywhere.
"""


def log(message: str) -> None:
    """Print a status line; the ``LEGGED_SIM:`` prefix makes it easy to find in Isaac Sim's output."""
    print(f"LEGGED_SIM: {message}", flush=True)
