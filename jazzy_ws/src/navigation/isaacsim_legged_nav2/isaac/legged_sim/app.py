# SPDX-License-Identifier: Apache-2.0
"""Starts Isaac Sim. Everything else that imports Isaac Sim must be imported after start_app()."""

REQUIRED_EXTENSIONS = ("isaacsim.core.nodes", "isaacsim.ros2.bridge", "isaacsim.ros2.nodes")


def start_app(args):
    from isaacsim import SimulationApp

    kit_args = [f"--/exts/isaacsim.core.simulation_manager/default_engine={args.engine}"]
    if args.engine == "newton":
        kit_args += ["--enable", "isaacsim.physics.newton", "--enable", "isaacsim.physics.newton.tensors"]
    app = SimulationApp({"headless": args.headless, "extra_args": kit_args})

    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in REQUIRED_EXTENSIONS:
        manager.set_extension_enabled_immediate(extension, True)
    return app
