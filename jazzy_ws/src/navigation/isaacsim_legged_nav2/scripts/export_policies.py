#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download NVIDIA's pretrained locomotion checkpoints and export them for the simulator.

Isaac Sim 6.1's Python has no torch, so this runs once with a Python that has it (``pip install
torch``). For each robot it writes ``<cache>/<robot>/``: NumPy weights, torch reference outputs
that the simulator re-checks on load, and a manifest with the source URLs and SHA-256 hashes.

    python3 export_policies.py anymal_d [--cache DIR]

The checkpoint is loaded with ``weights_only=True`` (tensors only, no pickled code). The ANYdrive
actuator network is TorchScript published by NVIDIA alongside the robot assets.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

ASSETS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "isaacsim_legged_nav2", "policies")

ROBOTS = {
    "anymal_d": {
        # Isaac Lab Isaac-Velocity-Rough-Anymal-D-v0, PhysX backend
        "actor": f"{ASSETS}/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-AnymalD_physx_none_rsl_rl.pt",
        # ANYdrive 3 LSTM; Isaac Lab uses ANYmal-C's network for ANYmal-D (no public ANYmal-D network)
        "actuator_net": f"{ASSETS}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    },
}


def download(url: str, directory: str) -> tuple[str, str]:
    path = os.path.join(directory, os.path.basename(url))
    urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        return path, hashlib.sha256(f.read()).hexdigest()


def export_actor(torch, np, checkpoint: str, out_dir: str) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
    weights = {k.replace(".", "_"): v.numpy() for k, v in state.items() if k.startswith("mlp.")}
    np.savez(os.path.join(out_dir, "actor.npz"), **weights)
    # reference: torch's own forward pass of the same layers (Linear/ELU stack)
    layers = sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight"))
    inputs = torch.randn(8, state[f"mlp.{layers[0]}.weight"].shape[1], generator=torch.Generator().manual_seed(0))
    x = inputs
    for i, idx in enumerate(layers):
        x = torch.nn.functional.linear(x, state[f"mlp.{idx}.weight"], state[f"mlp.{idx}.bias"])
        if i < len(layers) - 1:
            x = torch.nn.functional.elu(x)
    np.savez(os.path.join(out_dir, "actor_reference.npz"), inputs=inputs.numpy(), outputs=x.numpy())
    return {"inputs": int(inputs.shape[1]), "outputs": int(x.shape[1])}


def export_actuator_net(torch, np, network_file: str, out_dir: str) -> dict:
    net = torch.jit.load(network_file, map_location="cpu").eval()
    state = net.state_dict()
    np.savez(os.path.join(out_dir, "actuator_net.npz"), **{k.replace(".", "_"): v.numpy() for k, v in state.items()})
    layers, hidden = len(net.lstm.state_dict()) // 4, state["lstm.weight_hh_l0"].shape[1]
    joints = 3
    gen = torch.Generator().manual_seed(0)
    steps = torch.randn(6, joints, 1, 2, generator=gen) * torch.tensor([0.3, 2.0])
    h = torch.zeros(layers, joints, hidden)
    c = torch.zeros_like(h)
    outputs = []
    with torch.inference_mode():
        for x in steps:
            y, (h, c) = net(x, (h, c))
            outputs.append(y.reshape(-1).numpy())
    np.savez(os.path.join(out_dir, "actuator_net_reference.npz"), inputs=steps[:, :, 0, :].numpy(),
             outputs=np.stack(outputs))
    return {"lstm_layers": layers, "hidden": int(hidden)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("robots", nargs="+", choices=sorted(ROBOTS))
    parser.add_argument("--cache", default=os.environ.get("LEGGED_SIM_POLICY_DIR", DEFAULT_CACHE))
    args = parser.parse_args()
    try:
        import numpy as np
        import torch
    except ImportError as error:
        print(f"this script needs numpy and torch ({error}); try: pip install torch", file=sys.stderr)
        return 1
    for robot in args.robots:
        out_dir = os.path.join(args.cache, robot)
        os.makedirs(out_dir, exist_ok=True)
        manifest = {"robot": robot, "sources": {}}
        with tempfile.TemporaryDirectory() as tmp:
            for name, url in ROBOTS[robot].items():
                path, digest = download(url, tmp)
                manifest["sources"][name] = {"url": url, "sha256": digest}
                exporter = export_actor if name == "actor" else export_actuator_net
                manifest["sources"][name]["shape"] = exporter(torch, np, path, out_dir)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"{robot}: exported to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
