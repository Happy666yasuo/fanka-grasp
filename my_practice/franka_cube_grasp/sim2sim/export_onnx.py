# Franka Cube Grasp — ONNX Export Script
# Conda env: beyondmimic (Python 3.10)
"""
Export a trained SAC actor network to ONNX format for Sim2Sim deployment.

This script:
1. Loads a trained SB3 SAC checkpoint (.zip)
2. Extracts the actor network (policy)
3. Exports it to ONNX with the correct observation dimension

Usage:
    conda activate beyondmimic
    cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
    python sim2sim/export_onnx.py --checkpoint logs/test_run/checkpoints/latest.zip

Output:
    sim2sim/policy_franka_grasp.onnx

⚠️ Only works with standard SAC (not HER), because HER uses Dict obs space.
"""
from __future__ import annotations

import os
import sys
import argparse

# Project root
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# We need to import IsaacLab for SAC.load to work (env registration)
# But we DON'T need to actually create the env.

import torch
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAC actor to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SB3 SAC model (.zip file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path (default: sim2sim/policy_franka_grasp.onnx).",
    )
    parser.add_argument(
        "--obs_dim",
        type=int,
        default=32,
        help="Observation dimension (default: 32).",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if not ckpt_path.endswith(".zip"):
        ckpt_path += ".zip"
    if not os.path.isfile(ckpt_path):
        # Try without .zip
        ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Resolve output path
    if args.output is None:
        sim2sim_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(sim2sim_dir, "policy_franka_grasp.onnx")
    else:
        output_path = args.output

    print("=" * 60)
    print("Franka Cube Grasp — ONNX Export")
    print("=" * 60)
    print(f"  checkpoint : {ckpt_path}")
    print(f"  output     : {output_path}")
    print(f"  obs_dim    : {args.obs_dim}")
    print("=" * 60)

    # Load model (without environment)
    from stable_baselines3 import SAC

    print("[INFO] Loading SAC model...")
    model = SAC.load(ckpt_path, device="cpu")
    print(f"[INFO] Model loaded. Policy class: {type(model.policy).__name__}")

    # Extract actor network
    actor = model.actor
    actor.eval()

    # Create dummy input
    obs_dim = args.obs_dim
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    # Test forward pass
    with torch.no_grad():
        test_out = actor(dummy_input)
        if isinstance(test_out, tuple):
            print(f"[INFO] Actor output: tuple of {len(test_out)} elements, "
                  f"action shape = {test_out[0].shape}")
        else:
            print(f"[INFO] Actor output shape: {test_out.shape}")

    # Export to ONNX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )

    # Verify ONNX file
    file_size = os.path.getsize(output_path) / 1024
    print(f"[INFO] ✅ ONNX exported: {output_path} ({file_size:.1f} KB)")

    # Optional: verify with onnxruntime if available
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_name = session.get_outputs()[0].name
        output_shape = session.get_outputs()[0].shape

        # Test inference
        test_obs = np.random.randn(1, obs_dim).astype(np.float32)
        result = session.run(None, {input_name: test_obs})
        print(f"[INFO] ONNX verification:")
        print(f"  input:  {input_name} {input_shape}")
        print(f"  output: {output_name} {output_shape}")
        print(f"  test action = {result[0][0][:4]}... (first 4 dims)")
    except ImportError:
        print("[INFO] onnxruntime not available, skipping verification.")

    print("[INFO] Export complete.")


if __name__ == "__main__":
    main()
