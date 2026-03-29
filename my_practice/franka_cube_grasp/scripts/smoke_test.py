# Franka Cube Grasp — Smoke Test
# Conda env: beyondmimic (Python 3.10)
"""
Minimal smoke test to verify the project structure is intact.
Phase 0: just checks imports and prints confirmation.
Phase 1+: will instantiate the IsaacLab environment and run random steps.

Usage:
    conda activate beyondmimic
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys


def main() -> None:
    """Run Phase 0 smoke test — verify core imports."""
    print("=" * 60)
    print("Franka Cube Grasp — Smoke Test (Phase 0)")
    print("=" * 60)

    errors: list[str] = []

    # Check IsaacSim
    try:
        import isaacsim  # noqa: F401
        print("[✓] isaacsim importable")
    except ImportError as e:
        errors.append(f"isaacsim: {e}")
        print(f"[✗] isaacsim: {e}")

    # Check IsaacLab
    try:
        import isaaclab
        print(f"[✓] isaaclab {isaaclab.__version__}")
    except ImportError as e:
        errors.append(f"isaaclab: {e}")
        print(f"[✗] isaaclab: {e}")

    # Check SB3 SAC
    try:
        from stable_baselines3 import SAC  # noqa: F401
        print("[✓] stable_baselines3.SAC ready")
    except ImportError as e:
        errors.append(f"SB3: {e}")
        print(f"[✗] SB3: {e}")

    # Check PyTorch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"[✓] torch {torch.__version__}, CUDA: {cuda_ok}")
        if not cuda_ok:
            errors.append("CUDA not available")
    except ImportError as e:
        errors.append(f"torch: {e}")
        print(f"[✗] torch: {e}")

    # Summary
    print("=" * 60)
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED ✅")
        sys.exit(0)


if __name__ == "__main__":
    main()
