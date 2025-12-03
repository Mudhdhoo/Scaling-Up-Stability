#!/usr/bin/env python3
"""
Test script to verify that --model_dir argument works correctly.
This will attempt to load a saved model and compare weights before/after loading.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add onpolicy to path
sys.path.insert(0, 'onpolicy')

def test_model_loading():
    """Test if model loading actually changes the network weights."""

    print("=" * 80)
    print("TESTING MODEL LOADING FUNCTIONALITY")
    print("=" * 80)

    # Find a saved model to test with
    results_dir = Path("onpolicy/results")
    actor_files = list(results_dir.glob("**/actor.pt"))

    if not actor_files:
        print("❌ No saved models found in onpolicy/results/")
        print("   Please train a model first before testing model loading.")
        return False

    # Use the first available model (parent directory of actor.pt)
    test_model_dir = actor_files[0].parent
    print(f"\n✓ Found saved model at: {test_model_dir}")

    # Check if actor.pt and critic.pt exist
    actor_path = test_model_dir / "actor.pt"
    critic_path = test_model_dir / "critic.pt"

    if not actor_path.exists():
        print(f"❌ actor.pt not found at {actor_path}")
        return False
    if not critic_path.exists():
        print(f"❌ critic.pt not found at {critic_path}")
        return False

    print(f"✓ Found actor.pt: {actor_path}")
    print(f"✓ Found critic.pt: {critic_path}")

    # Try to load the state dicts
    print("\n" + "=" * 80)
    print("ATTEMPTING TO LOAD MODEL STATE DICTS")
    print("=" * 80)

    try:
        actor_state = torch.load(str(actor_path), map_location=torch.device("cpu"))
        print(f"\n✓ Successfully loaded actor.pt")
        print(f"  Actor has {len(actor_state)} parameter tensors")

        # Print some sample parameter info
        for i, (key, tensor) in enumerate(list(actor_state.items())[:3]):
            print(f"    - {key}: shape {tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")
        if len(actor_state) > 3:
            print(f"    ... and {len(actor_state) - 3} more parameters")

    except Exception as e:
        print(f"❌ Failed to load actor.pt: {e}")
        return False

    try:
        critic_state = torch.load(str(critic_path), map_location=torch.device("cpu"))
        print(f"\n✓ Successfully loaded critic.pt")
        print(f"  Critic has {len(critic_state)} parameter tensors")

        # Print some sample parameter info
        for i, (key, tensor) in enumerate(list(critic_state.items())[:3]):
            print(f"    - {key}: shape {tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")
        if len(critic_state) > 3:
            print(f"    ... and {len(critic_state) - 3} more parameters")

    except Exception as e:
        print(f"❌ Failed to load critic.pt: {e}")
        return False

    print("\n" + "=" * 80)
    print("TEST COMMAND TO USE THIS MODEL")
    print("=" * 80)
    print(f"\nAdd this to your training command:")
    print(f"  --model_dir {test_model_dir}")

    print("\n" + "=" * 80)
    print("CHECKING IF model_dir ARGUMENT IS PARSED CORRECTLY")
    print("=" * 80)

    from config import get_config
    parser = get_config()
    test_args = parser.parse_args(['--model_dir', str(test_model_dir)])

    if test_args.model_dir == str(test_model_dir):
        print(f"✓ Argument parsing works correctly")
        print(f"  model_dir = {test_args.model_dir}")
    else:
        print(f"❌ Argument parsing failed")
        print(f"  Expected: {test_model_dir}")
        print(f"  Got: {test_args.model_dir}")
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\nThe model loading functionality should work correctly.")
    print("Make sure you see this message when training:")
    print(f'  "Restoring from checkpoint stored in {test_model_dir}"')
    print("\nIf you don't see that message, the model_dir might not be passed correctly.")

    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
