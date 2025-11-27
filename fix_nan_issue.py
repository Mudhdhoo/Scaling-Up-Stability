#!/usr/bin/env python
"""
Diagnostic script to identify where NaNs appear in the forward pass.
Run this before training to check for numerical issues.

Usage:
    python fix_nan_issue.py --num_agents 7 --scenario_name navigation_graph
"""

import torch
import torch.nn as nn
import argparse
import sys
import numpy as np

def add_nan_hooks(model, name="model"):
    """Add hooks to detect NaN values in forward pass"""
    def make_hook(module_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"❌ NaN detected in {module_name}")
                    print(f"   Output shape: {output.shape}")
                    print(f"   NaN count: {torch.isnan(output).sum().item()}")
                    # Print input stats for debugging
                    if isinstance(input, tuple) and len(input) > 0:
                        inp = input[0]
                        if isinstance(inp, torch.Tensor):
                            print(f"   Input stats: min={inp.min():.4f}, max={inp.max():.4f}, mean={inp.mean():.4f}")
                    raise RuntimeError(f"NaN detected in {module_name}")
                else:
                    # Check for extreme values that might lead to NaN
                    if output.abs().max() > 1e6:
                        print(f"⚠️  Warning: Large values in {module_name}")
                        print(f"   Max abs value: {output.abs().max():.2e}")
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            module.register_forward_hook(make_hook(name))

def check_gradients(model, name="model"):
    """Check for NaN or extreme gradients"""
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"❌ NaN gradient in {param_name}")
                return False
            if param.grad.abs().max() > 1e6:
                print(f"⚠️  Large gradient in {param_name}: {param.grad.abs().max():.2e}")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("NaN Detection Diagnostic Tool")
    print("=" * 60)

    print("\n✓ This script provides utilities to detect NaN issues.")
    print("✓ Import and use in your training script:\n")
    print("    from fix_nan_issue import add_nan_hooks, check_gradients")
    print("    add_nan_hooks(policy.actor)")
    print("    add_nan_hooks(policy.critic)")
    print("\n" + "=" * 60)
