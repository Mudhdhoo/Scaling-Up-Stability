#!/usr/bin/env python3
"""
Generate a training command that continues from a saved checkpoint.
This script reads the config.yaml from a saved run and generates
the exact command needed to load that model.
"""

import sys
import yaml
from pathlib import Path

def generate_command(run_dir, new_experiment_name=None, new_seed=None, new_num_env_steps=None):
    """Generate training command from saved config."""

    run_path = Path(run_dir)
    config_path = run_path / "config.yaml"
    model_path = run_path / "models"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return None

    if not (model_path / "actor.pt").exists():
        print(f"ERROR: Model checkpoint not found: {model_path}/actor.pt")
        return None

    # Load the config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override some parameters
    if new_experiment_name is None:
        new_experiment_name = config['experiment_name'] + "_continued"
    if new_seed is not None:
        config['seed'] = new_seed
    if new_num_env_steps is not None:
        config['num_env_steps'] = new_num_env_steps

    # Build the command
    cmd = "python -u onpolicy/scripts/train_mpe.py \\\n"

    # Critical architecture parameters that MUST match
    critical_params = [
        'env_name', 'algorithm_name', 'scenario_name',
        'num_agents', 'num_obstacles', 'num_landmarks',
        'discrete_action', 'use_mad_policy',
        'graph_feat_type', 'use_cent_obs',
        'hidden_size', 'layer_N', 'use_ReLU',
        'embed_hidden_size', 'embed_layer_N', 'embed_use_ReLU',
        'gnn_hidden_size', 'gnn_layer_N', 'gnn_num_heads',
        'actor_graph_aggr', 'critic_graph_aggr',
    ]

    # Other important training parameters
    training_params = [
        'n_rollout_threads', 'episode_length', 'num_env_steps',
        'ppo_epoch', 'lr', 'critic_lr', 'gain',
        'use_valuenorm', 'use_popart',
        'collision_rew', 'goal_rew',
        'auto_mini_batch_size', 'target_mini_batch_size',
        'use_wandb', 'seed',
    ]

    # MAD-specific parameters
    mad_params = [
        'kp_val', 'rmin', 'rmax', 'm_max_start', 'm_max_final',
        'm_max_warmup_episodes', 'm_max_step_episode', 'm_schedule_type',
        'learnable_kp',
    ]

    # Add parameters to command
    all_params = critical_params + training_params
    if config.get('use_mad_policy', False):
        all_params += mad_params

    for param in all_params:
        if param in config and config[param] is not None:
            value = config[param]
            # Handle boolean values
            if isinstance(value, bool):
                value = str(value)
            # Handle string values that might have spaces
            elif isinstance(value, str):
                value = f'"{value}"'

            cmd += f"  --{param} {value} \\\n"

    # Add experiment name and model_dir
    cmd += f"  --experiment_name {new_experiment_name} \\\n"
    cmd += f"  --model_dir {model_path} \\\n"

    # Add user name if specified
    if 'user_name' in config:
        cmd += f"  --user_name \"{config['user_name']}\" \\\n"

    # Add project name if specified
    if 'project_name' in config:
        cmd += f"  --project_name \"{config['project_name']}\" \\\n"

    # Remove trailing backslash
    cmd = cmd.rstrip(" \\\n")

    return cmd

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_continue_training_command.py <run_directory> [new_experiment_name] [new_seed] [new_num_env_steps]")
        print("\nExample:")
        print("  python generate_continue_training_command.py onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy/run43")
        print("  python generate_continue_training_command.py onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy/run43 continued_run 999 4000000")
        sys.exit(1)

    run_dir = sys.argv[1]
    new_experiment_name = sys.argv[2] if len(sys.argv) > 2 else None
    new_seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    new_num_env_steps = int(sys.argv[4]) if len(sys.argv) > 4 else None

    print("=" * 80)
    print("GENERATING CONTINUE TRAINING COMMAND")
    print("=" * 80)
    print(f"\nSource run: {run_dir}")

    cmd = generate_command(run_dir, new_experiment_name, new_seed, new_num_env_steps)

    if cmd:
        print("\n" + "=" * 80)
        print("GENERATED COMMAND:")
        print("=" * 80)
        print()
        print(cmd)
        print()
        print("=" * 80)
        print("\nYou can copy-paste this command to continue training from the checkpoint.")
        print("The model will be loaded from:", Path(run_dir) / "models")
        print("=" * 80)
    else:
        print("\nFailed to generate command.")
        sys.exit(1)

if __name__ == "__main__":
    main()
