# How to Continue Training from a Saved Model

## Summary

The `--model_dir` argument **DOES WORK**, but you need to **match ALL architecture parameters** from the original training run.

## The Issue You're Experiencing

When you use `--model_dir`, the model IS being loaded (you should see this message):
```
Restoring from checkpoint stored in <path>
```

However, if the architecture parameters don't match, you'll get an error like:
```
RuntimeError: Error(s) in loading state_dict for MAD_Critic:
    size mismatch for base.feature_norm.weight: copying a param with shape torch.Size([16])
    from checkpoint, the shape in current model is torch.Size([34]).
```

This happens because parameters like `num_agents`, `num_obstacles`, `graph_feat_type`, `hidden_size`, etc. affect the network architecture. If any of these differ from the original training, the saved weights won't fit.

## Solution: Use the Helper Scripts

### Step 1: Find Available Models

Run the helper script to see which models you can load:

```bash
bash find_loadable_models.sh
```

This will show you all models that have both:
- Saved checkpoints (`actor.pt` and `critic.pt`)
- Configuration file (`config.yaml`)

Example output:
```
[1] onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy/run43
    Models: onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy/run43/models/
    Config: onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy/run43/config.yaml
    Parameters: agents=3, obstacles=3, graph_feat=relative, mad=true, seed=0
```

### Step 2: Generate the Continue Training Command

Use the command generator script to automatically create the correct command:

```bash
python generate_continue_training_command.py <run_directory> [new_experiment_name] [new_seed] [new_num_env_steps]
```

**Examples:**

```bash
# Basic usage - use same config with new experiment name
python generate_continue_training_command.py \
  onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy/run43

# Specify new experiment name, seed, and training steps
python generate_continue_training_command.py \
  onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy/run43 \
  mad_continued \
  999 \
  4000000
```

This will output a complete training command with all the correct parameters!

### Step 3: Run the Generated Command

Copy-paste the generated command and run it. You should see:
```
Restoring from checkpoint stored in <path>
```

And training will continue with the loaded weights.

## Manual Method (Not Recommended)

If you don't have a `config.yaml` file for an old model, you need to manually match ALL these parameters:

### Critical Architecture Parameters (MUST MATCH):
- `--num_agents`
- `--num_obstacles`
- `--num_landmarks`
- `--discrete_action`
- `--use_mad_policy`
- `--graph_feat_type`
- `--use_cent_obs`
- `--hidden_size`
- `--layer_N`
- `--use_ReLU`
- `--embed_hidden_size`
- `--embed_layer_N`
- `--embed_use_ReLU`
- `--gnn_hidden_size`
- `--gnn_layer_N`
- `--gnn_num_heads`
- `--actor_graph_aggr`
- `--critic_graph_aggr`

If `--use_mad_policy` is True, also match:
- `--kp_val`
- `--rmin`
- `--rmax`
- `--ssm_hidden_dim` (if using SSM)
- `--learnable_kp`

### Example Manual Command:

```bash
python -u onpolicy/scripts/train_mpe.py \
  --env_name "GraphMPE" \
  --algorithm_name "rmappo" \
  --scenario_name "navigation_graph" \
  --num_agents 3 \
  --num_obstacles 3 \
  --discrete_action "False" \
  --use_mad_policy \
  --graph_feat_type "relative" \
  --hidden_size 64 \
  --use_ReLU \
  --gnn_hidden_size 16 \
  --gnn_layer_N 2 \
  --num_env_steps 2000000 \
  --experiment_name "continued_training" \
  --model_dir onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy/run43/models
```

## Verification

You know the model loading worked if you see:
1. ✅ Message: `Restoring from checkpoint stored in <path>`
2. ✅ No error about "size mismatch" or "unexpected keys"
3. ✅ Training starts successfully

## Important Notes

1. **Training step counter resets** - The loaded model continues training from step 0 (counter-wise), but the network weights are initialized from the checkpoint
2. **Use a different `--experiment_name`** - This prevents overwriting your original checkpoint
3. **All `.pt` files are in the `models/` subdirectory** - Make sure `--model_dir` points to the `models/` folder, not the run folder
4. **The newer config.yaml files are automatically saved** - All new training runs save their config, making future model loading easier

## Files Created for You

- `find_loadable_models.sh` - Find all models you can load
- `generate_continue_training_command.py` - Generate the exact command to continue training
- `test_model_loading.py` - Test if model loading works
- `HOW_TO_CONTINUE_TRAINING.md` - This guide
