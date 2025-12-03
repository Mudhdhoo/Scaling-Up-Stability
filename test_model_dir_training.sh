#!/bin/bash

# Test script to verify that --model_dir actually loads the model during training
# This runs a very short training (just 1000 steps) to test the loading

echo "=========================================="
echo "Testing --model_dir during training"
echo "=========================================="

# Find a saved model
MODEL_DIR=$(find onpolicy/results -name "actor.pt" | head -1 | xargs dirname)

if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: No saved models found!"
    echo "Please train a model first."
    exit 1
fi

echo ""
echo "Using model from: $MODEL_DIR"
echo ""
echo "Look for this message in the output:"
echo "  'Restoring from checkpoint stored in $MODEL_DIR'"
echo ""
echo "=========================================="
echo "Starting training with --model_dir..."
echo "=========================================="
echo ""

# Run a very short training to test model loading
# Using MAD policy configuration since that's what we found
python -u onpolicy/scripts/train_mpe.py \
  --use_valuenorm --use_popart \
  --project_name "test_model_loading" \
  --env_name "GraphMPE" \
  --algorithm_name "rmappo" \
  --seed 999 \
  --experiment_name "test_loading" \
  --scenario_name "navigation_graph" \
  --num_agents 3 \
  --collision_rew 5 \
  --n_training_threads 1 \
  --n_rollout_threads 4 \
  --num_mini_batch 1 \
  --episode_length 25 \
  --num_env_steps 1000 \
  --ppo_epoch 2 \
  --use_ReLU \
  --gain 0.01 \
  --lr 7e-4 \
  --critic_lr 7e-4 \
  --user_name "test" \
  --use_cent_obs "False" \
  --graph_feat_type "relative" \
  --auto_mini_batch_size \
  --target_mini_batch_size 32 \
  --use_wandb "False" \
  --use_mad_policy \
  --discrete_action "False" \
  --model_dir "$MODEL_DIR" \
  --verbose "True" 2>&1 | grep -E "(Restoring|model_dir|Model|Loading|actor|critic)" || echo "No loading messages found!"

echo ""
echo "=========================================="
echo "Test complete"
echo "=========================================="
