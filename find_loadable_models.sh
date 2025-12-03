#!/bin/bash

# Script to find all saved models that have both checkpoints and configs

echo "=========================================="
echo "Finding models with saved configs..."
echo "=========================================="
echo ""

count=0
for config in $(find onpolicy/results -name "config.yaml" 2>/dev/null); do
  dir=$(dirname "$config")
  if [ -f "$dir/models/actor.pt" ] && [ -f "$dir/models/critic.pt" ]; then
    count=$((count + 1))
    echo "[$count] $dir"
    echo "    Models: $dir/models/"
    echo "    Config: $dir/config.yaml"

    # Extract key parameters
    num_agents=$(grep "^num_agents:" "$config" | awk '{print $2}')
    num_obstacles=$(grep "^num_obstacles:" "$config" | awk '{print $2}')
    graph_feat=$(grep "^graph_feat_type:" "$config" | awk '{print $2}')
    use_mad=$(grep "^use_mad_policy:" "$config" | awk '{print $2}')
    seed=$(grep "^seed:" "$config" | awk '{print $2}')

    echo "    Parameters: agents=$num_agents, obstacles=$num_obstacles, graph_feat=$graph_feat, mad=$use_mad, seed=$seed"
    echo ""
  fi
done

if [ $count -eq 0 ]; then
  echo "No models found with both checkpoints and config files!"
  echo "Train a model first, then you can load it."
else
  echo "=========================================="
  echo "Found $count loadable model(s)"
  echo "=========================================="
fi
