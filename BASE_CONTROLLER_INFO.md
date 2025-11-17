# Base Controller Implementation

## Overview

This is a simplified base controller implementation that adds a proportional controller to the GNN-based policy:

```
u_total = u_base + u_gnn
```

Where:
- **u_base = K_p × (goal - position)**: Proportional controller for baseline navigation
- **u_gnn**: Learned adjustment from GNN features

## Key Files

- `onpolicy/algorithms/graph_base_policy.py`: Policy wrapper
- `onpolicy/algorithms/graph_base_control_actor_critic.py`: Actor/Critic with base controller

## Important Requirements

### ⚠️ MUST Use Continuous Action Space

This implementation **requires continuous actions** (not discrete). When training, you must set:

```bash
--discrete_action False
```

Or in the environment setup, ensure the action space is `gym.spaces.Box` (continuous).

### Why Continuous?

- The base controller outputs continuous forces: `u_base ∈ ℝ²`
- The GNN outputs continuous adjustments: `u_gnn ∈ ℝ²`
- Total action is continuous: `u_total ∈ ℝ²`
- Cannot work with discrete action indices

## How It Works

### Forward Pass (Rollout)

1. Extract position and goal from observations:
   ```python
   current_pos = obs[:, 2:4]  # [pos_x, pos_y]
   goal_pos = obs[:, 4:6]     # [goal_x, goal_y]
   ```

2. Compute base controller:
   ```python
   error = goal_pos - current_pos
   u_base = K_p * error
   ```

3. GNN outputs learned adjustment:
   ```python
   u_gnn ~ Gaussian(μ(GNN_features), σ(GNN_features))
   ```

4. Total action:
   ```python
   actions = u_base + u_gnn
   ```

### Training (evaluate_actions)

**Critical Implementation Detail:**

During training, the buffer stores **total actions** `u_total = u_base + u_gnn`. However, we only want gradients w.r.t. the learned component `u_gnn`.

The `evaluate_actions()` method handles this correctly by:
1. **Recomputing** `u_base` deterministically from observations: `u_base = K_p * (goal - pos)`
2. **Extracting** the learned component: `u_gnn = action - u_base`
3. **Evaluating** `log_prob(u_gnn)` under the current Gaussian policy

This ensures:
- Gradients flow **only** through the GNN parameters (via `u_gnn`)
- The deterministic base controller doesn't affect the policy gradient
- PPO correctly updates the learned adjustment term

## Observation Structure

The implementation assumes observations have the structure:
```
[vel_x, vel_y, pos_x, pos_y, goal_x, goal_y, ...]
```

If your observation structure is different, update line 180-181 in `graph_base_control_actor_critic.py`:
```python
current_pos = obs[:, 2:4]  # Adjust indices
goal_pos = obs[:, 4:6]     # Adjust indices
```

## Training Command Example

```bash
python onpolicy/scripts/train_mpe.py \
  --env_name "GraphMPE" \
  --scenario_name "navigation_graph" \
  --discrete_action False \
  --num_agents 3 \
  --kp_val 0.1 \
  --num_env_steps 2000000
```

## Configuration Parameters

- `--kp_val`: Proportional gain K_p (default: 0.1)
- `--discrete_action`: **Must be False** for base controller

## Differences from MAD Policy

| Feature | Base Controller | MAD Policy |
|---------|----------------|------------|
| Components | u_base + u_gnn | u_base + \|M(x₀)\| × D(neighbors) |
| Magnitude term | None | LRU-based L_p-stable magnitude |
| Direction term | u_gnn (Gaussian) | GNN + Tanh normalization |
| Stability guarantees | None | L_p-stable by construction |
| Complexity | Simple | More complex |

## Return Signatures

All methods return standard MAPPO signatures (no extra u_tot):

- `get_actions()`: `(values, actions, action_log_probs, rnn_states_actor, rnn_states_critic)`
- `act()`: `(actions, rnn_states_actor)`
- `evaluate_actions()`: `(values, action_log_probs, dist_entropy)`

**Self-contained**: No runner modifications required!
