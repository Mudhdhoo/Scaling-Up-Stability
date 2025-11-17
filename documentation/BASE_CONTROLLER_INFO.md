# Base Controller in MAD Policy

## Overview

The MAD policy implementation now includes an **optional proportional (P) base controller** that provides baseline stability, with the MAD policy acting as a performance-optimizing addition.

## Policy Structure

### With Base Controller (Default)
```
u_total = u_base + u_MAD
u_total = K_p * (goal - current_pos) + |M_t(x_0)| * D_t(neighbors)
```

where:
- **u_base**: Proportional controller ensuring basic stability
  - `K_p`: Learnable proportional gains (initialized to 1.0)
  - Provides direct feedback from position error to goal

- **u_MAD**: MAD policy for performance optimization
  - `M_t(x_0)`: Magnitude from LRU (seeded with initial conditions)
  - `D_t(neighbors)`: Direction from GNN (based on neighborhood states)

### Without Base Controller
```
u_total = u_MAD = |M_t(x_0)| * D_t(neighbors)
```

## Configuration

### Enable Base Controller (Default)
```bash
python onpolicy/scripts/train_mpe.py \
    --use_mad_policy \
    --use_base_controller True  # Default, can be omitted
```

### Disable Base Controller
```bash
python onpolicy/scripts/train_mpe.py \
    --use_mad_policy \
    --use_base_controller False
```

## Implementation Details

### Proportional Gains (K_p)

The proportional gains are **learnable parameters** of the network:

```python
self.K_p = nn.Parameter(torch.ones(self.action_dim) * 1.0)
```

- **Initialization**: 1.0 for each action dimension
- **Training**: Updated via gradient descent along with other policy parameters
- **Purpose**: Automatically adapt to environment dynamics during training

### Observation Structure Assumption

The base controller assumes observations contain:
```
obs = [pos_x, pos_y, vel_x, vel_y, goal_x, goal_y, ...]
       ^^^^^^^^              ^^^^^^^^^^^^^^^^^^
       current               goal position
```

**Indices used**:
- `current_pos = obs[:, 0:2]`  # Position (x, y)
- `goal_pos = obs[:, 4:6]`     # Goal (x, y)

**Important**: If your observation structure is different, you need to adjust these indices in `mad_actor_critic.py` lines 277-278.

### Fallback Behavior

If the observation dimension is less than 6 (insufficient for goal extraction), the base controller is automatically disabled for that forward pass:

```python
if obs_dim >= 6:
    # Use base controller
    actions = u_base + u_mad
else:
    # Fall back to MAD only
    actions = u_mad
```

## Relationship to MAD Paper

From the MAD paper (Section IV, Corridor Example):

> "Each vehicle must reach a target position p̄[i] ∈ ℝ² with zero velocity in a stable way. This elementary goal can be achieved by using a base proportional controller πb that sets:
>
> F'[i]_t = K'[i](p̄[i] - p[i]_t)
>
> The overall dynamics f(x_{t-1}, u_{t-1}) in (1) is given by (19)-(20) with:
>
> F[i]_t = F'[i]_t + u[i]_t
>
> where u[i]_t is a performance-boosting control input to be designed."

This matches our implementation:
- **F'[i]_t** = u_base = K_p * (goal - current)
- **u[i]_t** = u_MAD = |M_t| * D_t
- **F[i]_t** = u_total = u_base + u_MAD

## Stability Analysis

### With Base Controller
1. **Base Stability**: P controller provides basic stability (goal-seeking behavior)
2. **MAD Enhancement**:
   - Magnitude M_t ∈ L_p (guaranteed by LRU with |λ| < 1)
   - Direction |D_t| ≤ 1 (guaranteed by tanh)
   - Combined: |u_MAD| ≤ |M_t|
3. **Total System**: u_base ensures stability, u_MAD optimizes performance

### Without Base Controller
1. **Relies entirely on MAD**:
   - M_t must provide sufficient control authority
   - System must be inherently stable or pre-stabilized
2. **Use case**: When system dynamics F already include stabilization

## Training Recommendations

### Standard Setup (With Base Controller)
Recommended for most navigation tasks:

```bash
python onpolicy/scripts/train_mpe.py \
    --use_mad_policy \
    --use_base_controller True \
    --lru_hidden_dim 64 \
    --lr 7e-4
```

**Advantages**:
- Faster convergence (base controller provides good initialization)
- More robust training (baseline stability always present)
- Better sample efficiency

### Pure MAD (Without Base Controller)
For research or when base controller is undesired:

```bash
python onpolicy/scripts/train_mpe.py \
    --use_mad_policy \
    --use_base_controller False \
    --lru_hidden_dim 128  # May need more capacity
    --lr 5e-4             # May need lower learning rate
```

**Advantages**:
- Pure MAD policy learning
- No assumptions about observation structure
- Useful for ablation studies

## Customizing the Base Controller

If you need a different base controller structure, modify `mad_actor_critic.py`:

### Example: PD Controller

```python
# In MAD_Actor.__init__
self.K_p = nn.Parameter(torch.ones(self.action_dim) * 1.0)  # P gains
self.K_d = nn.Parameter(torch.ones(self.action_dim) * 0.5)  # D gains

# In MAD_Actor.forward
current_pos = obs[:, 0:2]
current_vel = obs[:, 2:4]
goal_pos = obs[:, 4:6]

error_p = goal_pos - current_pos
error_d = -current_vel  # Velocity damping

u_base = self.K_p.unsqueeze(0) * error_p + self.K_d.unsqueeze(0) * error_d
```

### Example: Fixed Gains (Non-learnable)

```python
# In MAD_Actor.__init__
# Use register_buffer instead of Parameter
self.register_buffer('K_p', torch.ones(self.action_dim) * 1.0)

# Forward pass remains the same, but K_p won't be updated during training
```

### Example: Different Observation Structure

If your observations are structured differently, adjust the indices:

```python
# Example: obs = [goal_x, goal_y, pos_x, pos_y, vel_x, vel_y, ...]
goal_pos = obs[:, 0:2]      # Goals first
current_pos = obs[:, 2:4]   # Position next
```

## Comparison: With vs Without Base Controller

| Aspect | With Base Controller | Without Base Controller |
|--------|---------------------|------------------------|
| **Stability** | Strong (P controller + MAD) | Moderate (MAD only) |
| **Training Speed** | Faster | Slower |
| **Sample Efficiency** | Higher | Lower |
| **Flexibility** | Assumes obs structure | No assumptions |
| **Use Case** | Navigation, goal-reaching | General RL, ablations |
| **Parameters** | K_p + MAD | MAD only |

## Debugging

### Issue: Base controller not working

**Check observation structure**:
```python
# Add print in mad_actor_critic.py forward():
print(f"Obs shape: {obs.shape}")
print(f"Current pos: {obs[:, 0:2]}")
print(f"Goal pos: {obs[:, 4:6]}")
```

**Verify K_p values**:
```python
# After training, check learned gains
print(f"Learned K_p: {model.actor.K_p}")
```

### Issue: K_p values exploding or vanishing

**Solution**: Constrain K_p during training:
```python
# In MAD_Actor.forward, add clamping
K_p_clamped = torch.clamp(self.K_p, min=0.1, max=10.0)
u_base = K_p_clamped.unsqueeze(0) * error
```

## Examples

### Full Training Command with Base Controller

```bash
python -u onpolicy/scripts/train_mpe.py \
    --use_valuenorm --use_popart \
    --project_name "mad_with_base" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --scenario_name "navigation_graph" \
    --num_agents 3 \
    --num_obstacles 3 \
    --use_mad_policy \
    --use_base_controller True \
    --lru_hidden_dim 64 \
    --n_rollout_threads 128 \
    --num_env_steps 2000000 \
    --lr 7e-4 \
    --use_wandb
```

### Ablation Study: Compare With/Without

```bash
# Train WITH base controller
python onpolicy/scripts/train_mpe.py \
    --experiment_name "with_base" \
    --use_mad_policy \
    --use_base_controller True \
    --seed 0

# Train WITHOUT base controller
python onpolicy/scripts/train_mpe.py \
    --experiment_name "without_base" \
    --use_mad_policy \
    --use_base_controller False \
    --seed 0
```

## Summary

✅ **Base controller is now included by default**
✅ **Proportional gains K_p are learnable**
✅ **Can be toggled with `--use_base_controller`**
✅ **Follows MAD paper's additive structure**
✅ **Provides baseline stability for faster training**

The implementation now correctly follows the structure from the MAD paper where the learned policy is **added** to a base stabilizing controller, rather than replacing it entirely.
