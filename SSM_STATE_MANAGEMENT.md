# SSM Hidden State Management in Training Loop

This document explains how SSM (State-Space Model) hidden states are managed throughout the training loop for the Graph-based MAPPO policy with SSM magnitude term.

## Overview

The SSM hidden states are **complex-valued vectors** that maintain temporal information across timesteps within an episode. They are properly initialized, propagated, stored, and reset to ensure correct training dynamics.

## Data Flow

```
Episode Start → Initialize States → Collect Rollout → Store in Buffer → Train → Repeat
```

---

## 1. Buffer Initialization

**Location**: `onpolicy/utils/graph_buffer.py:145-161`

When the buffer is created, it allocates space for SSM hidden states:

```python
self.use_mad_policy = getattr(args, 'use_mad_policy', False)
if self.use_mad_policy:
    self.lru_hidden_dim = getattr(args, 'lru_hidden_dim', 64)
    self.lru_hidden_states = np.zeros(
        (
            self.episode_length + 1,  # Time steps
            self.n_rollout_threads,   # Parallel environments
            num_agents,                # Agents per environment
            self.lru_hidden_dim,       # SSM state dimension
            2,                         # [real, imaginary] parts
        ),
        dtype=np.float32,
    )
else:
    self.lru_hidden_states = None
```

**Key Points**:
- States are stored as **numpy arrays** with separate real and imaginary components
- Shape: `[T+1, N_envs, N_agents, hidden_dim, 2]`
- The `+1` in `episode_length + 1` stores the initial state for the next rollout

---

## 2. Collection Phase (Rollout)

**Location**: `onpolicy/runner/shared/graph_mpe_runner.py:142-223`

During data collection, SSM states flow through the following steps:

### Step 2a: Load from Buffer (Numpy → PyTorch Complex)

```python
if self.buffer.lru_hidden_states is not None:
    # Convert from numpy [batch, agents, hidden_dim, 2] to torch.complex
    ssm_states_np = np.concatenate(self.buffer.lru_hidden_states[step])
    ssm_states = torch.complex(
        torch.from_numpy(ssm_states_np[..., 0]),  # Real part
        torch.from_numpy(ssm_states_np[..., 1])   # Imaginary part
    ).to(self.device)
else:
    ssm_states = None
```

### Step 2b: Forward Pass Through Policy

```python
policy_output = self.trainer.policy.get_actions(
    np.concatenate(self.buffer.share_obs[step]),
    np.concatenate(self.buffer.obs[step]),
    np.concatenate(self.buffer.node_obs[step]),
    np.concatenate(self.buffer.adj[step]),
    np.concatenate(self.buffer.agent_id[step]),
    np.concatenate(self.buffer.share_agent_id[step]),
    np.concatenate(self.buffer.rnn_states[step]),
    np.concatenate(self.buffer.rnn_states_critic[step]),
    ssm_states,  # Complex-valued PyTorch tensor
    np.concatenate(self.buffer.masks[step]),
)

# MAD/SSM policy returns 6 outputs (standard policy returns 5)
value, action, action_log_prob, rnn_states, rnn_states_critic, ssm_states = policy_output
```

**Inside the Actor** (`graph_base_ssm_actor_critic.py:166-227`):

1. **Reset Detection**: Check if `masks == 0` (episode reset)
2. **State Initialization**: Initialize/reset SSM states for new episodes
   ```python
   if ssm_states is None or ssm_states.shape[0] != batch_size:
       ssm_states = torch.complex(
           torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
           torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
       )
   ```
3. **SSM Kickstart**: Input relative goal at t=0, zeros otherwise
   ```python
   rel_goal = obs[:, 4:6]  # [rel_goal_x, rel_goal_y]
   ssm_input = torch.where(
       reset_mask.unsqueeze(-1).expand_as(rel_goal),
       rel_goal,              # At reset: use relative goal
       torch.zeros_like(rel_goal)  # Otherwise: zero input
   )
   ```
4. **SSM Step**: Process through state-space model
   ```python
   ssm_out, ssm_states = self.ssm.step(ssm_input, ssm_states)
   ```
5. **Action Computation**: Combine base controller + magnitude * direction
   ```python
   actions = u_base + torch.abs(ssm_out) * u_gnn
   ```

### Step 2c: Convert Back to Numpy (PyTorch Complex → Numpy)

```python
# Convert SSM states from torch.complex to numpy [batch, agents, hidden_dim, 2]
if ssm_states is not None:
    ssm_states_np = torch.stack([ssm_states.real, ssm_states.imag], dim=-1)
    ssm_states = np.array(np.split(_t2n(ssm_states_np), self.n_rollout_threads))
```

---

## 3. Insert into Buffer

**Location**: `onpolicy/runner/shared/graph_mpe_runner.py:225-286`

After environment step, data is inserted into the buffer:

```python
self.buffer.insert(
    share_obs,
    obs,
    node_obs,
    adj,
    agent_id,
    share_agent_id,
    rnn_states,
    rnn_states_critic,
    actions,
    action_log_probs,
    values,
    rewards,
    masks,
    lru_hidden_states=ssm_states,  # Numpy array [N_envs, N_agents, hidden_dim, 2]
)
```

**Buffer Storage** (`graph_buffer.py:283-284`):
```python
if lru_hidden_states is not None and self.lru_hidden_states is not None:
    self.lru_hidden_states[self.step + 1] = lru_hidden_states.copy()
```

**Important**: The buffer stores states at `step + 1` because states are used to compute actions at the next timestep.

---

## 4. Training Phase

**Location**: `onpolicy/algorithms/graph_mappo.py:140-190`

During PPO updates, SSM states are retrieved from the buffer and passed to `evaluate_actions`:

### Step 4a: Sample from Buffer

The buffer generators (`feed_forward_generator_graph`, `recurrent_generator_graph`, or `naive_recurrent_generator_graph`) yield batches including `lru_hidden_states_batch`.

**Example from buffer** (`graph_buffer.py:482-512`):
```python
if self.lru_hidden_states is not None:
    lru_hidden_states = self.lru_hidden_states[:-1].reshape(
        -1, *self.lru_hidden_states.shape[3:]
    )
    # ...
    lru_hidden_states_batch = lru_hidden_states[indices] if lru_hidden_states is not None else None

yield (..., lru_hidden_states_batch)
```

### Step 4b: Extract from Sample

```python
if len(sample) == 17:  # MAD/SSM policy
    (..., lru_hidden_states_batch) = sample
else:  # Standard policy
    (...) = sample
    lru_hidden_states_batch = None
```

### Step 4c: Evaluate Actions (⚠️ IMPORTANT)

**Current Implementation** (`graph_mappo.py:177-190`):
```python
values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
    share_obs_batch,
    obs_batch,
    node_obs_batch,
    adj_batch,
    agent_id_batch,
    share_agent_id_batch,
    rnn_states_batch,
    rnn_states_critic_batch,
    actions_batch,
    masks_batch,
    available_actions_batch,
    active_masks_batch
    # NOTE: lru_hidden_states_batch is NOT passed here!
)
```

**⚠️ CRITICAL FIX NEEDED**: The `lru_hidden_states_batch` should be passed to `evaluate_actions` for correct gradient computation:

```python
values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
    share_obs_batch,
    obs_batch,
    node_obs_batch,
    adj_batch,
    agent_id_batch,
    share_agent_id_batch,
    rnn_states_batch,
    rnn_states_critic_batch,
    actions_batch,
    masks_batch,
    available_actions_batch,
    active_masks_batch,
    ssm_states=lru_hidden_states_batch  # ADD THIS!
)
```

**Inside evaluate_actions** (`graph_base_ssm_actor_critic.py:284-444`):

1. **Reconstruct SSM states** from the stored buffer values
2. **Recompute magnitude term** using the same SSM input logic (kickstart at reset)
3. **Invert the action** to recover the direction sample
4. **Compute log probabilities** under the current policy distribution

This ensures gradients flow correctly through the policy for PPO updates.

---

## 5. Episode Resets

**During Collection** (`graph_mpe_runner.py:243-252`):

When an environment episode ends (`dones == True`):

```python
# Reset RNN states
rnn_states[dones == True] = np.zeros(...)
rnn_states_critic[dones == True] = np.zeros(...)

# Create masks (0 for done episodes)
masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
masks[dones == True] = np.zeros(...)
```

**Inside Actor Forward Pass**:

The SSM hidden states are automatically reset when `masks == 0` is detected:

```python
reset_mask = (masks.squeeze(-1) == 0)
reset_indices = reset_mask.nonzero(as_tuple=True)[0]

# Reset SSM states for environments that are resetting
if len(reset_indices) > 0:
    ssm_states = ssm_states.clone()
    for idx in reset_indices:
        ssm_states[idx] = torch.complex(
            torch.zeros(self.ssm.LRUR.state_features, device=obs.device),
            torch.zeros(self.ssm.LRUR.state_features, device=obs.device),
        )
```

This ensures each new episode starts with a clean SSM state.

---

## 6. After Update

**Location**: `graph_buffer.py:288-299`

After training updates, the buffer copies the last timestep to the first index:

```python
def after_update(self) -> None:
    """Copy last timestep data to first index."""
    self.share_obs[0] = self.share_obs[-1].copy()
    # ... other buffers ...
    if self.lru_hidden_states is not None:
        self.lru_hidden_states[0] = self.lru_hidden_states[-1].copy()
```

This maintains continuity of SSM states across rollout chunks.

---

## Summary: State Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Buffer Initialization                                    │
│    • Allocate numpy array [T+1, N_envs, N_agents, D, 2]   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Collection Loop (for each timestep)                     │
│    a. Load from buffer: numpy → torch.complex              │
│    b. Actor forward:                                        │
│       • Detect resets (masks == 0)                         │
│       • Reset SSM states if needed                         │
│       • Kickstart with rel_goal at t=0, zeros otherwise    │
│       • SSM step: h_{t+1} = Λh_t + Γ·B·u_t                │
│       • Compute action: u = u_base + |M|·D                 │
│    c. Store: torch.complex → numpy                         │
│    d. Insert into buffer at [step+1]                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Training Phase                                           │
│    • Sample batches from buffer (includes SSM states)      │
│    • evaluate_actions: Reconstruct SSM states & magnitude  │
│    • Compute policy gradients                              │
│    • Update actor/critic networks                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. After Update                                             │
│    • Copy final state to initial position for next rollout │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

1. **Complex-Valued Representation**: SSM states are complex numbers (inherent to LRU/SSM dynamics), stored as separate real/imaginary components in numpy for compatibility with buffer storage.

2. **Kickstart Mechanism**: SSM receives relative goal `obs[:, 4:6]` only at episode start (`masks == 0`), then zeros. This "seeds" the SSM with task-relevant information that propagates through stable dynamics.

3. **Automatic Reset**: SSM states are automatically zeroed when episodes end, ensuring clean slate for new episodes.

4. **Buffer Storage**: States stored at `[step+1]` because they're needed to compute actions at the next timestep.

5. **Gradient Flow**: During training, SSM states must be passed to `evaluate_actions` so magnitude term can be recomputed with gradients enabled.

---

## Potential Issues & Fixes

### ⚠️ Issue 1: Missing SSM States in evaluate_actions

**Problem**: In `graph_mappo.py:177-190`, `lru_hidden_states_batch` is extracted but not passed to `evaluate_actions`.

**Fix**: Add `ssm_states=lru_hidden_states_batch` to the `evaluate_actions` call.

### ⚠️ Issue 2: Observation Structure Assumption

**Problem**: Code assumes observations have structure `[vel_x, vel_y, pos_x, pos_y, rel_goal_x, rel_goal_y, ...]`.

**Fix**: If your observation structure differs, update the indexing in:
- `graph_base_ssm_actor_critic.py:200` (forward)
- `graph_base_ssm_actor_critic.py:343` (evaluate_actions)

---

## Testing Checklist

- [ ] SSM states are properly initialized at episode start
- [ ] SSM receives relative goal at t=0, zeros afterward
- [ ] SSM states are correctly stored in buffer
- [ ] SSM states are reset when episodes end
- [ ] SSM states are passed to evaluate_actions during training
- [ ] Gradients flow through SSM magnitude term
- [ ] Actions combine base controller + magnitude * direction correctly

---

## References

- SSM Implementation: `onpolicy/algorithms/utils/ssm.py`
- Actor/Critic: `onpolicy/algorithms/graph_base_ssm_actor_critic.py`
- Policy Wrapper: `onpolicy/algorithms/graph_base_ssm_policy.py`
- Buffer: `onpolicy/utils/graph_buffer.py`
- Runner: `onpolicy/runner/shared/graph_mpe_runner.py`
- Training: `onpolicy/algorithms/graph_mappo.py`
