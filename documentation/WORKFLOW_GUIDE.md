# InforMARL Workflow & Architecture Guide

This document provides a comprehensive walkthrough of the InforMARL codebase workflow, explaining how all core components work together from initialization to training.

## Table of Contents

- [High-Level Concept](#high-level-concept)
- [Complete Training Pipeline](#complete-training-pipeline)
- [Data Flow Phases](#data-flow-phases)
- [GNN Processing Deep Dive](#gnn-processing-deep-dive)
- [Component Relationships](#component-relationships)
- [File Reference](#file-reference)
- [Key Insights](#key-insights)

---

## High-Level Concept

InforMARL is a **Graph Neural Network (GNN)-based Multi-Agent RL framework** that allows agents to:
1. Only observe their **local neighborhood** (partial observability)
2. Use **GNNs to aggregate information** from nearby agents/obstacles
3. Make **decentralized decisions** that scale to many agents
4. Optionally use **MAD policy** for stability-constrained control

---

## Complete Training Pipeline

### 1. Entry Point: `scripts/train_mpe.py`

```
┌─────────────────────────────────────────┐
│  python train_mpe.py                    │
│  --env_name "GraphMPE"                  │
│  --scenario_name "navigation_graph"     │
│  --use_mad_policy (optional)            │
└─────────────────────────────────────────┘
                    │
                    ▼
    Parse Config → Create Environment → Initialize Runner → Start Training
```

**Key Decisions Made Here:**
- **Environment type**: `GraphMPE` (with graphs) vs `MPE` (standard)
- **Policy type**: MAD policy vs standard MAPPO
- **Parallelization**: How many rollout threads
- **Hyperparameters**: Learning rates, PPO epochs, episode length

**Execution Flow:**

```python
train_mpe.py:main()
  ├── Parse configuration: get_config() + graph_config() [for GraphMPE]
  ├── Setup CUDA/CPU device
  ├── Create training environments:
  │   ├── make_train_env()
  │   │   └── Creates GraphMPEEnv or MPEEnv
  │   │   └── Wraps in GraphSubprocVecEnv or SubprocVecEnv (parallelizes)
  │   └── make_eval_env() [optional]
  │
  ├── Initialize runner based on env_name and policy type:
  │   └── For GraphMPE with shared policy:
  │       └── GMPERunner from graph_mpe_runner.py
  │
  └── runner.run() → Begin training loop
```

---

### 2. Environment Creation: `multiagent/MPE_env.py`

**GraphMPEEnv wraps the multi-agent particle environment:**

```python
# Environment outputs for GraphMPE:
obs, agent_id, node_obs, adj, rewards, dones, infos = env.step(actions)
```

**Data Structures:**
- **`obs`**: `[n_threads, n_agents, obs_dim]` - Local observations (position, velocity, goal)
- **`node_obs`**: `[n_threads, n_agents, max_nodes, node_feat_dim]` - Features of all entities in sensing range
- **`adj`**: `[n_threads, n_agents, max_nodes, max_nodes]` - Adjacency matrices defining connectivity
- **`agent_id`**: `[n_threads, n_agents, 1]` - Agent IDs for indexing

**What's in the graph?** Each agent sees a local graph containing:
- Itself (ego node)
- Nearby agents (within sensing radius)
- Nearby goals
- Nearby obstacles

Each node has features: `[relative_x, relative_y, vel_x, vel_y, goal_x, goal_y, entity_type]`

**Graph Observation Construction** (`navigation_graph.py`):

```
For each agent i:
  ├── Get local neighborhood (entities within sensing radius)
  ├── Construct node features: [pos, vel, goal, entity_type]
  ├── Construct adjacency matrix: distance between all local nodes
  ├── Filter by max_edge_dist: only keep close connections
  └── Return: node_obs [n_nodes, feat_dim], adj [n_nodes, n_nodes]

Entities in local graph:
  ├── The agent itself
  ├── Nearby agents (within sensing radius)
  ├── Nearby goals
  └── Nearby obstacles
```

---

### 3. Runner Initialization: `runner/shared/base_runner.py`

The runner creates **3 core components**:

```python
# 1. Policy Network (Actor + Critic)
self.policy = Policy(args, obs_space, node_obs_space, action_space, device)

# 2. Training Algorithm (PPO)
self.trainer = TrainAlgo(args, self.policy, device)

# 3. Replay Buffer
self.buffer = GraphReplayBuffer(args, num_agents, obs_space, node_obs_space, ...)
```

**Policy Selection Logic** (lines 78-90):
```python
if env_name == "GraphMPE":
    if use_mad_policy:
        from onpolicy.algorithms.mad_MAPPOPolicy import MAD_MAPPOPolicy as Policy
        from onpolicy.algorithms.graph_mappo import GR_MAPPO as TrainAlgo
    else:
        from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy as Policy
        from onpolicy.algorithms.graph_mappo import GR_MAPPO as TrainAlgo
```

**Three Core Components Initialized:**

1. **Policy Network** (lines 99-116):
   - Contains actor and critic networks
   - For GraphMPE: requires node_obs_space and edge_obs_space
   - For MAD: uses MAD_Actor with LRU magnitude term

2. **Training Algorithm** (line 124):
   - Implements PPO update logic
   - Manages value and policy losses

3. **Replay Buffer** (lines 127-146):
   - For GraphMPE: stores obs, node_obs, adj, agent_id, rewards, etc.
   - For standard MPE: stores obs, rewards, actions, etc.

---

### 4. Training Loop: `runner/shared/graph_mpe_runner.py`

The core training loop has **4 phases**:

```python
def run(self):
    # Warmup: Initialize buffer with first obs from environment
    obs, node_obs, adj = env.reset()

    for episode in range(num_episodes):
        for step in range(episode_length):
            # PHASE 1: COLLECT - Get actions from policy
            values, actions, log_probs = self.collect(step)

            # PHASE 2: STEP - Execute in environment
            obs, node_obs, adj, rewards, dones = env.step(actions)

            # PHASE 3: INSERT - Store in buffer
            self.insert(obs, node_obs, adj, rewards, masks, ...)

        # PHASE 4a: COMPUTE - Calculate returns
        next_values = self.compute()

        # PHASE 4b: TRAIN - Update networks
        train_infos = self.train()
```

**Main Training Epoch Structure** (lines 26-111):

```
GMPERunner.run()
  ├── warmup(): Initialize buffer with first obs from environment
  │
  ├── For each episode:
  │   ├── For each step (0 to episode_length):
  │   │   ├── collect(step) → Get actions from policy
  │   │   │   ├── trainer.prep_rollout() [set to eval mode]
  │   │   │   └── policy.get_actions(obs, node_obs, adj, agent_id, rnn_states, masks)
  │   │   │       └── Returns: values, actions, log_probs, rnn_states
  │   │   │
  │   │   ├── env.step(actions) → Execute actions
  │   │   │   └── Returns: obs, agent_id, node_obs, adj, rewards, dones, infos
  │   │   │
  │   │   └── insert(data) → Store experience in buffer
  │   │       └── buffer.insert(obs, node_obs, adj, rewards, etc.)
  │   │
  │   ├── compute() → Calculate returns
  │   │   └── policy.get_values(cent_obs, node_obs, adj, rnn_states_critic, masks)
  │   │   └── buffer.compute_returns(next_values, value_normalizer)
  │   │
  │   └── train() → Update networks
  │       ├── trainer.prep_training() [set to train mode]
  │       └── trainer.train(buffer) → PPO update loop
  │           ├── For each PPO epoch:
  │           │   └── For each mini-batch:
  │           │       ├── policy.evaluate_actions() → Recompute log probs
  │           │       ├── Calculate policy loss (clipped surrogate)
  │           │       ├── Calculate value loss
  │           │       ├── Backward pass & gradient clipping
  │           │       └── Update actor & critic optimizers
  │           │
  │           └── Return training metrics
  │
  └── Log & save results
```

---

## Data Flow Phases

### PHASE 1: COLLECT (Get Actions from Policy)

```python
# In graph_mpe_runner.py:collect()
values, actions, action_log_probs, rnn_states = \
    self.policy.get_actions(
        cent_obs,      # Centralized observations for critic
        obs,           # Local observations
        node_obs,      # Graph node features
        adj,           # Adjacency matrices
        agent_id,      # Agent IDs
        rnn_states,    # Recurrent hidden states
        masks          # Episode masks
    )
```

#### For Standard Graph MAPPO:

```
Input: obs + node_obs + adj
    │
    ├─► ACTOR:
    │   ├─ GNNBase(node_obs, adj, agent_id)
    │   │   └─► TransformerConvNet
    │   │       ├─ EmbedConv: Embed entity types + features
    │   │       ├─ TransformerConv layers: Aggregate neighbor info
    │   │       └─ Global pooling: Get aggregated features
    │   │           Output: gnn_features [batch, hidden_dim]
    │   │
    │   ├─ Concatenate: [obs, gnn_features]
    │   ├─ MLP base: Process combined features
    │   ├─ RNN (optional): Temporal processing
    │   └─ ACTLayer: Sample actions from distribution
    │       ├─ Continuous: Gaussian → tanh → [-1, 1]
    │       └─ Discrete: Categorical
    │
    └─► CRITIC:
        ├─ GNNBase(node_obs, adj, agent_id) → Global aggregation
        ├─ MLP base
        ├─ RNN (optional)
        └─ Value head → Value estimate
```

**Policy Forward Pass** (`graph_MAPPOPolicy.py` lines 111-177):

```python
policy.get_actions(cent_obs, obs, node_obs, adj, agent_id, ...)
  ├── actor.forward(obs, node_obs, adj, agent_id, rnn_states, masks)
  │   ├── Convert inputs to torch tensors
  │   ├── gnn_base(node_obs, adj, agent_id) → node embeddings
  │   ├── Concatenate: [obs, gnn_features]
  │   ├── base(concatenated) → MLP features
  │   ├── rnn(features, rnn_states, masks) → [optional recurrent processing]
  │   ├── act(features) → Sample actions + log_probs
  │   └── Return: (actions, action_log_probs, rnn_states)
  │
  └── critic.forward(cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks)
      ├── gnn_base(node_obs, adj, share_agent_id) → aggregated graph features
      ├── base(gnn_features + cent_obs) → Value features
      ├── rnn(features, rnn_states_critic, masks) → [optional]
      ├── value_head(features) → Value estimates
      └── Return: (values, rnn_states_critic)
```

#### For MAD Policy:

```
Input: obs + node_obs + adj
    │
    ├─► BASE CONTROLLER:
    │   ├─ Extract: current_pos = obs[:, 0:2]
    │   ├─ Extract: goal_pos = obs[:, 4:6]
    │   ├─ Error: goal_pos - current_pos
    │   └─ u_base = K_p * error  (K_p is learnable)
    │
    ├─► DIRECTION TERM:
    │   ├─ GNNBase(node_obs, adj) → neighbor features
    │   ├─ Concatenate: [obs, gnn_features]
    │   ├─ MLP + RNN → features
    │   ├─ Output: mean, log_std
    │   ├─ Sample: z ~ N(mean, std)
    │   └─ direction = tanh(z)  [bounded to |D| ≤ 1]
    │
    ├─► MAGNITUDE TERM:
    │   ├─ If first step of episode:
    │   │   └─ Initialize x_0 from obs
    │   ├─ LRU.step(v_t, hidden_state)
    │   │   ├─ v_t = x_0 if t=0, else 0
    │   │   ├─ ξ_{t+1} = Λ ξ_t + Γ(Λ) B v_t
    │   │   └─ M_t = Re(C ξ_t) + D
    │   └─ magnitude = |M_t|
    │
    └─► COMBINE:
        actions = u_base + magnitude * direction
```

**MAD Decomposition** (`mad_actor_critic.py` lines 1-310):

**MAD Policy Structure**: `u_total = u_base + u_MAD = K_p*(goal-current) + |M_t(x_0)| * D_t(neighbors)`

**Components:**

1. **BASE CONTROLLER (u_base)** [lines 85-90]:
   - Proportional gain: K_p (learnable, initialized to 1.0)
   - Error: goal_pos - current_pos
   - u_base = K_p * error
   - Provides baseline stability

2. **DIRECTION TERM (D_t)** [lines 233-245]:
   - Same GNN + MLP + RNN pipeline as standard policy
   - Outputs mean and log_std of Gaussian
   - Sample: z ~ N(mean, std)
   - Normalize: direction = tanh(z) ∈ [-1, 1]
   - Ensures |D_t| ≤ 1 bounded

3. **MAGNITUDE TERM (|M_t(x_0)|)** [lines 247-263]:
   - LRU (Linear Recurrent Unit) [see LRU section below]
   - Seeded with initial observation x_0
   - Outputs: magnitude ∈ [0, ∞) at each timestep
   - Architecture:
     - Complex-valued recurrent layer with diagonal transition matrix
     - Eigenvalues |λ_i| < 1 ensure L_p stability
   - Takes zero input after first step

**Episode Reset Handling** [lines 192-206]:
- Detect reset via mask == 0
- Re-initialize x_0 (initial obs)
- Reset LRU hidden state
- Ensures fresh magnitude computation per episode

**Key Insight:** MAD decomposes control into:
- **Stability** (u_base + magnitude via LRU with |λ| < 1)
- **Reactivity** (direction from GNN observing neighbors)

---

### PHASE 2 & 3: STEP & INSERT

```python
# Execute actions in environment
obs, agent_id, node_obs, adj, rewards, dones, infos = env.step(actions)

# Store everything in buffer
buffer.insert(
    share_obs=cent_obs,
    obs=obs,
    node_obs=node_obs,
    adj=adj,
    rnn_states=rnn_states,
    actions=actions,
    action_log_probs=action_log_probs,
    value_preds=values,
    rewards=rewards,
    masks=masks,
    ...
)
```

**GraphReplayBuffer** (`utils/graph_buffer.py` lines 19-186):

```
GraphReplayBuffer stores for each timestep (0 to episode_length):
  ├── share_obs: [T+1, n_threads, n_agents, *share_obs_shape]
  ├── obs: [T+1, n_threads, n_agents, *obs_shape]
  ├── node_obs: [T+1, n_threads, n_agents, *node_obs_shape]
  ├── adj: [T+1, n_threads, n_agents, *adj_shape]
  ├── agent_id & share_agent_id
  ├── rnn_states & rnn_states_critic
  ├── actions, action_log_probs, values
  ├── rewards & masks
  └── ... (more fields)
```

---

### PHASE 4: COMPUTE & TRAIN

#### 4a. Compute Returns:

```python
# Get value estimate for next state
next_value = self.policy.get_values(next_obs, next_node_obs, next_adj, ...)

# Compute returns using GAE (Generalized Advantage Estimation)
self.buffer.compute_returns(next_value, self.value_normalizer)

# Inside buffer:
for step in reversed(range(T)):
    delta = rewards[step] + gamma * V[step+1] - V[step]
    advantages[step] = delta + gamma * lambda * advantages[step+1]
    returns[step] = advantages[step] + V[step]
```

#### 4b. Train with PPO:

**GR_MAPPO.train() Loop** (`graph_mappo.py` lines 113-210+):

```python
# In graph_mappo.py:train()
trainer.train(buffer)
  ├── For each PPO epoch (default 10):
  │   └── Generate mini-batches from buffer
  │       └── For each mini-batch:
  │           ├── EVALUATE OLD ACTIONS:
  │           │   └── policy.evaluate_actions(sample)
  │           │   └── Recomputes action log_probs & entropy
  │           │       (for standard policy: straightforward)
  │           │       (for MAD policy: invert MAD to recover direction)
  │           │
  │           ├── CALCULATE LOSSES:
  │           │   ├── Policy Loss (PPO clipped):
  │           │   │   └── policy_loss = -min(surr1, surr2)
  │           │   │   └── surr1 = importance_weight * advantage
  │           │   │   └── surr2 = clipped_importance_weight * advantage
  │           │   │
  │           │   └── Value Loss:
  │           │       ├── Normalize returns if using PopArt/ValueNorm
  │           │       ├── Clip value predictions
  │           │       └── Use MSE or Huber loss
  │           │
  │           ├── BACKWARD & OPTIMIZE:
  │           │   ├── Backward: (policy_loss - entropy_coef * entropy).backward()
  │           │   ├── Clip gradients
  │           │   └── optimizer.step()
  │           │
  │           └── Repeat for critic
  │
  └── Return train metrics (loss, entropy, grad_norm, etc.)
```

**Detailed Training Step:**

```python
for epoch in range(ppo_epochs):  # default: 10
    for mini_batch in buffer.get_mini_batches():

        # Re-evaluate actions with current policy
        values, action_log_probs, dist_entropy = \
            policy.evaluate_actions(mini_batch)

        # Importance sampling ratio
        ratio = torch.exp(action_log_probs - old_action_log_probs)

        # Clipped surrogate objective (PPO)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_param, 1+clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        value_pred_clipped = old_values + torch.clamp(
            values - old_values, -clip_param, clip_param)
        value_loss1 = (values - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)
        value_loss = torch.max(value_loss1, value_loss2).mean()

        # Total loss
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * dist_entropy

        # Backward & optimize
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
```

**Value Loss Calculation** (lines 63-112):

```python
cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
  ├── Compute clipped value predictions
  ├── Normalize returns if ValueNorm/PopArt enabled
  ├── Compute error (predicted vs actual returns)
  ├── Apply Huber or MSE loss
  ├── Clip to prevent large swings
  └── Apply active masks to ignore dead agents
```

---

## GNN Processing Deep Dive

### GNN Architecture: `algorithms/utils/gnn.py`

**GNNBase Wrapper** (lines 476-560+):

```python
class GNNBase:
    def __init__(self, node_obs_shape, edge_dim, graph_aggr='global'):
        self.net = TransformerConvNet(
            in_channels=node_feat_dim,
            edge_dim=edge_dim,
            hidden_channels=64,
            num_layers=2,
            graph_aggr=graph_aggr  # 'node' or 'global'
        )

    def forward(self, node_obs, adj, agent_id):
        # Convert to PyTorch Geometric format
        batch_data = self.to_pyg_batch(node_obs, adj, agent_id)

        # Process through GNN
        node_embeddings = self.net(batch_data)

        return node_embeddings  # [batch, hidden_dim]
```

**TransformerConvNet Forward Pass** (lines 275-304):

```
TransformerConvNet.forward(batch: PyG_Data)
  ├── Input: PyG Batch object with:
  │   ├── x: node features [total_nodes, node_feat_dim]
  │   ├── edge_index: [2, num_edges] - (src, dst) edge pairs
  │   ├── edge_attr: [num_edges, edge_feat_dim] - edge weights/distances
  │   └── batch: [total_nodes] - batch assignment indices
  │
  ├── embed_layer(x, edge_index, edge_attr) → EmbedConv
  │   ├── Extract node features & entity types from x
  │   │   └── x = [node_feat, entity_type_id]
  │   ├── Embed entity type: entity_embed(type_id) → [embed_dim]
  │   ├── Concatenate: [node_feat, entity_embed, edge_attr]
  │   └── Pass through MLPs → Enhanced node features
  │
  ├── For each TransformerConv layer:
  │   ├── Apply transformer attention on graph
  │   ├── Multi-head attention aggregates neighbor features
  │   └── Output: updated node features
  │
  ├── Aggregation (based on graph_aggr parameter):
  │   ├── If 'node': Return node-specific features [total_nodes, hidden_dim]
  │   └── If 'global': Apply global pooling → [batch_size, hidden_dim]
  │       ├── mean_pool: Average all node features per graph
  │       ├── max_pool: Take maximum across nodes
  │       └── add_pool: Sum all node features
  │
  └── Return aggregated node embeddings
```

**GNN Layers:**

1. **EmbedConv** (custom layer at `gnn.py:21`):
   - Embeds entity types (agent=0, goal=1, obstacle=2)
   - Combines node features + entity embeddings + edge features
   - Initial message passing layer

2. **TransformerConv** layers:
   - Multi-head attention over graph neighbors
   - Each node attends to connected neighbors
   - Allows information to flow between entities

3. **Aggregation**:
   - `graph_aggr='global'`: Pool all nodes → single vector per graph
   - `graph_aggr='node'`: Keep per-node features

**Usage in Actor** (from `graph_actor_critic.py` lines 170-172):

```python
nbd_features = self.gnn_base(node_obs, adj, agent_id)
# nbd_features: [batch, hidden_dim] if graph_aggr='global'
#            or [batch*n_agents, hidden_dim] if graph_aggr='node'

actor_features = torch.cat([obs, nbd_features], dim=1)
actor_features = self.base(actor_features)  # MLP
```

**Why this matters:**
- Agents can "see" through the GNN what nearby agents are doing
- Enables coordination without centralized communication
- Scales to arbitrary numbers of agents

---

### LRU Magnitude Term: `algorithms/utils/lru.py`

**Purpose:** Generate L_p-stable magnitude signal seeded with initial condition

**Architecture** (lines 31-70):

```
LRU(input_dim, hidden_dim, output_dim)

State Transition Matrix Λ (diagonal, complex-valued):
  └── λ_i = r_i * exp(iθ_i)
  └── r_i ∈ (0, 1) via sigmoid → Ensures |λ_i| < 1
  └── θ_i ∈ [0, 2π] phase

Dynamics [lines 159-213]:
  ξ_{t+1} = Λ ξ_t + Γ(Λ) B v_t

  where:
  ├── ξ_t: complex hidden state [batch, hidden_dim, 2]
  ├── Λ: diagonal transition matrix
  ├── Γ(Λ) = sqrt(1 - |λ|²): normalization
  ├── B: input matrix [hidden_dim, input_dim]
  ├── v_t = x_0 if t=0, else 0 (seeded with x_0 only first step)

Output:
  M_t = Re(C ξ_t) + D v_t + F v_t
  └── Takes real part of hidden state output
  └── Feedthrough terms for direct input influence
```

**Usage in MAD_Actor** (lines 260-263):

```python
magnitude, self.lru_hidden = self.lru.step(v_t, self.lru_hidden, is_first_step)
magnitude = torch.abs(magnitude)  # Ensure positive
u_mad = magnitude * direction
```

---

## Component Relationships

### System Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│                    TRAINING ECOSYSTEM                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │ Environment  │────────►│    Runner    │                │
│  │  (GraphMPE)  │         │ (GMPERunner) │                │
│  └──────────────┘         └───────┬──────┘                │
│         │                         │                        │
│         │ obs, node_obs, adj      │                        │
│         │                         │                        │
│         └─────────────┬───────────┘                        │
│                       │                                    │
│              ┌────────▼────────┐                           │
│              │  ReplayBuffer   │                           │
│              │  (GraphBuffer)  │                           │
│              └────────┬────────┘                           │
│                       │                                    │
│              ┌────────▼────────┐                           │
│              │     Policy      │                           │
│              │  ┌──────────┐   │                           │
│              │  │  Actor   │   │◄─── GNNBase               │
│              │  │  (GNN +  │   │     (TransformerConv)     │
│              │  │   MLP)   │   │                           │
│              │  └──────────┘   │                           │
│              │  ┌──────────┐   │                           │
│              │  │  Critic  │   │◄─── GNNBase               │
│              │  │  (GNN +  │   │     (Global pool)         │
│              │  │   MLP)   │   │                           │
│              │  └──────────┘   │                           │
│              │                 │                           │
│              │ [MAD Addition]  │                           │
│              │  ┌──────────┐   │                           │
│              │  │   LRU    │   │◄─── Magnitude (L_p-stable)│
│              │  │ Magnitude│   │                           │
│              │  └──────────┘   │                           │
│              └────────┬────────┘                           │
│                       │                                    │
│              ┌────────▼────────┐                           │
│              │ Training Algo   │                           │
│              │  (GR_MAPPO)     │                           │
│              │   PPO Updates   │                           │
│              └─────────────────┘                           │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Complete Data Flow Diagram

```
┌──────────────────────────────────┐
│   Environment (GraphMPEEnv)      │
│  Scenario: navigation_graph.py   │
└────────────┬─────────────────────┘
             │ Returns: obs, node_obs, adj, agent_id, rewards, dones
             ▼
┌──────────────────────────────────┐
│   GMPERunner.collect()           │
│  - Gather obs into batch         │
│  - Call policy.get_actions()     │
└────────────┬─────────────────────┘
             │ Actions, values, log_probs, rnn_states
             ▼
┌──────────────────────────────────┐
│   Policy Network                 │
│  (GR_MAPPOPolicy or MAD_Policy)  │
│                                  │
│  ├─ GR_Actor                     │
│  │   ├─ GNNBase (with Embed)     │
│  │   │   └─ TransformerConvNet   │
│  │   ├─ MLP base                 │
│  │   ├─ RNN (optional)           │
│  │   └─ ACTLayer                 │
│  │       └─ Continuous: tanh     │
│  │       └─ Discrete: softmax    │
│  │                              │
│  └─ GR_Critic                    │
│      ├─ GNNBase                  │
│      ├─ MLP base                 │
│      ├─ RNN (optional)           │
│      └─ Value head               │
│                                  │
│  [MAD-specific additions]        │
│  ├─ LRU for magnitude            │
│  ├─ Direction mean/std outputs   │
│  └─ Base controller (K_p)        │
└────────────┬─────────────────────┘
             │ actions, values, action_log_probs
             ▼
┌──────────────────────────────────┐
│  GMPERunner.insert()             │
│  Store in GraphReplayBuffer      │
└────────────┬─────────────────────┘
             │ After episode_length steps
             ▼
┌──────────────────────────────────┐
│  GMPERunner.compute()            │
│  - Get next values               │
│  - Compute returns via GAE       │
│  - Advantages = returns - values │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  GR_MAPPO.train()                │
│  PPO Update Loop                 │
│                                  │
│  For each epoch:                 │
│  ├─ For each mini-batch:         │
│  │  ├─ Evaluate actions          │
│  │  ├─ Compute policy loss       │
│  │  ├─ Compute value loss        │
│  │  ├─ Backward pass             │
│  │  └─ Gradient clip & step      │
│  │                               │
│  └─ Return metrics               │
└──────────────────────────────────┘
```

---

## File Reference

### Workflow Summary by File

| **Component** | **File** | **Purpose** |
|---------------|----------|-------------|
| **Entry** | `scripts/train_mpe.py` | Parse config, create env, launch runner |
| **Runner Base** | `runner/shared/base_runner.py` | Initialize policy, trainer, buffer |
| **Graph Runner** | `runner/shared/graph_mpe_runner.py` | Orchestrate collect→insert→compute→train |
| **Environment** | `multiagent/MPE_env.py` | Wrap MPE, return graph observations |
| **Scenario** | `multiagent/custom_scenarios/navigation_graph.py` | Define world, observations, rewards |
| **Policy** | `algorithms/graph_MAPPOPolicy.py` | Actor-critic with GNN |
| **MAD Policy** | `algorithms/mad_MAPPOPolicy.py` | MAD variant with LRU + base controller |
| **Actor/Critic** | `algorithms/graph_actor_critic.py` | Networks with GNN aggregation |
| **MAD Actors** | `algorithms/mad_actor_critic.py` | MAD-specific actor with magnitude/direction |
| **GNN** | `algorithms/utils/gnn.py` | TransformerConv, EmbedConv layers |
| **LRU** | `algorithms/utils/lru.py` | L_p-stable magnitude term |
| **Buffer** | `utils/graph_buffer.py` | Store graph observations |
| **Algorithm** | `algorithms/graph_mappo.py` | PPO training loop |
| **Config** | `config.py` | Hyperparameter configuration parser |

### Critical File Locations

| Component | File Path |
|-----------|-----------|
| Main entry | `onpolicy/scripts/train_mpe.py` |
| Runner base | `onpolicy/runner/shared/base_runner.py` |
| Graph runner | `onpolicy/runner/shared/graph_mpe_runner.py` |
| Standard policy | `onpolicy/algorithms/graph_MAPPOPolicy.py` |
| MAD policy | `onpolicy/algorithms/mad_MAPPOPolicy.py` |
| Standard actor-critic | `onpolicy/algorithms/graph_actor_critic.py` |
| MAD actor-critic | `onpolicy/algorithms/mad_actor_critic.py` |
| GNN modules | `onpolicy/algorithms/utils/gnn.py` |
| LRU magnitude | `onpolicy/algorithms/utils/lru.py` |
| Replay buffer | `onpolicy/utils/graph_buffer.py` |
| Training algorithm | `onpolicy/algorithms/graph_mappo.py` |
| Environment wrapper | `multiagent/MPE_env.py` |
| Graph scenario | `multiagent/custom_scenarios/navigation_graph.py` |
| Config | `onpolicy/config.py` |

---

## Key Differences: Standard MAPPO vs MAD vs Graph

### Standard MAPPO (non-graph)
- Local observations only
- No graph structure
- Simple actor-critic

### Graph MAPPO (GR_MAPPO)
- Local observations + node observations + adjacency
- GNN aggregates neighbor information
- Enables information sharing between nearby agents
- Scales better than fully centralized critics

### MAD Policy (Magnitude & Direction)
- Decomposes action into magnitude and direction
- Magnitude from LRU ensures L_p stability
- Direction from GNN provides reactive control
- Base controller adds baseline stability
- Better for constrained/safety-critical tasks

---

## Complete Execution Flow

### Execution Checklist

**When training starts:**
1. Config parsed with graph-specific params (graph_feat_type, gnn_hidden_size, etc.)
2. Environments created and parallelized (typically 128 threads)
3. Policy networks initialized with proper shapes
4. Buffer allocated for episode_length steps × n_threads × n_agents
5. Warmup: collect initial observations
6. **Training loop**: collect → insert → compute → train
7. At each step: GNN aggregates graph info → Actor produces actions → Critic evaluates
8. After episode: Returns computed, advantages calculated, PPO update applied
9. Gradient clipping, value normalization, entropy bonus applied
10. Models saved at intervals

### Step-by-Step Execution

1. **Start**: `python train_mpe.py --env_name GraphMPE --use_mad_policy`
2. **Initialize**: Create 128 parallel environments, policy networks, buffer
3. **Warmup**: Collect initial observations
4. **Episode Loop** (25 steps per episode):
   - **Collect**: Policy sees local obs + GNN aggregates neighbors → outputs actions
   - **Step**: Environment executes actions → returns new obs + rewards
   - **Insert**: Store transition in buffer
5. **After Episode**:
   - **Compute**: Calculate returns using GAE
   - **Train**: 10 PPO epochs, update actor & critic
6. **Repeat**: For 2M environment steps
7. **Save**: Checkpoint models periodically

---

## Key Insights

### 1. Scalability
GNN aggregation means computational cost doesn't explode with more agents (only local neighborhoods matter). Each agent processes its local graph independently.

### 2. Decentralization
Each agent has its own actor using only local info, enabling true multi-agent deployment without requiring centralized coordination at execution time.

### 3. Information Sharing
GNN allows agents to implicitly coordinate by "seeing" nearby agents' positions/velocities through graph message passing, without explicit communication.

### 4. MAD Stability
The LRU magnitude term ensures the policy has L_p-stability guarantees by construction (eigenvalues |λ| < 1), making it suitable for safety-critical applications.

### 5. Graph Flexibility
The adjacency matrix can encode different connectivity patterns:
- Sensing radius (physical proximity)
- Communication topology (who can talk to whom)
- Task-specific relationships

### 6. Parallel Efficiency
By running 128 parallel environments, the framework efficiently explores the state space and collects diverse experiences for robust training.

### 7. Modular Design
The separation between environment, policy, buffer, and algorithm makes it easy to:
- Swap different GNN architectures
- Add new scenarios
- Implement new policy types
- Modify reward structures

---

## Related Documentation

- **`README.md`**: Main repository documentation with paper details
- **`CLAUDE.md`**: Instructions for Claude Code when working with this repository
- **`MAD_POLICY_README.md`**: Comprehensive MAD policy documentation
- **`MAD_IMPLEMENTATION_SUMMARY.md`**: Implementation details and decisions
- **`BASE_CONTROLLER_INFO.md`**: Base controller structure and customization

---

## References

1. **InforMARL**: Nayak et al. (2023). "Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation". ICML 2023.

2. **MAD Policy**: Furieri et al. (2025). "MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning". arXiv:2504.02565

3. **LRU**: Orvieto et al. (2023). "Resurrecting Recurrent Neural Networks for Long Sequences". ICML 2023.

4. **MAPPO**: Yu et al. (2021). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
