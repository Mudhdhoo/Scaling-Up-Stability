# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

InforMARL is a Graph Neural Network (GNN) framework for Multi-Agent Reinforcement Learning (MARL) with limited local observability. The repository implements the ICML'23 paper "Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation" and includes various baseline MARL algorithms for comparison. It also includes an implementation of the MAD (Magnitude And Direction) policy parameterization for stability-constrained RL.

**Key Concept**: InforMARL uses GNNs to aggregate local neighborhood information for both actor and critic networks, enabling decentralized multi-agent navigation and collision avoidance that scales to arbitrary numbers of agents and obstacles.

## Repository Structure

### Core Implementation (`onpolicy/`)
- **`algorithms/`**: MARL algorithm implementations
  - `graph_mappo.py` & `graph_MAPPOPolicy.py`: Graph-based MAPPO (InforMARL's main algorithm)
  - `mad_MAPPOPolicy.py` & `mad_actor_critic.py`: MAD policy with stability guarantees
  - `mappo.py` & `MAPPOPolicy.py`: Standard MAPPO baseline
  - `graph_actor_critic.py`: Graph-based actor-critic networks using GNNs
  - `utils/gnn.py`: GNN modules including `EmbedConv` layer for node/edge feature aggregation
  - `utils/lru.py`: Linear Recurrent Unit for MAD magnitude term with L_p-stability
- **`runner/`**: Training orchestration
  - `shared/graph_mpe_runner.py`: Training runner for graph-based environments
  - `shared/mpe_runner.py`: Standard MPE runner
  - `shared/base_runner.py`: Base runner with MAD policy support
- **`envs/mpe/`**: Modified Multi-Agent Particle Environment
  - `scenarios/`: Various MPE scenarios (spread, tag, adversary, etc.)
- **`utils/`**: Buffers and utilities
  - `graph_buffer.py`: Replay buffer for graph observations (stores adjacency matrices)
  - `shared_buffer.py`: Standard shared replay buffer
- **`scripts/train_mpe.py`**: Main training entry point
- **`scripts/eval_mpe.py`**: Evaluation script for trained models
- **`config.py`**: Hyperparameter configuration parser

### Graph-Compatible Environment (`multiagent/`)
Custom MPE environment modified to output graph observations:
- **`environment.py`**: Base multi-agent environment classes
- **`MPE_env.py`**: Environment wrappers (`MPEEnv`, `GraphMPEEnv`)
- **`custom_scenarios/`**: Navigation scenarios
  - `navigation_graph.py`: Graph-compatible navigation (returns node obs + adjacency matrices)
  - `navigation.py`, `navigation_gpg.py`, `navigation_dgn.py`: Baseline-specific scenarios
- **`core.py`**: World physics and entity definitions

### Baselines (`baselines/`)
Implementations of comparison algorithms from their original papers:
- **`offpolicy/`**: Off-policy algorithms (MADDPG, MATD3, QMIX, VDN, MQMIX, MVDN)
- **`mpnn/`**: Message Passing Neural Network baselines
- **`gpg/`**: Graph Policy Gradients
- **`dgn/`**: Graph Convolutional RL (DGN)

### Training Scripts (`scripts/`)
Shell scripts for running experiments with different configurations. 34 shell scripts total including:
- `baselines.sh`, `baselines_n.sh`: Baseline comparisons
- `compare*.sh`: Comparison experiments
- `dgn_baseline.sh`, `gpg_baseline.sh`: Specific baseline runs

## Training Commands

### Train InforMARL (Graph MAPPO)
```bash
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "informarl" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 0 \
--experiment_name "informarl" \
--scenario_name "navigation_graph" \
--num_agents 3 \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 128
```

**Key parameters**:
- `--env_name "GraphMPE"`: Use graph-compatible environment
- `--scenario_name "navigation_graph"`: Graph navigation scenario
- `--graph_feat_type`: "relative" (relative positions) or "global" (absolute positions)
- `--use_cent_obs`: Whether critic uses centralized observations
- `--num_agents`, `--num_obstacles`: Environment complexity

### Train MAD Policy (Stability-Constrained RL)
```bash
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "mad_informarl" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 0 \
--experiment_name "mad_navigation" \
--scenario_name "navigation_graph" \
--num_agents 3 \
--num_obstacles 3 \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 128 \
--use_mad_policy \
--use_base_controller \
--lru_hidden_dim 64
```

**MAD-specific parameters**:
- `--use_mad_policy`: Enable MAD policy parameterization (default: False)
- `--use_base_controller`: Include proportional base controller u_base = K_p*(goal-current) (default: True)
- `--lru_hidden_dim`: Hidden dimension for LRU magnitude term (default: 64)

**MAD Policy Structure**: `u_total = u_base + u_MAD = K_p*(goal-current) + |M_t(x_0)| * D_t(neighbors)`
- u_base: Learnable proportional controller for baseline stability
- M_t: LRU-based magnitude term seeded with x_0 (L_p-stable)
- D_t: GNN-based stochastic direction term (|D_t| ≤ 1)

### Train Standard MAPPO Baseline
```bash
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--env_name "MPE" \
--algorithm_name "rmappo" \
--scenario_name "navigation" \
--num_agents 3 \
--n_rollout_threads 128 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4
```

### Train Off-Policy Baselines (MADDPG/MATD3/QMIX/VDN)
```bash
algo="maddpg"  # or "matd3", "qmix", "vdn", "mqmix", "mvdn"

python baselines/offpolicy/scripts/train/train_mpe.py \
--env_name "MPE" \
--algorithm_name ${algo} \
--scenario_name "navigation" \
--num_agents 3 \
--num_landmarks 3 \
--seed 0 \
--episode_length 25 \
--use_soft_update \
--lr 7e-4 \
--num_env_steps 10000000 \
--use_wandb
```

### Train MPNN Baseline
```bash
python baselines/mpnn/nav/main.py \
--n_rollout_threads=128 \
--scenario_name='navigation' \
--use_wandb \
--verbose \
--obs_type 'global'
```

### Train GPG Baseline
```bash
python -u -W ignore baselines/gpg/rl_navigation/main.py \
--env_name "MPE" \
--algorithm_name "gpg" \
--scenario_name "navigation_gpg" \
--num_agents=3 \
--num_env_steps 2000000 \
--use_wandb
```

### Evaluation
```bash
python onpolicy/scripts/eval_mpe.py \
--model_dir <path_to_saved_model> \
--env_name "GraphMPE" \
--scenario_name "navigation_graph" \
--num_agents <num_agents> \
--render_episodes 10
```

### Quick Test (Reduced Steps)
```bash
# Test standard InforMARL
python onpolicy/scripts/train_mpe.py --num_env_steps 10000 --scenario_name "navigation_graph" --env_name "GraphMPE" --num_agents 3

# Test MAD policy
python onpolicy/scripts/train_mpe.py --num_env_steps 10000 --scenario_name "navigation_graph" --env_name "GraphMPE" --num_agents 3 --use_mad_policy
```

## Architecture Details

### Graph Environment Outputs
When using `GraphMPEEnv`, `env.reset()` and `env.step()` return:
- **`obs_n`**: Local observations per agent (position, velocity, relative goal)
- **`agent_id_n`**: Agent IDs for indexing agent-specific features
- **`node_obs_n`**: Graph node features for each agent's local graph
  - Nodes represent entities: agents, goals, obstacles
  - Features: relative position, relative velocity, goal information
- **`adj_n`**: Adjacency matrices defining graph connectivity
  - Inter-agent edges are bidirectional
  - Agent-to-entity edges are unidirectional
  - Connectivity determined by sensing radius (`max_edge_dist`)

### GNN Information Aggregation
1. **Per-Agent Aggregation**: Each agent's GNN processes its local graph (nodes within sensing radius) to produce aggregated feature vector `x_agg^i`
2. **Actor Input**: Concatenation `[obs^i, x_agg^i]` fed to actor network
3. **Global Aggregation**: All agents' `x_agg^i` averaged to get `X_agg` for centralized critic
4. **Critic Input**: `X_agg` (or centralized observations if `use_cent_obs=True`)

### Key GNN Components
- **`EmbedConv`** (onpolicy/algorithms/utils/gnn.py:21): Custom message-passing layer
  - Embeds entity types (agent/goal/obstacle)
  - Processes node and edge features
  - Uses GCN-style message passing
- **Graph Batching**: Handles variable-sized graphs across parallel environments using PyTorch Geometric batching

### MAD Policy Components
Located in `onpolicy/algorithms/`:
- **`utils/lru.py`**: Linear Recurrent Unit (LRU) for magnitude term
  - Complex-valued recurrent layer with diagonal state transition matrix Λ where |λ_i| < 1
  - Ensures L_p-stability by construction
  - Seeded with initial observation x_0, then receives zero input
  - Dynamics: ξ_{t+1} = Λ ξ_t + Γ(Λ) B v_t, where v_t = x_0 at t=0, else 0
- **`mad_actor_critic.py`**: MAD Actor and Critic
  - **Direction Term D_t**: GNN → MLP → Gaussian params → Sample → Tanh (ensures |D_t| ≤ 1)
  - **Magnitude Term M_t**: LRU produces positive magnitude values
  - **Base Controller u_base**: K_p*(goal - current_pos) with learnable gains
  - **Combined**: u_total = u_base + |M_t| * D_t
- **`mad_MAPPOPolicy.py`**: Policy wrapper for MAPPO integration

## Dependencies

Install from `requirement.txt`:
```bash
pip install -r requirement.txt
```

Key dependencies:
- gymnasium (multi-agent particle environment)
- torch, torch-geometric, torch-scatter, torch-sparse (GNN support)
- wandb, tensorboardX (logging)
- pyglet, PyOpenGL, PyQt5 (rendering)

**PyTorch Geometric with CUDA** (if using GPU):
```bash
TORCH="1.8.0"
CUDA="cu102"  # Adjust based on your CUDA version
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

## Development Notes

### Logging
- By default uses W&B (Weights & Biases): `--use_wandb`
- Tensorboard alternative: omit `--use_wandb` flag
- Set `--user_name` for W&B organization

### Rendering
Interactive visualization:
```python
from multiagent.environment import MultiAgentGraphEnv
from multiagent.custom_scenarios.navigation_graph import Scenario

# Create environment (see README.md lines 80-133 for full example)
env = MultiAgentGraphEnv(world, ...)
env.render()  # Creates viewer window

# Step through episode
obs, ids, node_obs, adj = env.reset()
while True:
    actions = get_actions(obs)
    obs, ids, node_obs, adj, rewards, dones, info = env.step(actions)
    env.render()
```

### Common Issues

**MacOS Big Sur pyglet error**:
```bash
pip install --user --upgrade git+http://github.com/pyglet/pyglet@pyglet-1.5-maintenance
```

**OMP initialization error**:
```bash
conda install nomkl
```

**PyTorch Geometric CUDA errors**: Reinstall torch-geometric packages (see Dependencies section)

**MAD Policy LRU not resetting between episodes**: Check that masks are properly passed to the actor's forward method. The actor detects episode resets through `masks == 0`.

### Environment Configuration
Key environment parameters (set in scenario or via args):
- `max_edge_dist`: Sensing radius for graph connectivity (default: 1.0)
- `graph_feat_type`: "relative" (relative coords) vs "global" (absolute coords)
- `collision_rew`, `goal_rew`: Reward shaping parameters
- `world_size`: Environment bounds
- `max_speed`: Agent speed limit

### Code Modifications

#### Modifying GNN Architecture
1. Update `onpolicy/algorithms/utils/gnn.py` for new GNN layers
2. Modify `graph_actor_critic.py` to integrate new layers into actor/critic
3. Adjust `graph_buffer.py` if changing graph observation format
4. Update `navigation_graph.py` scenario if changing graph construction logic

#### Adding New Scenarios
1. Create scenario file in `multiagent/custom_scenarios/`
2. Implement required callbacks: `make_world`, `reset_world`, `observation`, `graph_observation`, `reward`, `done`
3. Register in `multiagent/custom_scenarios/__init__.py`
4. Add corresponding wrapper logic in `multiagent/MPE_env.py` if needed

#### Customizing MAD Base Controller
The base controller assumes observations have structure: `[pos_x, pos_y, vel_x, vel_y, goal_x, goal_y, ...]`

To customize for different observation structures, modify `mad_actor_critic.py` lines 277-278:
```python
# Current implementation
current_pos = obs[:, 0:2]  # Position (x, y)
goal_pos = obs[:, 4:6]     # Goal (x, y)

# Example: Different structure [goal_x, goal_y, pos_x, pos_y, ...]
goal_pos = obs[:, 0:2]
current_pos = obs[:, 2:4]
```

## Important Documentation Files

- **`README.md`**: Main repository documentation with paper details
- **`MAD_POLICY_README.md`**: Comprehensive MAD policy documentation
- **`MAD_IMPLEMENTATION_SUMMARY.md`**: Implementation details and decisions
- **`BASE_CONTROLLER_INFO.md`**: Base controller structure and customization
- **`papers.md`**: Related paper references

## References

1. **InforMARL**: Nayak et al. (2023). "Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation". ICML 2023.

2. **MAD Policy**: Furieri et al. (2025). "MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning". arXiv:2504.02565

3. **LRU**: Orvieto et al. (2023). "Resurrecting Recurrent Neural Networks for Long Sequences". ICML 2023.

4. **Baselines**:
   - MAPPO: Yu et al. (2021). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
   - MADDPG: Lowe et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
   - QMIX: Rashid et al. (2018). "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning"
   - Graph Policy Gradients (GPG): Li et al. (2019)
   - Graph Convolutional RL (DGN): Jiang et al. (2018)
