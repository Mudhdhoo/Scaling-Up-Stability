# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

InforMARL is a Graph Neural Network (GNN) framework for Multi-Agent Reinforcement Learning (MARL) with limited local observability. The repository implements the ICML'23 paper "Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation" and includes various baseline MARL algorithms for comparison.

**Key Concept**: InforMARL uses GNNs to aggregate local neighborhood information for both actor and critic networks, enabling decentralized multi-agent navigation and collision avoidance that scales to arbitrary numbers of agents and obstacles.

## Repository Structure

### Core Implementation (`onpolicy/`)
- **`algorithms/`**: MARL algorithm implementations
  - `graph_mappo.py` & `graph_MAPPOPolicy.py`: Graph-based MAPPO (InforMARL's main algorithm)
  - `mappo.py` & `MAPPOPolicy.py`: Standard MAPPO baseline
  - `graph_actor_critic.py`: Graph-based actor-critic networks using GNNs
  - `utils/gnn.py`: GNN modules including `EmbedConv` layer for node/edge feature aggregation
- **`runner/`**: Training orchestration
  - `shared/graph_mpe_runner.py`: Training runner for graph-based environments
  - `shared/mpe_runner.py`: Standard MPE runner
- **`envs/mpe/`**: Modified Multi-Agent Particle Environment
  - `scenarios/`: Various MPE scenarios (spread, tag, adversary, etc.)
- **`utils/`**: Buffers and utilities
  - `graph_buffer.py`: Replay buffer for graph observations (stores adjacency matrices)
  - `shared_buffer.py`: Standard shared replay buffer
- **`scripts/train_mpe.py`**: Main training entry point
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
Shell scripts for running experiments with different configurations (informarl.sh, baselines.sh, compare*.sh, etc.)

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
  - Connectivity determined by sensing radius

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

## Dependencies

Install dependencies from `requirement.txt`:
```bash
pip install gym==0.10.5
pip install torch==1.8.1 torch-geometric==2.0.4 torch-scatter==2.0.7 torch-sparse==0.6.10
pip install wandb tensorboardX numpy==1.19.4 pyglet==1.5.26
```

**PyTorch Geometric with CUDA** (if using GPU):
```bash
TORCH="1.8.0"
CUDA="cu102"  # Adjust based on your CUDA version
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

## Development Notes

### Running Tests
No formal test suite exists. Verify functionality by running short training runs:
```bash
# Quick test (reduced steps)
python onpolicy/scripts/train_mpe.py --num_env_steps 10000 --scenario_name "navigation_graph" --env_name "GraphMPE" --num_agents 3
```

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

### Environment Configuration
Key environment parameters (set in scenario or via args):
- `max_edge_dist`: Sensing radius for graph connectivity (default: 1.0)
- `graph_feat_type`: "relative" (relative coords) vs "global" (absolute coords)
- `collision_rew`, `goal_rew`: Reward shaping parameters
- `world_size`: Environment bounds
- `max_speed`: Agent speed limit

### Code Modifications
When modifying GNN architecture:
1. Update `onpolicy/algorithms/utils/gnn.py` for new GNN layers
2. Modify `graph_actor_critic.py` to integrate new layers into actor/critic
3. Adjust `graph_buffer.py` if changing graph observation format
4. Update `navigation_graph.py` scenario if changing graph construction logic

When adding new scenarios:
1. Create scenario file in `multiagent/custom_scenarios/`
2. Implement required callbacks: `make_world`, `reset_world`, `observation`, `graph_observation`, `reward`, `done`
3. Register in `multiagent/custom_scenarios/__init__.py`
4. Add corresponding wrapper logic in `multiagent/MPE_env.py` if needed
