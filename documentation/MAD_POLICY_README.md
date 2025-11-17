# MAD Policy Implementation for InforMARL

This document describes the implementation of MAD (Magnitude And Direction) policies for stability-constrained reinforcement learning in the InforMARL framework.

## Overview

The MAD policy parameterization, introduced by Furieri et al. (2025) in ["MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning"](https://arxiv.org/abs/2504.02565), provides a policy structure that guarantees closed-loop stability for pre-stabilized nonlinear systems while maintaining expressive state-feedback components.

### Key Concept

The MAD policy combines a **base stabilizing controller** with a learned policy:

```
u_total = u_base + u_MAD
u_total = K_p * (goal - current) + |M_t(x_0)| * D_t(x_t, neighborhood_states)
```

where:
- **u_base**: **Base proportional controller** - Provides baseline stability through position error feedback
  - K_p: Learnable proportional gains (initialized to 1.0)

- **u_MAD**: **MAD policy enhancement** - Optimizes performance on top of base controller
  - **M_t**: Magnitude term - An L_p-stable operator based on LRU seeded with x_0
  - **D_t**: Direction term - GNN-based stochastic policy (|D_t| ≤ 1)

## Implementation Components

### 1. Linear Recurrent Unit (LRU)

Located in: `onpolicy/algorithms/utils/lru.py`

The LRU implements a stable recurrent layer with the following dynamics:

```
ξ_{t+1} = Λ ξ_t + Γ(Λ) B v_t
M_t = Re(C ξ_t) + D v_t + F v_t
```

where:
- ξ_t ∈ ℂ^{n_hidden} is the internal complex-valued state
- Λ is a diagonal matrix with |λ_i| < 1 (ensuring stability)
- v_t = x_0 at t=0, then v_t = 0 for subsequent steps

**Key Features:**
- Stability guaranteed by construction (|λ| < 1)
- Seeded with initial conditions
- Produces magnitude values over time

### 2. MAD Actor

Located in: `onpolicy/algorithms/mad_actor_critic.py`

The MAD Actor implements the policy decomposition:

**Direction Term (D_t):**
- Uses GNN to aggregate neighborhood information
- Outputs Gaussian distribution parameters (mean, std)
- Applies tanh normalization to ensure |D_t| ≤ 1
- Samples stochastic direction from learned distribution

**Magnitude Term (M_t):**
- LRU initialized with initial state x_0
- Outputs positive magnitude values
- Ensures L_p stability through LRU design

**Combined Output:**
- Computes u_t = |M_t| * D_t
- Maintains stability guarantees while allowing expressive state-dependent behavior

### 3. MAD Policy Class

Located in: `onpolicy/algorithms/mad_MAPPOPolicy.py`

Wraps the MAD Actor and standard Graph Critic for integration with MAPPO training.

**Features:**
- Compatible with existing MAPPO trainer
- Supports episode reset for LRU reinitialization
- Standard policy interface for seamless integration

## Usage

### Training with MAD Policy

To train an agent using the MAD policy, use the `--use_mad_policy` flag:

```bash
python onpolicy/scripts/train_mpe.py \
    --use_valuenorm --use_popart \
    --project_name "mad_informarl" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --seed 0 \
    --experiment_name "mad_navigation" \
    --scenario_name "navigation_graph" \
    --num_agents 3 \
    --num_obstacles 3 \
    --collision_rew 5 \
    --n_training_threads 1 \
    --n_rollout_threads 128 \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 2000000 \
    --ppo_epoch 10 \
    --use_ReLU \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --user_name "marl" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --auto_mini_batch_size \
    --target_mini_batch_size 128 \
    --use_mad_policy \
    --lru_hidden_dim 64
```

### Key Parameters

#### MAD-Specific Parameters:
- `--use_mad_policy`: Enable MAD policy (default: False)
- `--use_base_controller`: Include proportional base controller (default: True)
  - If True: action = u_base + u_MAD (recommended for navigation)
  - If False: action = u_MAD only (for pure MAD learning)
- `--lru_hidden_dim`: Hidden dimension for LRU magnitude term (default: 64)

#### Standard InforMARL Parameters:
- `--env_name "GraphMPE"`: Use graph-compatible environment (required for MAD)
- `--scenario_name "navigation_graph"`: Graph navigation scenario
- `--graph_feat_type`: "relative" or "global" features
- `--num_agents`: Number of agents in the environment
- `--num_obstacles`: Number of obstacles

## Architecture Details

### Information Flow

1. **Episode Start**: LRU is seeded with initial observation x_0
2. **Each Timestep**:
   - **Base Controller**: u_base = K_p * (goal - current_pos)
   - **Direction**: GNN processes local neighborhood graph → MLP → Gaussian params → Sample → Tanh
   - **Magnitude**: LRU updates with zero input → Outputs magnitude
   - **MAD Policy**: u_MAD = |M_t| * D_t
   - **Total Action**: u_total = u_base + u_MAD
3. **Episode End**: LRU hidden state reset

### Stability Guarantees

The MAD policy ensures L_p stability through:

1. **LRU Stability**: Diagonal state transition matrix Λ with |λ_i| < 1
2. **Bounded Direction**: Tanh normalization ensures |D_t| ≤ 1
3. **Combined Stability**: |u_t| ≤ |M_t|, and M_t ∈ L_p by LRU design

For pre-stabilized systems (F ∈ L_p), this guarantees closed-loop stability.

## Implementation Notes

### Current Limitations

1. **Log Probability Computation**: The current implementation uses a simplified surrogate for log probabilities during `evaluate_actions`. For more accurate gradient estimation, consider storing the sampled direction values during the forward pass.

2. **Episode Resets**: The LRU must be properly reset at episode boundaries. This is currently handled through mask detection in the actor's forward method.

3. **Continuous Actions Only**: The current implementation supports Box action spaces only. Discrete action spaces would require adaptation.

### Future Enhancements

1. **Robust Stability**: Extend to handle model mismatch as described in the MAD paper (Proposition 1)
2. **Better Log-Prob Estimation**: Store direction samples for accurate log probability computation
3. **Adaptive Magnitude**: Learn time-varying magnitude scaling
4. **Multi-Agent Coordination**: Investigate shared magnitude terms across agents

## Comparison with Standard InforMARL

| Feature | Standard InforMARL | MAD Policy |
|---------|-------------------|------------|
| Stability Guarantees | Requires stabilizing base controller separately | Built-in through LRU magnitude term |
| State Feedback | Full state feedback | Direction only (magnitude from initial condition) |
| Stochasticity | Gaussian policy on full action | Gaussian on direction, deterministic magnitude |
| Training Complexity | Standard PPO | Standard PPO (compatible) |
| Generalization | High | High (enhanced by explicit state-feedback direction) |

## References

1. Furieri, L., Shenoy, S., Saccani, D., Martin, A., & Ferrari-Trecate, G. (2025). MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning. arXiv:2504.02565

2. Orvieto, A., Smith, S. L., Gu, A., Fernando, A., Gulcehre, C., Pascanu, R., & De, S. (2023). Resurrecting Recurrent Neural Networks for Long Sequences. ICML 2023.

3. InforMARL: Zhang et al. (2023). Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation. ICML 2023.

## Example Training Script

A complete example training script is provided below:

```bash
#!/bin/bash

# MAD Policy Training for Multi-Agent Navigation
# This script trains agents using the MAD policy parameterization
# for stability-constrained RL

python -u onpolicy/scripts/train_mpe.py \
    --use_valuenorm \
    --use_popart \
    --project_name "mad_experiments" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --seed 42 \
    --experiment_name "mad_nav_3agents_3obs" \
    --scenario_name "navigation_graph" \
    --num_agents 3 \
    --num_obstacles 3 \
    --collision_rew 5 \
    --goal_rew 10 \
    --n_training_threads 1 \
    --n_rollout_threads 128 \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 2000000 \
    --ppo_epoch 10 \
    --use_ReLU \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --auto_mini_batch_size \
    --target_mini_batch_size 128 \
    --use_mad_policy \
    --lru_hidden_dim 64 \
    --use_wandb "False"
```

Save this as `scripts/train_mad.sh` and run with:
```bash
chmod +x scripts/train_mad.sh
./scripts/train_mad.sh
```

## Troubleshooting

### Issue: LRU not resetting between episodes
**Solution**: Check that masks are properly passed to the actor's forward method. The actor detects episode resets through `masks == 0`.

### Issue: Training instability
**Solution**:
- Reduce `lr` and `critic_lr`
- Increase `lru_hidden_dim` for more expressive magnitude term
- Adjust `max_edge_dist` to control neighborhood size

### Issue: Poor performance compared to standard policy
**Solution**:
- Ensure the system is pre-stabilized (has a stabilizing base controller)
- Try different `lru_hidden_dim` values (32, 64, 128)
- Increase GNN expressiveness (`gnn_hidden_size`, `gnn_num_heads`)

## Contact

For questions or issues related to the MAD policy implementation, please open an issue on the repository or refer to the original MAD paper.
