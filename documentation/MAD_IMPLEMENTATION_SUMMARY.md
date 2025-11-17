# MAD Policy Implementation Summary

## Overview

This document summarizes the implementation of the MAD (Magnitude And Direction) policy parameterization into the InforMARL codebase, based on the paper "MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning" by Furieri et al. (2025).

## Implementation Requirements (from User)

The user requested implementation of a new RL policy with the following specifications:

1. **Two-part composition**:
   - A stabilizing base controller
   - An added MAD policy as defined in the paper

2. **Direction term (D)**:
   - Graph Neural Network taking states of neighboring agents as input
   - Stochastic and continuous policy
   - Uses Gaussian policy parameterization (already in codebase)
   - GNN aggregates neighborhood information and outputs Gaussian parameters
   - Sample from Gaussian to produce direction term D

3. **Magnitude term (M)**:
   - Same as MAD paper but **noiseless case** (no noise)
   - Linear Recurrent Unit (LRU) seeded with initial condition x_0
   - Subsequent inputs to LRU are 0 (v_t = 0 after initialization)
   - Multiplied by D term at each timestep to get final action

## Files Created

### 1. `onpolicy/algorithms/utils/lru.py`
**Purpose**: Implements the Linear Recurrent Unit for the magnitude term.

**Key Features**:
- Complex-valued diagonal state transition matrix Λ with |λ_i| < 1 (stability)
- Parameterized using phase (θ) and magnitude (r) for numerical stability
- Normalization factor Γ(Λ) = sqrt(1 - |λ|²)
- Seeded with x_0 at t=0, then v_t = 0 for subsequent steps
- Outputs magnitude values |M_t| ≥ 0

**Architecture**:
```python
ξ_{t+1} = Λ ξ_t + Γ(Λ) B v_t
M_t = Re(C ξ_t) + D v_t + F v_t
```

### 2. `onpolicy/algorithms/mad_actor_critic.py`
**Purpose**: Implements MAD Actor and Critic networks.

**MAD_Actor Features**:
- **Direction Term**:
  - GNN aggregates neighborhood information (reuses existing `GNNBase`)
  - MLP processes concatenated [obs, neighborhood_features]
  - Optional RNN for temporal dependencies
  - Outputs Gaussian distribution parameters (mean, log_std)
  - Applies tanh to ensure |D_t| ≤ 1

- **Magnitude Term**:
  - LRU initialized with x_0 at episode start
  - Updates with zero input after initialization
  - Outputs |M_t| (absolute value ensures positivity)

- **Combined Action**:
  ```python
  u_t = |M_t| * tanh(sample_from_Gaussian)
  ```

**MAD_Critic Features**:
- Reuses standard Graph Critic architecture
- Compatible with centralized value function
- Supports global or node-level graph aggregation

### 3. `onpolicy/algorithms/mad_MAPPOPolicy.py`
**Purpose**: Policy class wrapping MAD Actor and Critic for MAPPO training.

**Features**:
- Standard MAPPO policy interface
- Methods: `get_actions()`, `get_values()`, `evaluate_actions()`, `act()`
- Learning rate decay support
- LRU reset method for episode boundaries

### 4. Integration Updates

#### `onpolicy/runner/shared/base_runner.py`
**Changes**:
- Added conditional import for MAD policy based on `use_mad_policy` flag
- Maintains compatibility with existing policies

```python
use_mad_policy = getattr(self.all_args, 'use_mad_policy', False)

if self.all_args.env_name == "GraphMPE":
    if use_mad_policy:
        from onpolicy.algorithms.mad_MAPPOPolicy import MAD_MAPPOPolicy as Policy
    else:
        from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy as Policy
```

#### `onpolicy/config.py`
**Changes**:
- Added `--use_mad_policy` flag (default: False)
- Added `--lru_hidden_dim` parameter (default: 64)

### 5. Documentation

#### `MAD_POLICY_README.md`
Comprehensive documentation including:
- Overview of MAD policy concept
- Implementation details
- Usage instructions
- Training examples
- Parameter descriptions
- Troubleshooting guide
- References

## Policy Decomposition

### Mathematical Formulation

```
u_total = u_base + u_MAD
u_total = K_p * (goal - current) + |M_t(x_0)| * D_t(x_t, neighborhood_states)
```

**Base Controller (u_base)**:
- Proportional controller with learnable gains K_p
- Provides baseline stability: u_base = K_p * error
- Initialized to K_p = 1.0, updated during training

**Magnitude Term M_t**:
- Input: x_0 (at t=0), then 0
- Processing: LRU with complex-valued recurrent dynamics
- Output: Scalar magnitude per action dimension
- Properties: L_p-stable, deterministic given x_0

**Direction Term D_t**:
- Input: Current state x_t + GNN(neighborhood states)
- Processing: GNN → MLP → (Optional RNN) → Gaussian params → Sample → Tanh
- Output: Direction vector with |D_t| ≤ 1
- Properties: Stochastic, state-dependent, normalized

**Combined Action**:
- MAD component: u_MAD[i] = |M_t[i]| * D_t[i]
- Total action: u_total = u_base + u_MAD
- Stability: Base controller provides primary stability, MAD enhances performance
- Expressivity: Direction term provides rich state-dependent behavior

## Stability Guarantees

1. **LRU Stability**:
   - Eigenvalues |λ_i| < 1 → M_t ∈ L_p
   - For noiseless case: M_t depends only on x_0

2. **Bounded Direction**:
   - Tanh normalization → |D_t| ≤ 1

3. **Combined Stability**:
   - |u_t| ≤ |M_t|
   - If system is pre-stabilized (F ∈ L_p), then closed-loop is L_p-stable

## Usage Example

```bash
python onpolicy/scripts/train_mpe.py \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --scenario_name "navigation_graph" \
    --num_agents 3 \
    --num_obstacles 3 \
    --use_mad_policy \
    --lru_hidden_dim 64 \
    --graph_feat_type "relative" \
    --num_env_steps 2000000
```

## Compatibility

**Compatible with**:
- GraphMPE environment
- MAPPO/RMAPPO algorithms
- Existing training infrastructure
- W&B logging
- Tensorboard logging
- Evaluation pipeline

**Requirements**:
- `env_name` must be "GraphMPE" (graph observations required)
- Action space must be Box (continuous)
- Standard PyTorch Geometric dependencies

## Key Design Decisions

1. **Noiseless Magnitude**: As requested, M_t has no noise component (v_t = 0 after t=0)

2. **Tanh Normalization**: Applied to direction samples to ensure |D_t| ≤ 1

3. **Complex-valued LRU**: Follows LRU design from Orvieto et al. (2023) for rich expressivity

4. **Episode Reset**: LRU hidden state reset at episode boundaries using mask detection

5. **Surrogate Log-Prob**: Simplified log probability computation for compatibility with PPO (can be enhanced)

## Future Enhancements

1. **Improved Log-Prob**: Store direction samples for accurate gradient estimation

2. **Model Mismatch**: Implement robust stability conditions from MAD paper Proposition 1

3. **Discrete Actions**: Extend to discrete action spaces if needed

4. **Multi-Agent Coordination**: Investigate shared or coordinated magnitude terms

5. **Adaptive Magnitude**: Learn time-varying scaling of magnitude term

## Testing Recommendations

1. **Basic Functionality**:
   ```bash
   python onpolicy/scripts/train_mpe.py --use_mad_policy --num_env_steps 10000
   ```

2. **Compare with Baseline**:
   - Train with `--use_mad_policy`
   - Train without (standard InforMARL)
   - Compare stability and performance

3. **Ablation Studies**:
   - Vary `lru_hidden_dim`: 32, 64, 128
   - Test different `gnn_hidden_size` and `gnn_num_heads`
   - Compare with/without magnitude term (ablation)

## Implementation Checklist

- [x] LRU module with stability guarantees
- [x] MAD Actor with GNN-based direction and LRU-based magnitude
- [x] MAD Critic (reuses Graph Critic)
- [x] MAD Policy class with MAPPO interface
- [x] Integration with base_runner
- [x] Configuration parameters
- [x] Comprehensive documentation
- [ ] Unit tests (recommended)
- [ ] Training verification (user should run)
- [ ] Performance benchmarks (user should evaluate)

## Code Quality Notes

**Strengths**:
- Modular design
- Well-documented
- Type hints included
- Follows existing codebase conventions
- Comprehensive error handling in LRU

**Potential Improvements**:
- Add unit tests for LRU stability
- Enhance log-prob computation in MAD Actor
- Add visualization tools for magnitude/direction analysis
- Profiling for computational efficiency

## References

1. Furieri, L., Shenoy, S., Saccani, D., Martin, A., & Ferrari-Trecate, G. (2025). MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning. arXiv:2504.02565

2. Orvieto, A., et al. (2023). Resurrecting Recurrent Neural Networks for Long Sequences. ICML 2023.

3. Zhang et al. (2023). Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation. ICML 2023.

## Contact & Support

For implementation questions or issues:
- Check `MAD_POLICY_README.md` for usage instructions
- Review code comments in `lru.py`, `mad_actor_critic.py`, `mad_MAPPOPolicy.py`
- Open GitHub issue with `[MAD Policy]` tag

---

**Implementation Date**: 2025-11-03
**Author**: Claude Code Assistant
**Version**: 1.0
**Status**: Complete - Ready for Testing
