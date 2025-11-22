# SSM Training Fix: Variable Ordering Verification

## Problem

The SSM/LRU hidden states (`lru_hidden_states_batch`) were being extracted from the buffer but **not passed** to the `evaluate_actions` method during training, preventing proper gradient computation through the SSM magnitude term.

## Solution

Fixed variable ordering and added `lru_hidden_states` parameter to all policy `evaluate_actions` methods for consistency and backward compatibility.

---

## Changes Made

### 1. Updated `graph_mappo.py`

**File**: `onpolicy/algorithms/graph_mappo.py:177-191`

**Before**:
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
    # lru_hidden_states_batch NOT PASSED!
)
```

**After**:
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
    lru_hidden_states_batch  # ✓ NOW PASSED!
)
```

### 2. Updated Policy Signatures for Backward Compatibility

Added `lru_hidden_states=None` parameter to all graph-based MAPPO policies:

#### a. `graph_base_ssm_policy.py` (SSM Policy)

**File**: `onpolicy/algorithms/graph_base_ssm_policy.py:200-214`

Changed parameter name from `ssm_states` to `lru_hidden_states` for consistency:

```python
def evaluate_actions(
    self,
    cent_obs,
    obs,
    node_obs,
    adj,
    agent_id,
    share_agent_id,
    rnn_states_actor,
    rnn_states_critic,
    action,
    masks,
    available_actions=None,
    active_masks=None,
    lru_hidden_states=None,  # ✓ Changed from ssm_states
) -> Tuple[Tensor, Tensor, Tensor]:
```

#### b. `graph_MAPPOPolicy.py` (Standard Graph Policy)

**File**: `onpolicy/algorithms/graph_MAPPOPolicy.py:207-222`

Added `lru_hidden_states=None` parameter (unused but accepted for compatibility):

```python
def evaluate_actions(
    self,
    cent_obs,
    obs,
    node_obs,
    adj,
    agent_id,
    share_agent_id,
    rnn_states_actor,
    rnn_states_critic,
    action,
    masks,
    available_actions=None,
    active_masks=None,
    lru_hidden_states=None,  # ✓ Added for compatibility (unused)
) -> Tuple[Tensor, Tensor, Tensor]:
```

#### c. `graph_base_policy.py` (Base Controller Policy)

**File**: `onpolicy/algorithms/graph_base_policy.py:214-229`

Added `lru_hidden_states=None` parameter (unused but accepted for compatibility):

```python
def evaluate_actions(
    self,
    cent_obs,
    obs,
    node_obs,
    adj,
    agent_id,
    share_agent_id,
    rnn_states_actor,
    rnn_states_critic,
    action,
    masks,
    available_actions=None,
    active_masks=None,
    lru_hidden_states=None,  # ✓ Added for compatibility (unused)
) -> Tuple[Tensor, Tensor, Tensor]:
```

#### d. `mad_MAPPOPolicy.py` (MAD Policy)

**No changes needed** - already has `lru_hidden_states=None` parameter ✓

---

## Variable Ordering Verification

### Buffer Generator Output

**Location**: `onpolicy/utils/graph_buffer.py` (all three generators)

All buffer generators yield **17 variables** in this exact order when `lru_hidden_states` is present:

```python
yield (
    share_obs_batch,           # 1
    obs_batch,                 # 2
    node_obs_batch,            # 3
    adj_batch,                 # 4
    agent_id_batch,            # 5
    share_agent_id_batch,      # 6
    rnn_states_batch,          # 7
    rnn_states_critic_batch,   # 8
    actions_batch,             # 9
    value_preds_batch,         # 10
    return_batch,              # 11
    masks_batch,               # 12
    active_masks_batch,        # 13
    old_action_log_probs_batch,# 14
    adv_targ,                  # 15
    available_actions_batch,   # 16
    lru_hidden_states_batch    # 17 ✓
)
```

When `lru_hidden_states` is NOT present (standard policy), only **16 variables** are yielded (without #17).

### Unpacking in graph_mappo.py

**Location**: `onpolicy/algorithms/graph_mappo.py:140-154`

```python
if len(sample) == 17:
    # MAD/SSM policy with LRU hidden states
    share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, \
    share_agent_id_batch, rnn_states_batch, rnn_states_critic_batch, \
    actions_batch, value_preds_batch, return_batch, masks_batch, \
    active_masks_batch, old_action_log_probs_batch, adv_targ, \
    available_actions_batch, lru_hidden_states_batch = sample
else:
    # Standard policy (16 variables)
    share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, \
    share_agent_id_batch, rnn_states_batch, rnn_states_critic_batch, \
    actions_batch, value_preds_batch, return_batch, masks_batch, \
    active_masks_batch, old_action_log_probs_batch, adv_targ, \
    available_actions_batch = sample
    lru_hidden_states_batch = None
```

**Verification**: Order matches buffer output ✓

### evaluate_actions Call

**Location**: `onpolicy/algorithms/graph_mappo.py:177-191`

```python
values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
    share_obs_batch,           # 1. cent_obs ✓
    obs_batch,                 # 2. obs ✓
    node_obs_batch,            # 3. node_obs ✓
    adj_batch,                 # 4. adj ✓
    agent_id_batch,            # 5. agent_id ✓
    share_agent_id_batch,      # 6. share_agent_id ✓
    rnn_states_batch,          # 7. rnn_states_actor ✓
    rnn_states_critic_batch,   # 8. rnn_states_critic ✓
    actions_batch,             # 9. action ✓
    masks_batch,               # 10. masks ✓
    available_actions_batch,   # 11. available_actions (keyword) ✓
    active_masks_batch,        # 12. active_masks (keyword) ✓
    lru_hidden_states_batch    # 13. lru_hidden_states (keyword) ✓
)
```

### Policy evaluate_actions Signature

All graph-based policies now have this signature:

```python
def evaluate_actions(
    self,
    cent_obs,              # 1 ✓
    obs,                   # 2 ✓
    node_obs,              # 3 ✓
    adj,                   # 4 ✓
    agent_id,              # 5 ✓
    share_agent_id,        # 6 ✓
    rnn_states_actor,      # 7 ✓
    rnn_states_critic,     # 8 ✓
    action,                # 9 ✓
    masks,                 # 10 ✓
    available_actions=None,# 11 ✓
    active_masks=None,     # 12 ✓
    lru_hidden_states=None,# 13 ✓ (NEW!)
) -> Tuple[Tensor, Tensor, Tensor]:
```

**Verification**: Parameter order matches call order ✓✓✓

---

## Testing Checklist

- [x] Variable ordering verified from buffer to policy
- [x] All 13 arguments passed in correct order
- [x] SSM policy accepts `lru_hidden_states`
- [x] MAD policy accepts `lru_hidden_states` (already did)
- [x] Standard policies accept but ignore `lru_hidden_states`
- [x] Backward compatibility maintained
- [ ] **TODO**: Run training to verify SSM gradients flow correctly
- [ ] **TODO**: Verify no errors with standard policies

---

## Why This Fix Is Critical

### Before Fix

During PPO training, when `evaluate_actions` was called without SSM states:

1. ❌ SSM states were not passed to actor
2. ❌ SSM magnitude term was computed with default/wrong states
3. ❌ Gradients didn't flow through correct SSM pathway
4. ❌ Policy couldn't learn to modulate magnitude properly

### After Fix

Now when `evaluate_actions` is called with SSM states:

1. ✓ SSM states from buffer are passed to actor
2. ✓ SSM magnitude term reconstructed with correct hidden states
3. ✓ Gradients flow through proper SSM computation graph
4. ✓ Policy can learn to adjust magnitude based on task

---

## Summary

**Files Modified**:
1. ✓ `onpolicy/algorithms/graph_mappo.py` - Added `lru_hidden_states_batch` to call
2. ✓ `onpolicy/algorithms/graph_base_ssm_policy.py` - Renamed `ssm_states` → `lru_hidden_states`
3. ✓ `onpolicy/algorithms/graph_MAPPOPolicy.py` - Added `lru_hidden_states=None` parameter
4. ✓ `onpolicy/algorithms/graph_base_policy.py` - Added `lru_hidden_states=None` parameter

**Result**:
- SSM/MAD policies can now receive hidden states during training
- Standard policies remain compatible (ignore the extra parameter)
- Variable ordering is consistent throughout the pipeline
- Training should now work correctly for all policy types

---

## Next Steps

1. **Test with SSM policy**: Train an agent with `use_ssm_policy=True` (need to add flag)
2. **Verify gradients**: Check that SSM magnitude term has non-zero gradients during training
3. **Verify standard policies**: Ensure non-SSM/MAD policies still work without errors
4. **Add integration**: Add SSM policy selection to `base_runner.py` (similar to MAD policy)

---

## Related Documentation

- See `SSM_STATE_MANAGEMENT.md` for complete state lifecycle details
- See `CLAUDE.md` for SSM policy training commands and configuration
