"""
MAD (Magnitude And Direction) Actor-Critic implementation.
Implements the MAD policy parameterization from Furieri et al. (2025).
"""
import argparse
from typing import Tuple

import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.gnn import GNNBase, ZeroPreservingGNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.ssm import SSM
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from loguru import logger
import time

def minibatchGenerator(
    obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, max_batch_size: int
):
    """
    Split a big batch into smaller batches.
    """
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size : (i + 1) * max_batch_size],
            node_obs[i * max_batch_size : (i + 1) * max_batch_size],
            adj[i * max_batch_size : (i + 1) * max_batch_size],
            agent_id[i * max_batch_size : (i + 1) * max_batch_size],
        )


class MAD_GR_Actor(nn.Module):
    """
    MAD Graph-based Actor for distributed multi-agent control.

    Implements the policy: a_t = u_base + |M_t(w_t)| * D_t(x_t)

    where:
        - u_base: Proportional base controller K_p * (goal - current_pos)
        - M_t: Magnitude term using GNN + SSM on global state disturbances w_t (L_p-stable)
        - D_t: Direction term using GNN + (optional RNN) on neighbor observations x_t
        - w_t = global_states at t=0 (episode reset), else 0
        - global_states: [vel_x, vel_y, pos_x, pos_y] in global frame (no auxiliary info)
        - |M_t| ensures positive magnitude
        - D_t is normalized via tanh: |D_t| <= 1
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        action_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(MAD_GR_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.K_p = args.kp_val

        obs_shape = get_shape_from_obs_space(obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[1]
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]

        # Get action dimension
        if action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0]
        else:
            raise NotImplementedError("MAD_GR policy currently only supports Box action spaces")

        # State dimensions for magnitude term (exclude auxiliary info like goals)
        # Typically: [vel_x, vel_y, pos_x, pos_y] = 4 dimensions for 2D particle systems
        self.system_state_dim = getattr(args, 'system_state_dim', 4)

        # ---------------- Direction Modules ----------------

        # GNN for direction term - processes neighbor observations x_t
        self.direction_gnn = GNNBase(args, node_obs_shape, edge_dim, args.actor_graph_aggr)
        direction_gnn_out_dim = self.direction_gnn.out_dim

        # MLP for direction term
        direction_mlp_in_dim = direction_gnn_out_dim + obs_shape[0]
        self.direction_mlp = MLPBase(args, obs_shape=None, override_obs_dim=direction_mlp_in_dim)

        # Optional RNN for direction term
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        # ---------------- Magnitude Modules ----------------

        # GNN for magnitude term - processes disturbance observations w_t (state-only)
        # Input: only dynamical system states (position, velocity), no auxiliary info (goals)
        # Uses ZeroPreservingGNNBase to ensure f(0)=0 for L_p-stability
        self.magnitude_gnn = ZeroPreservingGNNBase(args, self.system_state_dim, edge_dim, args.actor_graph_aggr)
        mag_gnn_out_dim = self.magnitude_gnn.out_dim

        # MLP for magnitude term (no concatenation with obs to maintain state-only pathway)
        # Use bias=False, use_layer_norm=False, use_feature_norm=False to ensure zero-preservation
        magnitude_mlp_in_dim = mag_gnn_out_dim
        self.magnitude_mlp = MLPBase(args, obs_shape=None, bias=False, use_layer_norm=False,
                                     use_feature_norm=False, override_obs_dim=magnitude_mlp_in_dim)

        # SSM for magnitude term (L_p-stable temporal processing)
        ssm_state_features = getattr(args, 'ssm_hidden_dim', 64)
        ssm_mlp_hidden = getattr(args, 'ssm_mlp_hidden', 64)

        self.ssm = SSM(
            in_features=self.hidden_size,
            out_features=1,
            state_features=ssm_state_features,
            scan=False,  # Use step-by-step for online execution
            mlp_hidden_size=ssm_mlp_hidden,
        )

        self.to(device)

    def forward(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        ssm_states,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute actions using MAD decomposition: a = u_base + |M(w)| * D(x)

        Args:
            obs: Agent observations [batch, obs_dim]
            node_obs: Neighbor observations (x_t) [batch, num_nodes, node_dim]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            agent_id: Agent IDs [batch]
            rnn_states: RNN hidden states [batch, hidden_dim]
            ssm_states: SSM hidden states [batch, state_features] (complex)
            masks: Episode masks [batch, 1] (0 = reset)
            available_actions: Unused for continuous actions
            deterministic: Whether to sample or use mean

        Returns:
            actions: Output actions [batch, action_dim]
            action_log_probs: Log probabilities [batch, 1]
            rnn_states: Updated RNN states [batch, hidden_dim]
            ssm_states: Updated SSM hidden states [batch, state_features] (complex)
        """
        # Convert inputs to proper format
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        batch_size = obs.shape[0]

        # Detect episode resets (mask == 0)
        reset_mask = (masks.squeeze(-1) == 0)
        reset_indices = reset_mask.nonzero(as_tuple=True)[0]

        # Initialize or reset SSM hidden states
        if ssm_states is None or ssm_states.shape[0] != batch_size:
            # Initialize for all environments
            ssm_states = torch.complex(
                torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
                torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
            )
        elif len(reset_indices) > 0:
            # Reset only environments that are resetting
            ssm_states = ssm_states.clone()  # Clone to avoid in-place modification
            for idx in reset_indices:
                ssm_states[idx] = torch.complex(
                    torch.zeros(self.ssm.LRUR.state_features, device=obs.device),
                    torch.zeros(self.ssm.LRUR.state_features, device=obs.device),
                )

        # ==================== Magnitude Term M_t ====================

        states_obs = node_obs[:, :, :self.system_state_dim]

        # Prepare disturbance observations: w_t = state_obs_global at t=0 (reset), else 0
        w_t_states = torch.where(
            reset_mask.unsqueeze(-1).unsqueeze(-1).expand_as(states_obs),
            states_obs,  # w_0 = states at reset
            torch.zeros_like(states_obs)  # w_t = 0 otherwise
        )

        # Process through GNN with optional batch splitting
        # NOTE: ZeroPreservingGNN ignores edge attributes to maintain f(0)=0 property
        if self.split_batch and (obs.shape[0] > self.max_batch_size):
            batchGenerator = minibatchGenerator(obs, w_t_states, adj, agent_id, self.max_batch_size)
            magnitude_features_list = []
            for batch in batchGenerator:
                obs_batch, w_t_batch, adj_batch, agent_id_batch = batch
                mag_nbd_batch = self.magnitude_gnn(w_t_batch, adj_batch, agent_id_batch)
                mag_feat_batch = self.magnitude_mlp(mag_nbd_batch)
                magnitude_features_list.append(mag_feat_batch)
            magnitude_features = torch.cat(magnitude_features_list, dim=0)
        else:
            magnitude_features = self.magnitude_gnn(w_t_states, adj, agent_id)
            magnitude_features = self.magnitude_mlp(magnitude_features)

        # Pass through SSM (L_p-stable temporal processing)
        magnitude, ssm_states = self.ssm.step(magnitude_features, ssm_states)
        magnitude = torch.abs(magnitude)

        # ==================== Direction Term D_t ====================
        # Process through GNN with optional batch splitting
        if self.split_batch and (obs.shape[0] > self.max_batch_size):
            batchGenerator = minibatchGenerator(obs, node_obs, adj, agent_id, self.max_batch_size)
            direction_features_list = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                dir_nbd_batch = self.direction_gnn(node_obs_batch, adj_batch, agent_id_batch)
                dir_feat_batch = torch.cat([obs_batch, dir_nbd_batch], dim=1)
                dir_feat_batch = self.direction_mlp(dir_feat_batch)
                direction_features_list.append(dir_feat_batch)
            direction_features = torch.cat(direction_features_list, dim=0)
        else:
            direction_nbd_features = self.direction_gnn(node_obs, adj, agent_id)
            direction_features = torch.cat([obs, direction_nbd_features], dim=1)
            direction_features = self.direction_mlp(direction_features)

        # Optional RNN for temporal dependencies
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            direction_features, rnn_states = self.rnn(direction_features, rnn_states, masks)

        direction, action_log_probs = self.act(
            direction_features, available_actions, deterministic)

        direction = torch.tanh(direction)

        # ==================== Combine: |M_t| * D_t ====================
        u_mad = magnitude * direction
       # u_mad = direction

        # ==================== Base Controller ====================

        # Observation structure: [vel_x, vel_y, pos_x, pos_y, rel_goal_x, rel_goal_y, ...]
        # NOTE: goal position is already RELATIVE to current position in the observation
        # rel_goal = goal - current_pos (computed in navigation_graph.py:407)
        rel_goal = obs[:, 4:6]  # [rel_goal_x, rel_goal_y]

        # Proportional controller: u_base = K_p * rel_goal
        # Since rel_goal is already the error (goal - current), we use it directly
        u_base = self.K_p * rel_goal  # Broadcasting K_

        # Total action: base controller + MAD policy
        actions = u_base + u_mad

        return actions, action_log_probs, rnn_states, ssm_states

    def evaluate_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        ssm_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Evaluate log probability and entropy of given actions.

        Reconstructs the magnitude and direction terms to compute proper log probabilities
        for PPO training. Accounts for the base controller and MAD transformation.

        Args:
            obs: Agent observations [batch, obs_dim]
            node_obs: Neighbor observations [batch, num_nodes, node_dim]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            agent_id: Agent IDs [batch]
            rnn_states: RNN hidden states [batch, hidden_dim]
            ssm_states: SSM hidden states from buffer [batch, state_features] (complex)
            action: Actions to evaluate [batch, action_dim] (includes base controller)
            masks: Episode masks [batch, 1]
            available_actions: Unused for continuous actions
            active_masks: Active agent masks [batch, 1]

        Returns:
            action_log_probs: Log probabilities of actions [batch, 1]
            dist_entropy: Entropy of direction distribution (scalar)
        """
        # Convert inputs
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        batch_size = obs.shape[0]

        # ==================== Recompute Direction Term ====================
        if self.split_batch and (obs.shape[0] > self.max_batch_size):
            batchGenerator = minibatchGenerator(obs, node_obs, adj, agent_id, self.max_batch_size)
            direction_features_list = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                dir_nbd_batch = self.direction_gnn(node_obs_batch, adj_batch, agent_id_batch)
                dir_feat_batch = torch.cat([obs_batch, dir_nbd_batch], dim=1)
                dir_feat_batch = self.direction_mlp(dir_feat_batch)
                direction_features_list.append(dir_feat_batch)
            direction_features = torch.cat(direction_features_list, dim=0)
        else:
            direction_nbd_features = self.direction_gnn(node_obs, adj, agent_id)
            direction_features = torch.cat([obs, direction_nbd_features], dim=1)
            direction_features = self.direction_mlp(direction_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            direction_features, rnn_states = self.rnn(direction_features, rnn_states, masks)

        # Get direction distribution from ACTLayer
        #direction_dist = self.act.action_out(direction_features)

        # Compute entropy
        # dist_entropy = direction_dist.entropy()
        # if active_masks is not None:
        #     dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        # else:
        #     dist_entropy = dist_entropy.mean()

        # ==================== Recompute Magnitude Term ====================
        # Use stored SSM hidden states from buffer
        if ssm_states is not None:
            ssm_states = check(ssm_states).to(**self.tpdv)
        else:
            # Fallback: initialize to zeros (shouldn't happen during training)
            ssm_states = torch.complex(
                torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
                torch.zeros(batch_size, self.ssm.LRUR.state_features, device=obs.device),
            )

        # Detect first steps
        reset_mask = (masks.squeeze(-1) == 0)

        # Extract neighboring states from node observations (same as forward method)
        states_obs = node_obs[:, :, :self.system_state_dim]

        # Prepare disturbance observations: w_t = neighboring states at t=0 (reset), else 0
        w_t_states = torch.where(
            reset_mask.unsqueeze(-1).unsqueeze(-1).expand_as(states_obs),
            states_obs,  # w_0 = states at reset
            torch.zeros_like(states_obs)  # w_t = 0 otherwise
        )

        # Recompute magnitude with gradients enabled
        # NOTE: ZeroPreservingGNN ignores edge attributes to maintain f(0)=0 property
        if self.split_batch and (obs.shape[0] > self.max_batch_size):
            batchGenerator = minibatchGenerator(obs, w_t_states, adj, agent_id, self.max_batch_size)
            magnitude_features_list = []
            for batch in batchGenerator:
                obs_batch, w_t_batch, adj_batch, agent_id_batch = batch
                mag_nbd_batch = self.magnitude_gnn(w_t_batch, adj_batch, agent_id_batch)
                mag_feat_batch = self.magnitude_mlp(mag_nbd_batch)
                magnitude_features_list.append(mag_feat_batch)
            magnitude_features = torch.cat(magnitude_features_list, dim=0)
        else:
            magnitude_features = self.magnitude_gnn(w_t_states, adj, agent_id)
            magnitude_features = self.magnitude_mlp(magnitude_features)

        magnitude, ssm_states = self.ssm.step(magnitude_features, ssm_states)
        magnitude = torch.abs(magnitude)

        # ==================== Compute Base Controller ====================
        # Observation structure: [vel_x, vel_y, pos_x, pos_y, rel_goal_x, rel_goal_y, ...]
        # NOTE: goal position is already RELATIVE to current position
        rel_goal = obs[:, 4:6]  # [rel_goal_x, rel_goal_y] = goal - current_pos
        u_base = self.K_p * rel_goal

        # ==================== Invert MAD Transformation ====================
        # action = u_base + magnitude * tanh(direction_sample)
        # Solve for direction_sample

        # Remove base controller to get MAD component
        u_mad = action - u_base

        # Get direction after tanh
        direction_inferred = u_mad / (magnitude + 1e-8)
        direction_inferred = torch.clamp(direction_inferred, -0.999, 0.999)

        # Inverse tanh: arctanh(y) = 0.5 * log((1+y)/(1-y))
        direction_sample_inferred = 0.5 * torch.log(
            (1 + direction_inferred + 1e-8) / (1 - direction_inferred + 1e-8)
        )

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            direction_features,
            direction_sample_inferred,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        # Compute log probability of the inferred direction sample
        # direction_log_prob = direction_dist.log_probs(direction_sample_inferred)

        # # Tanh correction: log|d(tanh(x))/dx| = log(1 - tanh^2(x))
        # tanh_correction = torch.log(1 - direction_inferred**2 + 1e-8).sum(-1, keepdim=True)

        # # Full Jacobian correction: includes magnitude scaling + tanh transformation
        # # log|det(J)| = n*log(magnitude) + sum_i log(1 - tanh^2(x_i))
        # magnitude_correction = self.action_dim * torch.log(magnitude + 1e-8)
        # action_log_probs = direction_log_prob - magnitude_correction - tanh_correction

        return action_log_probs, dist_entropy


class MAD_GR_Critic(nn.Module):
    """
    Critic network for MAD policy.
    Reuses the standard Graph Critic from the existing implementation.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(MAD_GR_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[1]
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]

        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.critic_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim

        if args.critic_graph_aggr == "node":
            gnn_out_dim *= args.num_agents

        mlp_base_in_dim = gnn_out_dim
        if self.args.use_cent_obs:
            mlp_base_in_dim += cent_obs_shape[0]

        self.base = MLPBase(args, cent_obs_space, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(
        self, cent_obs, node_obs, adj, agent_id, rnn_states, masks
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute value function predictions.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.split_batch and (cent_obs.shape[0] > self.max_batch_size):
            batchGenerator = minibatchGenerator(
                cent_obs, node_obs, adj, agent_id, self.max_batch_size
            )
            critic_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                critic_feats_batch = self.base(act_feats_batch)
                critic_features.append(critic_feats_batch)
            critic_features = torch.cat(critic_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            if self.args.use_cent_obs:
                critic_features = torch.cat([cent_obs, nbd_features], dim=1)
            else:
                critic_features = nbd_features
            critic_features = self.base(critic_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states
