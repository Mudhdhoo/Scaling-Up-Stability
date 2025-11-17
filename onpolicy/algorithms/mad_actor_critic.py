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
from onpolicy.algorithms.utils.gnn import GNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.lru import LRU
from onpolicy.algorithms.utils.distributions import FixedNormal
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from loguru import logger


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


class MAD_Actor(nn.Module):
    """
    MAD Actor network for stability-constrained RL.

    The policy is decomposed as:
        u_t = |M_t(x_0)| * D_t(neighborhood_states)

    where:
        - M_t is an LRU-based magnitude term seeded with initial condition x_0
        - D_t is a GNN-based stochastic direction term (Gaussian policy)
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
        super(MAD_Actor, self).__init__()
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

        obs_shape = get_shape_from_obs_space(obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[1]
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]

        # Get action dimension
        if action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0]
        else:
            raise NotImplementedError("MAD policy currently only supports Box action spaces")

        # Base controller parameters (P controller: u_base = K_p * (goal - current))
        if args.learnable_kp:
            # Learnable proportional gains (initialized to reasonable values)
            self.K_p = nn.Parameter(torch.ones(self.action_dim) * args.kp_val)
        else:
            self.K_p = args.kp_val


        # GNN for direction term - aggregates neighborhood information
        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim

        # MLP base to process concatenated features for direction
        mlp_base_in_dim = gnn_out_dim + obs_shape[0]
        self.base = MLPBase(args, obs_shape=None, override_obs_dim=mlp_base_in_dim)

        # Optional RNN for temporal dependencies in direction
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        # Direction term: outputs mean and log_std for Gaussian distribution
        # The direction should be normalized (|D_t| <= 1)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)

        self.direction_mean = init_(nn.Linear(self.hidden_size, self.action_dim))
        self.direction_logstd = nn.Parameter(torch.zeros(self.action_dim))

        # Magnitude term: LRU that takes initial state and outputs magnitude over time
        # LRU hidden dim can be a hyperparameter, using hidden_size for now
        lru_hidden_dim = getattr(args, 'lru_hidden_dim', 64)
        self.lru = LRU(
            input_dim=obs_shape[0],  # Takes full state observation
            hidden_dim=lru_hidden_dim,
            output_dim=self.action_dim,
            use_orthogonal=self._use_orthogonal,
            gain=self._gain
        )

        # Store initial observations for LRU
        # This will be set at the beginning of each episode
        self.x0 = None
        self.lru_hidden = None

        self.to(device)

    def set_initial_state(self, obs: torch.Tensor):
        """
        Set the initial state for the LRU magnitude term.
        Should be called at the beginning of each episode.

        Args:
            obs: Initial observation [batch_size, obs_dim] or [obs_dim]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        self.x0 = obs
        self.lru_hidden = self.lru.init_hidden(batch_size, device)

    def forward(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs using MAD decomposition.

        Args:
            obs: Observation inputs [batch, obs_dim]
            node_obs: Local agent graph node features [batch, num_nodes, node_dim]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            agent_id: Agent id for node indexing [batch]
            rnn_states: RNN hidden states [batch, hidden_dim]
            masks: Reset masks [batch, 1]
            available_actions: Available action mask (unused for Box)
            deterministic: Whether to use mean action (no sampling)

        Returns:
            actions: Actions to take [batch, action_dim]
            action_log_probs: Log probabilities [batch, 1]
            rnn_states: Updated RNN states [batch, hidden_dim]
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        batch_size = obs.shape[0]

        # Initialize LRU if needed (at episode start)
        # Check if masks indicate reset (mask == 0 means reset)
        # Each environment tracks its own episode independently
        reset_mask = (masks.squeeze(-1) == 0)  # [batch_size] boolean tensor
        reset_indices = reset_mask.nonzero(as_tuple=True)[0]

        if len(reset_indices) > 0 or self.x0 is None:
            if self.x0 is None or self.x0.shape[0] != batch_size:
                self.set_initial_state(obs)
            else:
                # Update only the reset episodes
                for idx in reset_indices:
                    self.x0[idx] = obs[idx]
                    # Properly initialize LRU hidden state for reset environments
                    self.lru_hidden[idx] = self.lru.init_hidden(1, obs.device).squeeze(0)

        # ==================== Direction Term ====================
        # Compute direction using GNN + MLP + (optional RNN) + Gaussian

        if self.split_batch and (obs.shape[0] > self.max_batch_size):
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Get direction distribution parameters
        direction_mean = self.direction_mean(actor_features)
        direction_std = torch.exp(self.direction_logstd)

        # Normalize direction to have |D| <= 1 using tanh
        # Sample from Gaussian then apply tanh
        direction_dist = FixedNormal(direction_mean, direction_std)

        if deterministic:
            direction = torch.tanh(direction_mean)
        else:
            direction_sample = direction_dist.sample()
            direction = torch.tanh(direction_sample)

        # ==================== Magnitude Term ====================
        # Get magnitude from LRU
        # Each environment independently tracks whether it's at the first step
        # Use masks: mask == 0 means this is the first step after reset
        is_first_step = reset_mask  # [batch_size] boolean tensor

        # For environments at their first step, use x0; otherwise use zeros
        v_t = torch.where(
            is_first_step.unsqueeze(-1).expand_as(self.x0),
            self.x0,
            torch.zeros_like(self.x0)
        )

        magnitude, self.lru_hidden = self.lru.step(v_t, self.lru_hidden, is_first_step)

        # Take absolute value to ensure magnitude is positive
        magnitude = torch.abs(magnitude)

        # ==================== Combine Magnitude and Direction ====================
        u_mad = magnitude * direction

        # ==================== Add Base Controller ====================
        # Extract goal from observation (assumes goal is part of obs)
        # For navigation: obs typically contains [pos, vel, goal_pos, ...]
        # Adjust indices based on your specific observation structure
        obs_dim = obs.shape[-1]

        # Observation structure: [vel_x, vel_y, pos_x, pos_y, goal_x, goal_y, ...]
        if obs_dim >= 6:
            current_pos = obs[:, 2:4]  # [pos_x, pos_y]
            goal_pos = obs[:, 4:6]     # [goal_x, goal_y]

            # Proportional controller: u_base = K_p * (goal - current)
            # This provides baseline stability
            error = goal_pos - current_pos
            u_base = self.K_p * error  # Broadcasting K_p

            # Total action: base controller + MAD policy
    
            actions = u_base + u_mad
        else:
            # If observation doesn't have goal, just use MAD policy
            actions = u_mad

        # Compute log probability
        # For MAD policies, we compute log prob of the direction before tanh
        # and adjust for the tanh transformation
        if deterministic:
            # For deterministic, we use the mean, so log_prob is technically undefined
            # We'll use a placeholder
            action_log_probs = torch.zeros(batch_size, 1, device=obs.device)
        else:
            # Log prob of sampled direction (before tanh)
            direction_log_prob = direction_dist.log_probs(direction_sample)

            # Adjust for tanh transformation: log|d/dx tanh(x)|
            # d/dx tanh(x) = 1 - tanh^2(x)
            tanh_correction = torch.log(1 - direction**2 + 1e-8).sum(-1, keepdim=True)
            action_log_probs = direction_log_prob - tanh_correction

        return actions, action_log_probs, rnn_states, self.lru_hidden

    def evaluate_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        lru_hidden_states=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.

        This method reconstructs the magnitude term and inverts the MAD transformation
        to compute proper log probabilities for PPO training.

        Args:
            obs: Observation inputs
            node_obs: Graph node features
            adj: Adjacency matrix
            agent_id: Agent IDs
            rnn_states: RNN hidden states
            action: Actions to evaluate [batch, action_dim]
            masks: Reset masks
            available_actions: Available actions (unused)
            active_masks: Active agent masks

        Returns:
            action_log_probs: Log probabilities of actions [batch, 1]
            dist_entropy: Entropy of action distribution (scalar)
        """
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

        # ==================== Recompute Base Controller ====================
        u_base = torch.zeros_like(action)
        obs_dim = obs.shape[-1]
        if obs_dim >= 6:
            current_pos = obs[:, 0:2]
            goal_pos = obs[:, 4:6]
            error = goal_pos - current_pos
            u_base = self.K_p * error

        # ==================== Direction Term ====================
        if self.split_batch and (obs.shape[0] > self.max_batch_size):
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Get direction distribution parameters
        direction_mean = self.direction_mean(actor_features)
        direction_std = torch.exp(self.direction_logstd)
        direction_dist = FixedNormal(direction_mean, direction_std)

        # Compute entropy (independent of magnitude)
        dist_entropy = direction_dist.entropy().sum(-1)
        if active_masks is not None:
            dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            dist_entropy = dist_entropy.mean()

        # ==================== Recompute Magnitude with Stored LRU States ====================
        # Now we properly recompute magnitude using stored LRU hidden states
        # This enables gradients to flow through the LRU!

        if lru_hidden_states is not None:
            # Use stored LRU hidden states from buffer
            lru_hidden = check(lru_hidden_states).to(**self.tpdv)
        else:
            # Fallback: initialize to zeros (shouldn't happen during training)
            lru_hidden = self.lru.init_hidden(batch_size, obs.device)

        # Detect first steps (mask == 0 means episode reset)
        reset_mask = (masks.squeeze(-1) == 0)

        # Prepare LRU input: x0 at first step, zeros otherwise
        v_t = torch.where(
            reset_mask.unsqueeze(-1).expand_as(obs),
            obs,
            torch.zeros_like(obs)
        )

        # Recompute magnitude with gradients enabled!
        magnitude, _ = self.lru.step(v_t, lru_hidden, reset_mask)
        magnitude = torch.abs(magnitude)

        # ==================== Invert MAD Transformation ====================
        # action = u_base + magnitude * tanh(direction_sample)
        # Solve for direction_sample given the stored action

        # Extract MAD component
        u_mad = action - u_base

        # Divide by magnitude to get direction after tanh
        direction_inferred = u_mad / (magnitude + 1e-8)
        direction_inferred = torch.clamp(direction_inferred, -0.999, 0.999)

        # Inverse tanh to recover direction_sample
        direction_sample_inferred = 0.5 * torch.log(
            (1 + direction_inferred + 1e-8) / (1 - direction_inferred + 1e-8)
        )

        # Compute log probability under Gaussian distribution
        direction_log_prob = direction_dist.log_probs(direction_sample_inferred)

        # Tanh correction: log|d/dx tanh(x)| = log(1 - tanh^2(x))
        tanh_correction = torch.log(1 - direction_inferred**2 + 1e-8).sum(-1, keepdim=True)

        action_log_probs = direction_log_prob - tanh_correction

        return action_log_probs, dist_entropy


class MAD_Critic(nn.Module):
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
        super(MAD_Critic, self).__init__()
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
