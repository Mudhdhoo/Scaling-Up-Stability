"""
Graph-based MAPPO Policy with SSM Magnitude Term and Base Controller.
Wraps actor and critic networks for multi-agent RL with GNN-based learning and SSM-based magnitude modulation.
"""
import gymnasium as gym
import argparse

import torch
from torch import Tensor
from typing import Tuple
from onpolicy.algorithms.graph_test_actor_critic import GR_Test_Actor, GR_Test_Critic
from onpolicy.utils.util import update_linear_schedule


class GraphTestPolicy:
    """
    Graph-based MAPPO Policy with SSM magnitude term and base controller.

    The policy uses the decomposition:
        u_t = u_base + |M_t(rel_goal_t=0)| * D_t(neighborhood_states)

    where:
        - u_base = K_p * rel_goal is a proportional base controller
        - M_t is an SSM-based magnitude term "kickstarted" with relative goal at t=0
        - D_t is a GNN-based stochastic direction term (normalized via tanh)

    Key Features:
        - Base controller provides task-relevant baseline behavior
        - SSM magnitude term is seeded with relative goal at episode start, then receives zeros
        - GNN direction term learns from neighborhood observations
        - Total action is base + magnitude * direction

    Args:
        args: Arguments containing relevant model and policy information
        obs_space: Observation space
        cent_obs_space: Centralized observation space (for critic)
        node_obs_space: Node observation space (graph features)
        edge_obs_space: Edge observation space
        act_space: Action space (must be continuous/Box)
        device: Device to run on (cpu/gpu)
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        act_space: gym.Space,
        device=torch.device("cpu"),
    ) -> None:
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.node_obs_space = node_obs_space
        self.edge_obs_space = edge_obs_space
        self.act_space = act_space
        self.split_batch = args.split_batch
        self.max_batch_size = args.max_batch_size

        self.actor = GR_Test_Actor(
            args,
            self.obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.act_space,
            self.device,
            self.split_batch,
            self.max_batch_size,
        )
        self.critic = GR_Test_Critic(
            args,
            self.share_obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.device,
            self.split_batch,
            self.max_batch_size,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode: int, episodes: int) -> None:
        """
        Decay the actor and critic learning rates.

        Args:
            episode: Current training episode
            episodes: Total number of training episodes
        """
        update_linear_schedule(
            optimizer=self.actor_optimizer,
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.lr,
        )
        update_linear_schedule(
            optimizer=self.critic_optimizer,
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.critic_lr,
        )

    def get_actions(
        self,
        cent_obs,
        obs,
        node_obs,
        adj,
        agent_id,
        share_agent_id,
        rnn_states_actor,
        rnn_states_critic,
        ssm_states,  # Added for compatibility with runner (not used by this policy)
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute actions and value function predictions for the given inputs.

        Args:
            cent_obs: Centralized input to the critic
            obs: Local agent inputs to the actor
            node_obs: Local agent graph node features
            adj: Adjacency matrix for the graph
            agent_id: Agent id for observations
            share_agent_id: Agent id for centralized observations
            rnn_states_actor: RNN states for actor
            rnn_states_critic: RNN states for critic
            ssm_states: SSM hidden states for magnitude term
            masks: Reset masks
            available_actions: Available actions (if None, all available)
            deterministic: Whether to use deterministic actions

        Returns:
            values: Value function predictions
            actions: Actions to take
            action_log_probs: Log probabilities of chosen actions
            rnn_states_actor: Updated actor RNN states
            rnn_states_critic: Updated critic RNN states
            pre_tanh_value: Raw Gaussian sample y (for correct policy gradient evaluation)
        """
        actions, action_log_probs, rnn_states_actor, pre_tanh_value = self.actor.forward(
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
        )

        values, rnn_states_critic = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, pre_tanh_value

    def get_values(
        self, cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
    ) -> Tensor:
        """
        Get value function predictions.

        Args:
            cent_obs: Centralized input to the critic
            node_obs: Local agent graph node features
            adj: Adjacency matrix
            share_agent_id: Agent id for centralized observations
            rnn_states_critic: RNN states for critic
            masks: Reset masks

        Returns:
            values: Value function predictions
        """
        values, _ = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )
        return values

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
        lru_hidden_states=None,  # Added for compatibility with runner (not used by this policy)
        pre_tanh_value=None,  # Raw Gaussian sample y for correct policy gradients
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get action logprobs / entropy and value function predictions for actor update.

        Args:
            cent_obs: Centralized input to the critic
            obs: Local agent inputs to the actor
            node_obs: Local agent graph node features
            adj: Adjacency matrix
            agent_id: Agent id for observations
            share_agent_id: Agent id for shared observations
            rnn_states_actor: RNN states for actor
            rnn_states_critic: RNN states for critic
            action: Actions to evaluate
            masks: Reset masks
            available_actions: Available actions
            active_masks: Active agent masks
            pre_tanh_value: Raw Gaussian sample y stored during rollout

        Returns:
            values: Value function predictions
            action_log_probs: Log probabilities of input actions
            dist_entropy: Action distribution entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks,
            pre_tanh_value,
        )

        values, _ = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )
        return values, action_log_probs, dist_entropy

    def act(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions using the given inputs.

        Args:
            obs: Local agent inputs to the actor
            node_obs: Local agent graph node features
            adj: Adjacency matrix
            agent_id: Agent id for nodes
            rnn_states_actor: RNN states for actor
            masks: Reset masks
            available_actions: Available actions
            deterministic: Whether to use deterministic actions

        Returns:
            actions: Actions to take
            rnn_states_actor: Updated actor RNN states
        """
        actions, _, rnn_states_actor = self.actor.forward(
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
        )
        return actions, rnn_states_actor
