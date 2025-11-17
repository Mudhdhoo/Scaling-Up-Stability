"""
MAD MAPPO Policy class.
Wraps MAD Actor and Critic networks for stability-constrained RL.
"""
import gymnasium as gym
import argparse

import torch
from torch import Tensor
from typing import Tuple
from onpolicy.algorithms.mad_actor_critic import MAD_Actor, MAD_Critic
from onpolicy.utils.util import update_linear_schedule


class MAD_MAPPOPolicy:
    """
    MAD MAPPO Policy class. Wraps MAD actor and critic networks
    to compute actions and value function predictions with stability guarantees.

    The policy uses MAD decomposition:
        u_t = |M_t(x_0)| * D_t(neighborhood_states)

    where:
        - M_t is an LRU-based magnitude term ensuring stability
        - D_t is a GNN-based stochastic direction term

    Args:
        args: Arguments containing relevant model and policy information
        obs_space: Observation space
        cent_obs_space: Centralized observation space (for critic)
        node_obs_space: Node observation space
        edge_obs_space: Edge observation space
        act_space: Action space
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

        self.actor = MAD_Actor(
            args,
            self.obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.act_space,
            self.device,
            self.split_batch,
            self.max_batch_size,
        )
        self.critic = MAD_Critic(
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
            masks: Reset masks
            available_actions: Available actions (if None, all available)
            deterministic: Whether to use deterministic actions

        Returns:
            values: Value function predictions
            actions: Actions to take
            action_log_probs: Log probabilities of chosen actions
            rnn_states_actor: Updated actor RNN states
            rnn_states_critic: Updated critic RNN states
        """
        actions, action_log_probs, rnn_states_actor, lru_hidden_states = self.actor.forward(
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
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, lru_hidden_states

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
        lru_hidden_states=None,
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
            lru_hidden_states,
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
    ) -> Tuple[Tensor, Tensor]:
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

    def reset_lru(self, obs):
        """
        Reset the LRU hidden state with initial observations.
        Should be called at the beginning of each episode.

        Args:
            obs: Initial observations [batch_size, obs_dim] or [obs_dim]
        """
        self.actor.set_initial_state(obs)
