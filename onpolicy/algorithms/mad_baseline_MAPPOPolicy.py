import argparse
import gymnasium as gym
import torch
from torch import Tensor
from typing import Tuple

from onpolicy.algorithms.mad_actor_critic import MAD_Critic
from onpolicy.algorithms.mad_baseline_actor_critic import MAD_Baseline_Actor
from onpolicy.utils.util import update_linear_schedule
from loguru import logger


class MAD_Baseline_MAPPOPolicy:
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

        self.actor = MAD_Baseline_Actor(
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
            self.actor.parameters(), lr=self.lr,
            eps=self.opti_eps, weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr,
            eps=self.opti_eps, weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode: int, episodes: int) -> None:
        update_linear_schedule(
            param_groups=[self.actor_optimizer.param_groups[0]],
            epoch=episode, total_num_epochs=episodes, initial_lr=self.lr,
        )
        update_linear_schedule(
            param_groups=[self.critic_optimizer.param_groups],
            epoch=episode, total_num_epochs=episodes, initial_lr=self.critic_lr,
        )

    def get_actions(
        self,
        cent_obs, obs, node_obs, adj, agent_id, share_agent_id,
        rnn_states_actor, rnn_states_critic,
        ssm_states, disturbances, masks,
        available_actions=None, deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        actions, action_log_probs, rnn_states_actor, ssm_states, pre_tanh_value = self.actor.forward(
            obs, node_obs, adj, agent_id,
            rnn_states_actor, ssm_states, disturbances, masks,
            available_actions, deterministic,
        )
        values, rnn_states_critic = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, ssm_states, pre_tanh_value

    def get_values(
        self, cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
    ) -> Tensor:
        values, _ = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )
        return values

    def evaluate_actions(
        self,
        cent_obs, obs, node_obs, adj, agent_id, share_agent_id,
        rnn_states_actor, rnn_states_critic,
        disturbances, action, masks,
        available_actions=None, active_masks=None,
        lru_hidden_states=None, pre_tanh_value=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, node_obs, adj, agent_id,
            rnn_states_actor, lru_hidden_states, disturbances,
            action, masks, available_actions, active_masks,
        )
        values, _ = self.critic.forward(
            cent_obs, node_obs, adj, share_agent_id, rnn_states_critic, masks
        )
        return values, action_log_probs, dist_entropy

    def act(
        self,
        obs, node_obs, adj, agent_id,
        rnn_states_actor, ssm_states, disturbances, masks,
        available_actions=None, deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        actions, _, rnn_states_actor, ssm_states, _ = self.actor.forward(
            obs, node_obs, adj, agent_id,
            rnn_states_actor, ssm_states, disturbances, masks,
            available_actions, deterministic,
        )
        return actions, rnn_states_actor, ssm_states
