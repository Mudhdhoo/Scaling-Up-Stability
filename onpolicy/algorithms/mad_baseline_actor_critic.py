import argparse
from re import L
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.ssm import SSM
from onpolicy.algorithms.mad_actor_critic import MAD_Critic as MAD_Baseline_Critic  # noqa: F401
from loguru import logger


class MAD_Baseline_Actor(nn.Module):
    """
    MAD policy actor. Adapted from Furieri et al (2025): https://arxiv.org/abs/2504.02565

    Args:
        args: argparse namespace (shares MAD hyperparameters with mad_actor_critic).
        obs_space: per-agent observation space.
        node_obs_space: 2D node-feature space (num_nodes, num_node_feats).
        edge_obs_space: unused; kept for constructor-signature parity with the other actors.
        action_space: must be continuous (Box).
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
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.m_max = args.m_max_start
        self.K_p = args.kp_val
        self.under_training = True

        obs_shape = get_shape_from_obs_space(obs_space)                # (obs_dim,)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)      # (num_nodes, node_feat)
        self.obs_dim = obs_shape[0]
        self.node_flat_dim = node_obs_shape[0] * node_obs_shape[1]     # stacked local node feature dimension
        self.state_dim = self.obs_dim + self.node_flat_dim             # centralized state dim

        # Direction MLP over centralized state
        self.direction_mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
        )
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.ssm_state_features = getattr(args, "ssm_hidden_dim", 64)
        ssm_mlp_hidden = getattr(args, "ssm_mlp_hidden", 64)

        # M: disturbance-feedback operator
        self.ssm_M = SSM(
            input_size=self.node_flat_dim,
            output_size=1,
            lru_output_size=self.hidden_size,
            state_features=self.ssm_state_features,
            scan=False,
            mlp_hidden_size=ssm_mlp_hidden,
            rmin=args.rmin,
            rmax=args.rmax,
        )
        # A: initial-condition operator (fed x_0 at t=0, zeros afterwards)
        # self.ssm_A = SSM(
        #     input_size=self.state_dim,
        #     output_size=1,
        #     lru_output_size=self.hidden_size,
        #     state_features=self.ssm_state_features,
        #     scan=False,
        #     mlp_hidden_size=ssm_mlp_hidden,
        #     rmin=args.rmin,
        #     rmax=args.rmax,
        # )

        self.ssm_A = SSM(
            input_size=self.node_flat_dim,
            output_size=1,
            lru_output_size=self.hidden_size,
            state_features=self.ssm_state_features,
            scan=False,
            mlp_hidden_size=ssm_mlp_hidden,
            rmin=args.rmin,
            rmax=args.rmax,
        )


        self.to(device)

    def _zero_state(self, batch_size: int, device: torch.device) -> Tensor:
        zeros = torch.zeros(batch_size, self.ssm_state_features, device=device)
        return torch.complex(zeros, zeros)

    def _unpack_ssm_states(self, ssm_states, batch_size: int, device: torch.device):
        """Return (state_M, state_A) as complex tensors (batch, state_features)."""
        packed_size = 2 * self.ssm_state_features
        if ssm_states is None:
            return self._zero_state(batch_size, device), self._zero_state(batch_size, device)

        # Convert from numpy [*, 2] or real torch tensor [*, 2] to complex
        if isinstance(ssm_states, np.ndarray):
            ssm_states = check(ssm_states).to(**self.tpdv)
            ssm_states = torch.complex(ssm_states[..., 0], ssm_states[..., 1])
        elif not torch.is_complex(ssm_states):
            ssm_states = ssm_states.to(**self.tpdv)
            ssm_states = torch.complex(ssm_states[..., 0], ssm_states[..., 1])

        if ssm_states.shape[0] != batch_size or ssm_states.shape[-1] != packed_size:
            return self._zero_state(batch_size, device), self._zero_state(batch_size, device)

        state_M, state_A = torch.chunk(ssm_states, 2, dim=-1)
        return state_M.contiguous(), state_A.contiguous()

    @staticmethod
    def _apply_reset(state: Tensor, reset_mask: Tensor) -> Tensor:
        """Zero out rows of a complex state where reset_mask (bool, shape [batch]) is True."""

        if not reset_mask.any():
            return state

        keep = (~reset_mask).to(state.real.dtype).unsqueeze(-1)  # (batch, 1) real

        return torch.complex(state.real * keep, state.imag * keep)


    def _magnitude_and_features(
        self,
        obs: Tensor,
        node_obs: Tensor,
        disturbances: Tensor,
        masks: Tensor,
        ssm_states,
    ):
        """
        Shared core used by `forward` and `evaluate_actions`. Computes:
            - magnitude = |M(w) + A(x_0)|
            - features  = direction_mlp(centralized_state)
            - updated (state_M, state_A) ready to be re-packed
        """
        batch_size = obs.shape[0]
        state = torch.cat([obs, node_obs.reshape(batch_size, -1)], dim=1)      # (B, state_dim)
        w_flat = disturbances.reshape(batch_size, -1)                          # (B, node_flat_dim)

        # x_0 for the A LRU, set input to zero for t > 0
        reset_mask = (masks.squeeze(-1) == 0)
       # ssm_a_input = state * reset_mask.float().unsqueeze(-1)
        init_input = node_obs.reshape(batch_size, -1) * reset_mask.float().unsqueeze(-1)

        # Reset hidden states on episode boundaries
        state_M, state_A = self._unpack_ssm_states(ssm_states, batch_size, obs.device)

        state_M = self._apply_reset(state_M, reset_mask)
        state_A = self._apply_reset(state_A, reset_mask)

        out_M, state_M, _ = self.ssm_M.step(w_flat, state_M)
       # out_A, state_A, _ = self.ssm_A.step(ssm_a_input, state_A)
        out_A, state_A, _ = self.ssm_A.step(init_input, state_A)

        magnitude = torch.abs(out_M + out_A).clamp(min=1e-6, max=self.m_max)   # (B, 1)
        features = self.direction_mlp(state)                                   # (B, hidden)

        return magnitude, features, state_M, state_A

    def forward(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        ssm_states,
        disturbances,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        disturbances = check(disturbances).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        magnitude, features, state_M, state_A = self._magnitude_and_features(
            obs, node_obs, disturbances, masks, ssm_states
        )

        # Pre-stabilizing proportional base controller: u_base = K_p * (goal - pos)
        # Observation layout: [vel_x, vel_y, pos_x, pos_y, rel_goal_x, rel_goal_y, ...]
        rel_goal = obs[:, 4:6]
        u_base = self.K_p * rel_goal

        y, y_log_probs = self.act(features, available_actions, deterministic)
        D = torch.tanh(y)

        actions = u_base + magnitude * D 

        # log probability computation
        action_dim = D.shape[-1]
        log_jac_tanh = torch.log(1.0 - D.pow(2) + 1e-8).sum(-1, keepdim=True)
        log_jac_M = torch.log(magnitude + 1e-8).sum(-1, keepdim=True) * action_dim
        action_log_probs = y_log_probs - log_jac_M - log_jac_tanh

        ssm_states_out = torch.cat([state_M, state_A], dim=-1)

        if self.under_training:
            return (actions, action_log_probs, rnn_states, ssm_states_out, y)

        return (actions, action_log_probs, rnn_states, ssm_states_out, y, magnitude)

    def evaluate_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        ssm_states,
        disturbances,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ) -> Tuple[Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        disturbances = check(disturbances).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        magnitude, features, _, _ = self._magnitude_and_features(
            obs, node_obs, disturbances, masks, ssm_states
        )

        # Reconstruct y from u = u_base + |M+A| * tanh(y)
        rel_goal = obs[:, 4:6]
        u_base = self.K_p * rel_goal
        D = torch.clamp((action - u_base) / (magnitude + 1e-8), -0.999, 0.999)
        y = torch.atanh(D)

        y_log_probs, dist_entropy_y = self.act.evaluate_actions(
            features,
            y,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        action_dim = D.shape[-1]
        log_jac_tanh = torch.log(1.0 - D.pow(2) + 1e-8).sum(-1, keepdim=True)
        log_jac_M = torch.log(magnitude + 1e-8).sum(-1, keepdim=True) * action_dim
        action_log_probs = y_log_probs - log_jac_M - log_jac_tanh
        dist_entropy = dist_entropy_y + (log_jac_M + log_jac_tanh).mean()

        return action_log_probs, dist_entropy

