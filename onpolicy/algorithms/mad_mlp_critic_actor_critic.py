import argparse
from typing import Tuple

import gymnasium as gym
import torch
from torch import Tensor
import torch.nn as nn

from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.util import check, init
from onpolicy.utils.util import get_shape_from_obs_space


class MAD_MLP_Critic(nn.Module):
    """
    Centralized MLP critic. Concatenates `cent_obs` with the flattened
    `node_obs` to form one global state vector and passes it through an MLP
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,  # unused; kept for signature parity
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)        # (cent_obs_dim,)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)        # (num_nodes, node_feats)
        self.cent_obs_dim = cent_obs_shape[0]
        self.node_flat_dim = node_obs_shape[0] * node_obs_shape[1]
        self.state_dim = self.cent_obs_dim + self.node_flat_dim

        # MLP over the centralized state.
        self.base = MLPBase(args, obs_shape=None, override_obs_dim=self.state_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

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
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        batch_size = cent_obs.shape[0]
        state = torch.cat([cent_obs, node_obs.reshape(batch_size, -1)], dim=1)

        critic_features = self.base(state)
        values = self.v_out(critic_features)

        return (values, rnn_states)
