"""
Compute average rewards over N random seeds. Write the results to a txt file.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.append(os.path.abspath(os.getcwd()))

from onpolicy.config import get_config
from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from onpolicy.algorithms.mad_MAPPOPolicy import MAD_MAPPOPolicy
from onpolicy.algorithms.mad_baseline_MAPPOPolicy import MAD_Baseline_MAPPOPolicy
from onpolicy.algorithms.mad_mlp_critic_MAPPOPolicy import MAD_MLP_Critic_MAPPOPolicy
from multiagent.MPE_env import GraphMPEEnv
from onpolicy.envs.env_wrappers import GraphDummyVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_model_path", type=str,
                        default="./eval_models/stable_gnn/models/actor.pt")
    parser.add_argument("--informarl_model_path", type=str,
                        default="./eval_models/informarl/models/actor.pt")
    parser.add_argument("--mad_model_path", type=str,
                        default="./eval_models/mad/models/actor.pt")
    parser.add_argument("--mad_mlp_critic_model_path", type=str,
                        default="./eval_models/mad_mlp_critic/models/actor.pt")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of random seeds per (policy, N) cell")
    parser.add_argument("--agent_counts", type=int, nargs="+", default=[5, 7, 10])
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument("--num_obstacles", type=int, default=3)
    parser.add_argument("--scenario_name", type=str, default="navigation_graph")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=str, default="./results/benchmark_table.txt")
    parser.add_argument("--collaborative", type=bool, default=True)

    return parser.parse_args()


def load_config(model_path):
    """Load config.yaml from the run directory of a model checkpoint."""
    config_path = Path(model_path).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Warning: config.yaml not found at {config_path}, using defaults")
        return {}
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def make_args(num_agents, num_obstacles, episode_length, scenario_name, seed, base_config, collaborative):
    args = get_config().parse_args([])

    for k, v in base_config.items():
        setattr(args, k, v)

    # Override eval-time values
    args.env_name = "GraphMPE"
    args.scenario_name = scenario_name
    args.num_agents = num_agents
    args.num_landmarks = num_agents
    args.num_obstacles = num_obstacles
    args.episode_length = episode_length
    args.seed = seed
    args.n_rollout_threads = 1

    # Consistent eval conditions
    args.use_disturbance = True
    args.discrete_action = False
    args.collaborative = collaborative
    return args


def make_env(args):
    def init_env():
        env = GraphMPEEnv(args)
        env.seed(args.seed)
        return env
    return GraphDummyVecEnv([init_env])


def evaluate(policy, args, num_seeds, episode_length, policy_kind):
    """
    policy_kind: "informarl" | "mad" | "mad_baseline" | "mad_mlp_critic"
    Returns mean reward across `num_seeds` random environment seeds.
    """
    is_mad_like = policy_kind in ("mad", "mad_baseline", "mad_mlp_critic")
    num_agents = args.num_agents

    policy.actor.eval()
    policy.actor.under_training = False

    episode_rewards = []
    for s in range(num_seeds):
        # Re-seed the env for each test "seed" so initializations differ
        args.seed = s
        envs = make_env(args)
        obs, agent_ids, node_obs, adj, disturbances = envs.reset()

        rnn_states = np.zeros((num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks = np.zeros((num_agents, 1), dtype=np.float32)
        ssm_states = [None] * num_agents if is_mad_like else None

        ep_reward = 0.0
        for _ in range(episode_length):
            actions_list = []
            for i in range(num_agents):
                with torch.no_grad():
                    if is_mad_like:
                        result = policy.actor.forward(
                            obs=obs[:, i, :],
                            node_obs=node_obs[:, i, :, :],
                            adj=adj[:, i, :, :],
                            agent_id=agent_ids[:, i],
                            rnn_states=rnn_states[i:i+1],
                            ssm_states=ssm_states[i],
                            disturbances=disturbances[:, i, :],
                            masks=masks[i:i+1],
                            deterministic=True,
                        )
                        action, rnn_out, ssm_out = result[0], result[2], result[3]
                        ssm_states[i] = ssm_out
                    else:
                        action, _, rnn_out = policy.actor.forward(
                            obs=obs[:, i, :],
                            node_obs=node_obs[:, i, :, :],
                            adj=adj[:, i, :, :],
                            agent_id=agent_ids[:, i],
                            rnn_states=rnn_states[i:i+1],
                            masks=masks[i:i+1],
                            deterministic=True,
                        )
                rnn_states[i:i+1] = rnn_out.cpu().numpy()
                actions_list.append(action[0].detach().cpu().numpy())

            actions = np.array(actions_list)[np.newaxis, :, :]
            obs, agent_ids, node_obs, adj, disturbances, rewards, _, _ = envs.step(actions)
            ep_reward += rewards[0].mean()
            masks = np.ones((num_agents, 1), dtype=np.float32)

        envs.close()
        episode_rewards.append(ep_reward)

    return float(np.mean(episode_rewards))


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_policy(kind, args, model_path, device):
    """Build a policy of the given kind, then load weights from `model_path`."""
    env = make_env(args)
    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    node_obs_space = env.node_observation_space[0]
    edge_obs_space = env.edge_observation_space[0]
    act_space = env.action_space[0]
    env.close()

    if kind == "informarl":
        policy = GR_MAPPOPolicy(args, obs_space, share_obs_space, node_obs_space, edge_obs_space, act_space, device=device)
    elif kind == "mad":
        policy = MAD_MAPPOPolicy(args, obs_space, share_obs_space, node_obs_space, edge_obs_space, act_space, device=device)
    elif kind == "mad_baseline":
        policy = MAD_Baseline_MAPPOPolicy(args, obs_space, share_obs_space, node_obs_space, edge_obs_space, act_space, device=device)
    elif kind == "mad_mlp_critic":
        policy = MAD_MLP_Critic_MAPPOPolicy(args, obs_space, share_obs_space, node_obs_space, edge_obs_space, act_space, device=device)
    else:
        raise ValueError(f"Unknown policy kind: {kind}")

    policy.actor.load_state_dict(torch.load(model_path, map_location=device))
    return policy


def main():
    args = parse_args()
    device = torch.device(args.device)

    policies = [
        ("Ours",      "mad",            args.ours_model_path),
        ("InforMARL", "informarl",      args.informarl_model_path),
        ("MAD",       "mad_baseline",   args.mad_model_path),
        ("MLP-Critic","mad_mlp_critic", args.mad_mlp_critic_model_path),
    ]

    # Pre-load each policy's training config
    configs = {label: load_config(path) for label, _, path in policies}

    # results[label][N] = mean reward
    results = {label: {} for label, _, _ in policies}

    for N in args.agent_counts:
        print(f"\n=== N = {N} agents ===")
        for label, kind, path in policies:
            # MAD baseline has a fixed centralized input dimension and does not scale
            # across agent counts — only evaluate it at its training size.
            if kind == "mad_baseline" and N != configs[label].get("num_agents"):
                results[label][N] = None
                print(f"  {label:<10s}: N/A (fixed-size centralized policy)")
                continue

            env_args = make_args(
                num_agents=N,
                num_obstacles=args.num_obstacles,
                episode_length=args.episode_length,
                scenario_name=args.scenario_name,
                seed=0,
                base_config=configs[label],
                collaborative=args.collaborative,
            )
            # Make sure the policy flag in args matches the kind we are loading
            env_args.use_stabilizing_policy = (kind in ("mad", "mad_baseline", "mad_mlp_critic"))

            policy = build_policy(kind, env_args, path, device)
            actor_params = count_parameters(policy.actor)
            mean_reward = evaluate(policy, env_args, args.num_seeds, args.episode_length, kind)
            results[label][N] = mean_reward
            print(f"  {label:<10s}: {mean_reward:.3f}")

    # Write a simple txt table
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    labels = [lbl for lbl, _, _ in policies]
    col_w = 14
    with open(args.output, "w") as f:
        f.write(f"Mean reward over {args.num_seeds} random test seeds, episode_length={args.episode_length}\n")
        header = f"{'N':<6}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for N in args.agent_counts:
            row = f"{N:<6}"
            for lbl in labels:
                v = results[lbl][N]
                row += f"{'N/A':>{col_w}}" if v is None else f"{v:>{col_w}.3f}"
            f.write(row + "\n")

    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
