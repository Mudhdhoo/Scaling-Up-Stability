from imp import new_module
from torch.nn.parameter import Parameter
from onpolicy.algorithms.graph_test_policy import GraphTestPolicy
from onpolicy.algorithms.graph_base_ssm_policy import GraphBaseSSMPolicy
from onpolicy.algorithms.mad_MAPPOPolicy import MAD_MAPPOPolicy
from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy 
from multiagent.custom_scenarios.navigation_graph import Scenario
from multiagent.environment import MultiAgentGraphEnv
import numpy as np
import torch
from typing import Optional
from loguru import logger
from onpolicy.config import graph_config, get_config
import os, sys
import time
from distutils.util import strtobool
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.getcwd()))

def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple_spread",
        help="Which scenario to run on",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=3, help="number of players")
    parser.add_argument(
        "--num_obstacles", type=int, default=3, help="Number of obstacles"
    )
    parser.add_argument(
        "--collaborative",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Number of agents in the env",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=2,
        help="Max speed for agents. NOTE that if this is None, "
        "then max_speed is 2 with discrete action space",
    )
    parser.add_argument(
        "--collision_rew",
        type=float,
        default=5,
        help="The reward to be negated for collisions with other "
        "agents and obstacles",
    )
    parser.add_argument(
        "--goal_rew",
        type=float,
        default=5,
        help="The reward to be added if agent reaches the goal",
    )
    parser.add_argument(
        "--min_dist_thresh",
        type=float,
        default=0.05,
        help="The minimum distance threshold to classify whether "
        "agent has reached the goal or not",
    )
    parser.add_argument(
        "--use_dones",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether we want to use the 'done=True' "
        "when agent has reached the goal or just return False like "
        "the `simple.py` or `simple_spread.py`",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args, parser

scenario = Scenario()

args = sys.argv[1:]

parser = get_config()
_, parser = parse_args(args, parser)

all_args, parser = graph_config(args, parser)

# create world
world = scenario.make_world(all_args)

# create multiagent environment
env = MultiAgentGraphEnv(world=world, reset_callback=scenario.reset_world, 
                    reward_callback=scenario.reward, 
                    observation_callback=scenario.observation, 
                    graph_observation_callback=scenario.graph_observation,
                    info_callback=scenario.info_callback, 
                    done_callback=scenario.done,
                    id_callback=scenario.get_id,
                    update_graph=scenario.update_graph,
                    shared_viewer=True,
                    discrete_action=False)


policy = MAD_MAPPOPolicy(all_args, env.observation_space[0], env.share_observation_space[0], env.node_observation_space[0], env.edge_observation_space[0], env.action_space[0])
#policy = GR_MAPPOPolicy(all_args, env.observation_space[0], env.share_observation_space[0], env.node_observation_space[0], env.edge_observation_space[0], env.action_space[0])

model_path = "/Users/johncao/Documents/Programming/Oxford/InforMARL/onpolicy/results/GraphMPE/navigation_graph/rmappo/mad_policy_relu/run2/models/actor.pt"
#model_path = "/Users/johncao/Documents/Programming/Oxford/InforMARL/onpolicy/results/GraphMPE/navigation_graph/rmappo/informarl/run42/models/actor.pt"

model_loaded = torch.load(model_path)

policy.actor.load_state_dict(model_loaded)

policy.actor.eval()
policy.actor.under_training = False

Lambda_mod = torch.exp(-torch.exp(policy.actor.ssm.LRUR.nu_log))
Lambda_re = Lambda_mod * torch.cos(torch.exp(policy.actor.ssm.LRUR.theta_log))
Lambda_im = Lambda_mod * torch.sin(torch.exp(policy.actor.ssm.LRUR.theta_log))
Lambda = torch.complex(Lambda_re, Lambda_im).abs()  # Eigenvalues matrix

# Initialize separate RNN states and masks for each agent
rnn_states = np.zeros((env.n, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
masks = np.zeros((env.n, 1), dtype=np.float32)  # 0 = first step (seeds LRU with x0)

# Initialize separate SSM states for each agent (MAD policy)
# Each agent's SSM is seeded with their own initial observation x_0
ssm_states = [None] * env.n  # Will be initialized in the first forward pass for each agent

# Frame rate control for smooth animation
target_fps = 30  # Adjust this value: higher = faster, lower = slower
frame_time = 1.0 / target_fps

num_collisions = 0

episodes = 1
epsiode_length = 200

magnitudes = []
for episode in range(episodes):
    obs_n, agent_id_n, node_obs_n, adj_n, disturbance_n = env.reset()
    env.render()
    for step in range(epsiode_length):
        frame_start = time.time()
        act_n = []
        mag_agents = []
        for i in range(env.n):
            # Each agent has its own RNN hidden state and SSM state
            with torch.no_grad():
                # Get full action
                action, _, rnn_states_out, ssm_states_out, y, magnitude = policy.actor.forward(
                    obs=obs_n[i][None, :],  # Add batch dim
                    node_obs=node_obs_n[i][None, :, :],  # Add batch dim
                    adj=adj_n[i][None, :, :],  # Add batch dim
                    agent_id=np.array([agent_id_n[i]]),
                    rnn_states=rnn_states[i:i+1],  # Use this agent's RNN state
                    ssm_states=ssm_states[i],  # Use this agent's SSM state
                    disturbances=disturbance_n[i][None, :],
                    masks=masks[i:i+1],  # Use this agent's mask
                    deterministic=True
                )

                # Update this agent's RNN and SSM states for next timestep
                rnn_states[i:i+1] = rnn_states_out
                ssm_states[i] = ssm_states_out

                mag_agents.append(magnitude[0,0].item())

            # For continuous action space, use actions directly
            action_array = action[0].detach().cpu().numpy()  # Shape: (action_dim,)

            act_n.append(action_array)

        magnitudes.append(mag_agents)

        # step environment
        obs_n, agent_id_n, node_obs_n, adj_n, disturbance_n, reward_n, done_n, info_n = env.step(act_n)

        new_num_collisions = 0
        for info in info_n:
            if info["Num_agent_collisions"] > 0 or info["Num_obst_collisions"] > 0:
                new_num_collisions += info["Num_agent_collisions"] + info["Num_obst_collisions"]

        if new_num_collisions > num_collisions:
            num_collisions = new_num_collisions
            logger.info(f"Collision detected: {num_collisions}")

        # Set all agent masks to 1 for next iteration (continuation)
        masks = np.ones((env.n, 1), dtype=np.float32)

        # render
        env.render()

        # Maintain consistent frame rate for smooth animation
        frame_elapsed = time.time() - frame_start
        if frame_elapsed < frame_time:
            time.sleep(frame_time - frame_elapsed)

    # Reset states for next episode
    rnn_states = np.zeros((env.n, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    masks = np.zeros((env.n, 1), dtype=np.float32)
    ssm_states = [None] * env.n
    num_collisions = 0

fig = plt.figure()
plt.plot(magnitudes)
plt.show()