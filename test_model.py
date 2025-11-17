from onpolicy.algorithms.mad_MAPPOPolicy import MAD_MAPPOPolicy as Policy
from onpolicy.algorithms.mad_actor_critic import MAD_Actor
from onpolicy.algorithms.graph_actor_critic import GR_Actor, GR_Critic
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

sys.path.append(os.path.abspath(os.getcwd()))

def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple_spread",
        help="Which scenario to run on",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=10, help="number of players")
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

#mad_actor = MAD_Actor(all_args, env.observation_space[0], env.node_observation_space[0], env.edge_observation_space[0], env.action_space[0])

actor = GR_Actor(all_args, env.observation_space[0], env.node_observation_space[0], env.edge_observation_space[0], env.action_space[0])

model_path = "/Users/johncao/Documents/Programming/Oxford/InforMARL/onpolicy/results/GraphMPE/navigation_graph/rmappo/informarl/run31/models/actor.pt"

actor.load_state_dict(torch.load(model_path))


# render call to create viewer window
env.render()

# execution loop
obs_n, agent_id_n, node_obs_n, adj_n = env.reset()

# Simple initialization - zeros for RNN and ZERO for masks (first step)
rnn_states = np.zeros((1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
masks = np.zeros((1, 1), dtype=np.float32)  # 0 = first step (seeds LRU with x0)

# Frame rate control for smooth animation
target_fps = 30  # Adjust this value: higher = faster, lower = slower
frame_time = 1.0 / target_fps

# Collision tracking
total_collisions = 0
agent_collisions = 0
obstacle_collisions = 0
# Track previous collision counts for each agent
prev_agent_collisions = np.zeros(env.n)
prev_obstacle_collisions = np.zeros(env.n)

step = 0
while True:
    frame_start = time.time()
    act_n = []

    for i in range(env.n):
        # Simple forward pass - reuse same RNN state and mask for all agents
        with torch.no_grad():
            action, _, _ = actor.forward(
                obs=obs_n[i][None, :],  # Add batch dim
                node_obs=node_obs_n[i][None, :, :],  # Add batch dim
                adj=adj_n[i][None, :, :],  # Add batch dim
                agent_id=np.array([agent_id_n[i]]),
                rnn_states=rnn_states,
                masks=masks,
                deterministic=True
            )

        # For continuous action space, use actions directly
        action_array = action[0].detach().cpu().numpy()  # Shape: (action_dim,)
        act_n.append(action_array)

    # After first step, set masks to 1 (continuation)
    if step == 0:
        masks = np.ones((1, 1), dtype=np.float32)

    # step environment
    obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)
    step += 1

    #print(obs_n[0].shape, node_obs_n[0].shape, adj_n[0].shape, len(agent_id_n), len(reward_n), len(done_n), len(info_n))

    # Check for collisions by comparing with previous counts
    step_collisions = 0
    for i, info in enumerate(info_n):
        # Check if agent collisions increased
        curr_agent_coll = info.get('Num_agent_collisions', 0)
        if curr_agent_coll > prev_agent_collisions[i]:
            new_agent_colls = int(curr_agent_coll - prev_agent_collisions[i])
            agent_collisions += new_agent_colls
            step_collisions += new_agent_colls
            print(f"[Step {step}] Agent {i} collided with another agent!")
            prev_agent_collisions[i] = curr_agent_coll

        # Check if obstacle collisions increased
        curr_obst_coll = info.get('Num_obst_collisions', 0)
        if curr_obst_coll > prev_obstacle_collisions[i]:
            new_obst_colls = int(curr_obst_coll - prev_obstacle_collisions[i])
            obstacle_collisions += new_obst_colls
            step_collisions += new_obst_colls
            print(f"[Step {step}] Agent {i} collided with an obstacle!")
            prev_obstacle_collisions[i] = curr_obst_coll

    if step_collisions > 0:
        total_collisions += step_collisions
        print(f"[Step {step}] Total collisions this step: {step_collisions}")
        print(f"[Cumulative] Agent-Agent: {agent_collisions}, Agent-Obstacle: {obstacle_collisions}, Total: {total_collisions}\n")

    # render
    env.render()

    # Maintain consistent frame rate for smooth animation
    frame_elapsed = time.time() - frame_start
    if frame_elapsed < frame_time:
        time.sleep(frame_time - frame_elapsed)