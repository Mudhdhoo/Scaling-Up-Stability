"""
Test script for base P-Controller policy for navigation graph environment.
"""

from multiagent.environment import MultiAgentGraphEnv
from multiagent.policy import InteractivePolicy
from multiagent.custom_scenarios.navigation_graph import Scenario
from typing import Optional
import numpy as np

# makeshift argparser
class Args:
    def __init__(self):
        self.num_agents:int=3
        self.world_size=5
        self.num_scripted_agents=0
        self.num_obstacles:int=10
        self.collaborative:bool=False 
        self.max_speed:Optional[float]=2
        self.collision_rew:float=5
        self.goal_rew:float=5
        self.min_dist_thresh:float=0.1
        self.use_dones:bool=False
        self.episode_length:int=25
        self.max_edge_dist:float=1
        self.graph_feat_type:str='relative'

def random_discrete_policy(obs):
    policy = np.zeros(5)
    policy[np.random.choice(5)] = 1

    return policy

def random_continuous_policy(obs):
    policy = np.random.uniform(-1, 1, 2)

    return policy

def p_control_policy(obs, goal_pos):
    kp = 0.1
    # pos = obs[2:4]
    pos = obs[0:2]
    error = goal_pos - pos
    u = kp * error 
    u = np.clip(u, -1, 1)

    return u + np.random.randn(2) * 0

args = Args()

scenario = Scenario()
# create world
world = scenario.make_world(args)
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

# render call to create viewer window
env.render()
# create interactive policies for each agent
#policies = [InteractivePolicy(env,i) for i in range(env.n)]

policies = [p_control_policy for i in range(env.n)]

# execution loop
obs_n, agent_id_n, node_obs_n, adj_n = env.reset()

goal_positions = []
for i, agent in enumerate(env.world.agents):
    goal = env.world.landmarks[agent.id]
    goal_positions.append(goal.state.p_pos)

while True:
    # query for action from each agent's policy
    act_n = []

    for i, policy in enumerate(policies):
        action = policy(obs_n[i], goal_positions[i])
        act_n.append(action)
    # step environment
    obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)
    print(obs_n[0].shape, node_obs_n[0].shape, adj_n[0].shape, len(agent_id_n), len(reward_n), len(done_n), len(info_n))
    # render all agent views
    env.render()