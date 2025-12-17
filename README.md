# Scaling up Stability
This is the code repository for the paper ***Scaling up Stability: Reinforcement Learning for Distributed Control of Networked Systems in the Space of Stabilizing Policies***.

Authors: John Cao, Luca Furieri

Acknowledgement: This code is built on top of the [InforMARL](https://github.com/nsidn98/InforMARL) code base.


## Installation

Setup environment and install packages:
```
conda env create -f environment.yml
conda activate stable_gnn_policy
```

## Training
Run the following command to train the policy:
```
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "stable_gnn" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 0 \
--experiment_name "stable_gnn_train" \
--scenario_name "navigation_graph" \
--num_agents 5 \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 128 \
--use_mad_policy \
--discrete_action "False" \
--use_disturbance
```

Run the following to train the InforMARL baseline:

```
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "informarl" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 0 \
--experiment_name "informarl" \
--scenario_name "navigation_graph" \
--num_agents 5 \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 128 \
--discrete_action "False" \
--use_disturbance
```

## Testing
Plot the testing results using these commands:
```
python -m plots.compare_stability --max_time_steps 100 --use_trained_policy False --use_disturbance # Compare stability without training

python -m plots.compare_stability --max_time_steps 100 False --use_disturbance # Compare stability after training

python compare_policies_across_agents.py # Compare average reward across different number of agents
```



