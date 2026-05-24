# Scaling up Stability
This is the code repository for the paper ***Distributed Control of Network Systems in the Space of Stabilizing Graph Neural Network Policies***.

Authors: John Cao, Luca Furieri

Acknowledgement: This code is built on top of the [InforMARL](https://github.com/nsidn98/InforMARL) code base.


## Installation

After setting up an environment, install packages with:
```
pip install -r requirements.txt
```

## Training
Run the following command to train:
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
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 128 \
--<use_stabilizing_policy, use_centrlized_actor, use_centralized_critic> \      # Omit this flag if training InforMARL
--discrete_action "False" \
--use_disturbance
```

## Testing
Plot stability comparison:
```
python -m plot_scripts.compare_stability \
--max_time_steps 100 \
--use_disturbance

python -m plot_scripts.compare_stability \
--max_time_steps 100 \
--use_trained_policy \
--model_path <trained-stable-model-path> \
--model_path_informarl <trained-informarl-path> \
--use_disturbance
```

Compute average rewards:
```
python -m plot_scripts.benchmark_table \
--ours_model_path <path-to-model> \
--informarl_model_path <path-to-model> \
--centralized_actor_model_path <path-to-model> \
--centralized_critic_model_path <path-to-model> \
--num_seeds 100
```



