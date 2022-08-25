from wrappers import multi_curriculum, autocurriculum, autocurriculum_RG, autocurriculum_TSCL
import gym
import pogema

from functools import cmp_to_key
import json
import yaml
import random
from os.path import join

import torch
from pogema import GridConfig

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict

from evaluation.eval_utils import ResultsHolder
from training_run import validate_config, register_custom_components

from utils.config_validation import Environment
from wrappers.pogema_wrappers import MultiTimeLimit, LogPogemaStats, MultipleConfigsWrapper, multipleConfigsLoader
from pogema.animation import AnimationMonitor


class APPOHolder:
    def __init__(self, path):
        register_custom_components()

        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        exp, flat_config = validate_config(config['full_config'])
        cfg = flat_config

        env = create_env(cfg.env, cfg=cfg, env_config={})

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        env.close()

        # force cpu workers for parallel evaluation
        cfg.device = 'cuda'
        if cfg.device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            print('CUDA')
            device = torch.device('cuda')

        actor_critic.model_to_device(device)
        policy_id = cfg.policy_index
        checkpoints = join(path, f'checkpoint_p{policy_id}')
        checkpoints = LearnerWorker.get_checkpoints(checkpoints)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = cfg

        self.rnn_states = None

    def act(self, observations):
        if self.rnn_states is None or len(self.rnn_states) != len(observations):
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)

        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)
            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        return actions.cpu().numpy()

    def after_step(self, dones):
        for agent_i, done_flag in enumerate(dones):
            if done_flag:
                self.rnn_states[agent_i] = torch.zeros([get_hidden_size(self.cfg)], dtype=torch.float32,
                                                       device=self.device)


holder = APPOHolder(path='weights/weights/c164')
map_grid_configs = multipleConfigsLoader('training_configs/train_maps_3')
#########################
# num_agents = 2
# grid_config = GridConfig(size=20, num_agents=num_agents, density=0.3)
# env = gym.make('Pogema-v0', config=grid_config)
# env = MultiTimeLimit(env, max_episode_steps=256)
# env = LogPogemaStats(env)
# env = autocurriculum_RG.AutoCurriculumWrapperReverseGoal(env)
# env = multi_curriculum.AlwaysNAgents(env, 64)
# for i in range(100):
#     observations = env.reset()
#     done = [False, ...]
#     infos = {}
#     while not all(done):
#         observations, rewards, done, infos = env.step(holder.act(observations))
#         holder.after_step(done) 
# env.reset()

# #########################
# num_agents = 2
# grid_config = GridConfig(size=20, num_agents=num_agents, density=0.3)
# env = gym.make('Pogema-v0', config=grid_config)
# env = MultiTimeLimit(env, max_episode_steps=256)
# env = LogPogemaStats(env)
# env = autocurriculum.AutoCurriculumWrapper(env, map_grid_configs)
# env = multi_curriculum.AlwaysNAgents(env, 64)
# for i in range(100):
#     observations = env.reset()
#     done = [False, ...]
#     infos = {}
#     while not all(done):
#         observations, rewards, done, infos = env.step(holder.act(observations))
#         holder.after_step(done) 
# env.reset()

# #####################
# map_grid_configs = multipleConfigsLoader('training_configs/train_maps')

# num_agents = 2
# grid_config = GridConfig(size=20, num_agents=num_agents, density=0.3)
# env = gym.make('Pogema-v0', config=grid_config)
# env = MultiTimeLimit(env, max_episode_steps=100)
# env = LogPogemaStats(env)
# env = MultipleConfigsWrapper(env, 'training_configs/train_maps', map_grid_configs)

# for _ in range(10):
#     observations = env.reset()
#     done = [False, ...]
#     infos = {}
#     while not all(done):
#         observations, rewards, done, infos = env.step(holder.act(observations))
#         holder.after_step(done) 
# env.reset()

#######################
# _dict_map_name = dict()
# _vect_map_name = []
# i = 0
# for conf in map_grid_configs:
#     _dict_map_name[conf.grid_config.map_name] = i
#     _vect_map_name.append(conf.grid_config.map_name)
#     i += 1
# print(_dict_map_name)
# print(_vect_map_name)

########################
num_agents = 2
grid_config = GridConfig(size=8, num_agents=8, density=0.3)
env = gym.make('Pogema-v0', config=grid_config)
env = MultiTimeLimit(env, max_episode_steps=50)
env = LogPogemaStats(env)
env = autocurriculum.AutoCurriculumWrapper(env, map_grid_configs)
env = multi_curriculum.AlwaysNAgents(env, 64)
for i in range(100):
    observations = env.reset()
    done = [False, ...]
    infos = {}
    while not all(done):
        observations, rewards, done, infos = env.step(holder.act(observations))
        holder.after_step(done) 
    # print(f'done: {infos}')
env.reset()


print('all right')