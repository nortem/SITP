import gym
import pogema

import json
import random
import numpy as np
from os.path import join

import torch
from pogema import GridConfig
from pogema.animation import AnimationMonitor

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict

from evaluation.appo import APPOHolder
from evaluation.eval_utils import ResultsHolder
from training_run import validate_config, register_custom_components

from utils.config_validation import Environment
from wrappers.pogema_wrappers import MultiTimeLimit, LogPogemaStats
from wrappers import multi_curriculum, autocurriculum, autocurriculum_RG
from wrappers.pogema_wrappers import multipleConfigsLoader

def main():
    holder = APPOHolder(path='weights/weights/pbt')
    directory = 'svg20/1/'

    env_config = Environment()
    list_of_seeds = [ 79853]

    for seed in list_of_seeds:
        env_config.grid_config = GridConfig(seed=seed, size=20, num_agents=1, density=0.3)
        env = gym.make('Pogema-v0', config=env_config.grid_config)
        env_config.max_episode_steps = 512
        env = MultiTimeLimit(env, max_episode_steps=env_config.max_episode_steps)
        env = LogPogemaStats(env)
        env = AnimationMonitor(env, inverse_style=True, show_obs_radius=True, directory=directory+str(seed))
        observations = env.reset()
        done = [False, ...]
        infos = {}
        while not all(done):
            observations, rewards, done, infos = env.step(holder.act(observations))
        print(seed, infos[0]['episode_extra_stats']['CSR'])

def main2():
    # holder = APPOHolder(path='weights/weights/single_agent_random')
    holder = APPOHolder(path='weights/weights/baseline20')
    # holder = APPOHolder(path='weights/weights/pbt')

    directory = 'svg20_/'

    env_config = Environment()
    list_of_seeds = [ 79853]

    env_config.grid_config = GridConfig(size=20, num_agents=64, density=0.3)
    env_config.grid_config = GridConfig(size=8, num_agents=4, density=0.3)
    env = gym.make('Pogema-v0', config=env_config.grid_config)
    env_config.max_episode_steps = 256
    env = MultiTimeLimit(env, max_episode_steps=env_config.max_episode_steps)
    env = LogPogemaStats(env)
    env = autocurriculum_RG.AutoCurriculumWrapperReverseGoal(env)
    env = AnimationMonitor(env)
    # env = AnimationMonitor(env, inverse_style=True, show_obs_radius=True, directory=directory)

    for i in range(1):
        observations = env.reset()
        # print(env.active)
        done = [False, ...]
        infos = {}
        k = 0
        # print(np.array(observations).shape)
        while not all(done):
            # observations, rewards, done, infos = env.step([env.action_space.sample() for _ in range(env.config.num_agents)])
        # print(np.array(observations).shape)
            observations, rewards, done, infos = env.step(holder.act(observations))
        ISR_ = np.array([infos[i]['episode_extra_stats']['ISR'] for i in range(len(infos))])
        print(i, '=> CSR =>',  infos[0]['episode_extra_stats']['CSR'], '=> ISR =>', ISR_.mean())


def main3():
    map_grid_configs = multipleConfigsLoader('training_configs/train_maps')
    for env_config in map_grid_configs:
        # env_config = Environment()
        # env_config.grid_config = GridConfig(size=8, num_agents=8, density=0.05)
        env = gym.make('Pogema-v0', config=env_config.grid_config)
        env = AnimationMonitor(env)

        observations = env.reset()
        # for _ in range(20):
        #     observations, rewards, done, infos = env.step([env.action_space.sample() for _ in range(env.config.num_agents)])
        env.save_animation(f"svg/{env_config.grid_config.map_name[:-4]}.svg")
    print('DONE')
    # while not all(done):
        # observations, rewards, done, infos = env.step([env.action_space.sample() for _ in range(env.config.num_agents)])

if __name__ == '__main__':
    print(pogema.__version__)
    main3()