import json
from argparse import Namespace

import gym
import numpy as np
import yaml
from sample_factory.algorithms.utils.algo_utils import EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_PER_POLICY_SUMMARIES

from pogema.animation import AnimationMonitor

from sample_factory.utils.utils import log

import wandb
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm

import sys
from models.residual_net import ResnetEncoder
from utils.config_validation import Experiment, Environment, Stage
from wrappers.pogema_wrappers import MultiTimeLimit, LogPogemaStats, AutoResetWrapper, PogemaStackFramesWrapper, MultipleConfigsWrapper, multipleConfigsLoader
from wrappers.multi_curriculum import CurriculumWrapper, AlwaysNAgents
from wrappers.autocurriculum import AutoCurriculumWrapper
# from wrappers.autocurriculum_TSCL import AutoCurriculumWrapper
from wrappers.autocurriculum_RG import AutoCurriculumWrapperReverseGoal

def make_pogema(full_env_name, cfg=None, env_config=None):
    # noinspection Pydantic
    environment_config: Environment = Environment(**cfg.full_config['environment'])

    if env_config is None or env_config.get("remove_seed", True):
        environment_config.grid_config.seed = None
    env = gym.make(environment_config.name, config=environment_config.grid_config)
    if environment_config.framestack > 1:
        env = PogemaStackFramesWrapper(env, framestack=environment_config.framestack)

    if environment_config.max_episode_steps:
        env = MultiTimeLimit(env, max_episode_steps=environment_config.max_episode_steps)
    
    env = LogPogemaStats(env)
    
    if environment_config.path_to_grid_configs:
        print('wrapper random')
        map_grid_configs = multipleConfigsLoader(environment_config.path_to_grid_configs)
        env = MultipleConfigsWrapper(env, environment_config.path_to_grid_configs, map_grid_configs)

    # with open(r'find_seed/sorted_seeds_mean_1.yml', 'r') as file:
    #     list_seeds = yaml.safe_load(file)

    if environment_config.curriculum:
        print('wrapper curriculum stages')
        num_agents = env.config.num_agents
        # noinspection Pydantic
        stages = [Stage(**stage.dict()) for stage in environment_config.curriculum]
        env = CurriculumWrapper(env, stages=stages, list_seeds=cfg.list_seeds)
        env = AlwaysNAgents(env, num_agents)
    
    if environment_config.autocurriculum:
        print('wrapper autocurriculum')
        
        num_agents = env.config.num_agents
        map_grid_configs = multipleConfigsLoader(environment_config.autocurriculum)
        env = AutoCurriculumWrapper(env, map_grid_configs)
        env = AlwaysNAgents(env, num_agents)
    
    # change-f
    if environment_config.autocurriculum_reverse_goal:
        print('wrapper autocurriculum_reverse_goal')
        env = AutoCurriculumWrapperReverseGoal(env)

    if environment_config.animation_monitor:
        env = AnimationMonitor(env, directory=environment_config.animation_dir)

    env = AutoResetWrapper(env)

    return env

def validate_config(config, list_seeds=None):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            # **exp.environment.dict(),
                            list_seeds=list_seeds,
                            full_config=exp.dict()
                            )
    return exp, flat_config

def main():
    exp, flat_config = validate_config({'environment': {'autocurriculum': 'training_configs/train_maps_3'}})
    # print(exp)
    # print(flat_config)

    env = make_pogema('1', flat_config, {})
    for _ in range(10):
        env.reset()
        env.render()
    print('END')

if __name__ == '__main__':
    main()