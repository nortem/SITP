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
from wrappers.pogema_wrappers import AlwaysNAgents
from wrappers.autocurriculum_SITP2 import AutoCurriculumWrapper
from wrappers.autocurriculum_TSCL import AutoCurriculumWrapperTSCL

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
        map_grid_configs = multipleConfigsLoader(environment_config.path_to_grid_configs)
        env = MultipleConfigsWrapper(env, environment_config.path_to_grid_configs, map_grid_configs)

    if environment_config.autocurriculum.curriculum_type:
        if 'SITP' in environment_config.autocurriculum.curriculum_type:
            log.debug(f'Starting multitask curriculum')
            num_agents = env.config.num_agents
            curr_param = environment_config.autocurriculum
            if environment_config.autocurriculum.map_config:
                map_grid_configs = multipleConfigsLoader(curr_param.map_config)
            else:
                map_grid_configs = None
            env = AutoCurriculumWrapper(env, map_grid_configs, curr_param.curriculum_type, curr_param.param)
            env = AlwaysNAgents(env, num_agents)

        if 'TSCL' in environment_config.autocurriculum.curriculum_type:
            log.debug(f'Starting TSCL curriculum')
            num_agents = env.config.num_agents
            curr_param = environment_config.autocurriculum
            if environment_config.autocurriculum.map_config:
                map_grid_configs = multipleConfigsLoader(curr_param.map_config)
            else:
                map_grid_configs = None
            env = AutoCurriculumWrapperTSCL(env, map_grid_configs, curr_param.param)
            env = AlwaysNAgents(env, num_agents)

    if environment_config.animation_monitor:
        env = AnimationMonitor(env, directory=environment_config.animation_dir)

    env = AutoResetWrapper(env)

    return env


def override_default_params_func(env, parser):
    parser.set_defaults(
        encoder_custom='pogema_residual',
        hidden_size=128,
    )


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='Pogema-v0',
        make_env_func=make_pogema,
        override_default_params_func=override_default_params_func,
    )

    register_custom_encoder('pogema_residual', ResnetEncoder)

    EXTRA_EPISODIC_STATS_PROCESSING.append(pogema_extra_episodic_stats_processing)
    EXTRA_PER_POLICY_SUMMARIES.append(pogema_extra_summaries)


def pogema_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    pass


def pogema_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    # for key in ['ISR', 'CSR', 'grid_size', 'num_agents', 'use_seeds', 'stage_step', 'mean_CSR']:
    for key in ['grid_size', 'num_agents', 'use_seeds', 'stage_step', 'reverse_goal', 'mean_reverse_goal']:
        if key not in policy_avg_stats:
            continue
        avg = np.mean(policy_avg_stats[key])
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f'{key}: {round(float(avg), 3)}')

    for key in policy_avg_stats:
        for metric in ['ISR', 'CSR', 'use']:
            if metric in key:
                avg = np.mean(policy_avg_stats[key])
                summary_writer.add_scalar(key, avg, env_steps)
                log.debug(f'{key}: {round(float(avg), 3)}')


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
    register_custom_components()

    import argparse

    parser = argparse.ArgumentParser(description='Process training config.')

    parser.add_argument('--config_path', type=str, action="store",
                        help='path to yaml file with single run configuration', required=False)

    parser.add_argument('--raw_config', type=str, action='store',
                        help='raw json config', required=False)

    parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                        help='Run wandb in thread mode. Usefull for some setups.', required=False)
    
    params = parser.parse_args()

    if params.raw_config:
        config = json.loads(params.raw_config)
    else:
        if params.config_path is None:
            raise ValueError("You should specify --config_path or --raw_config argument!")
        with open(params.config_path, "r") as f:
            config = yaml.safe_load(f)

    list_seeds = []
    exp, flat_config = validate_config(config, list_seeds)
    if exp.global_settings.use_wandb:
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project=exp.name, config=exp.dict(), save_code=False, sync_tensorboard=True)

    status = run_algorithm(flat_config)

    return status


if __name__ == '__main__':
    sys.exit(main())
