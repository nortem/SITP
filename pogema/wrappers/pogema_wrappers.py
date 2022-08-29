import random
from collections import deque
from pathlib import Path

import gym
import yaml
from gym.wrappers import TimeLimit
from pogema import GridConfig
from sample_factory.algorithms.utils.multi_agent_wrapper import is_multiagent_env
import numpy as np

from utils.config_validation import Environment
from utils.gs2dict import generate_variants

def multipleConfigsLoader(path_to_grid_configs):
    _path = Path(path_to_grid_configs)
    _env_configs = []
    for ec_path in _path.glob("*.yaml"):
            with open(ec_path, "r") as f:
                raw_config = yaml.safe_load(f)
                for resolved_vars, spec in generate_variants(raw_config):
                    ec = Environment(**spec)
                    _env_configs.append(ec)
    return _env_configs


class MultipleConfigsWrapper(gym.Wrapper):

    def __init__(self, env, path_to_grid_configs, map_grid_configs):
        super().__init__(env)

        self._path = Path(path_to_grid_configs)
        self._env_configs = map_grid_configs
        self._current = None

        self._dict_map_name = dict()
        i = 0
        for conf in self._env_configs:
            self._dict_map_name[conf.grid_config.map_name] = i
            i += 1
        self._vect_CSR = [0]*len(map_grid_configs)


    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        for info in infos:
            for key in ['ISR', 'CSR']:
                value = info['episode_extra_stats'].get(key, None)
                if value is not None:
                    info['episode_extra_stats'][f"{key}: {self._current.grid_config.map_name}"] = value
            value = info['episode_extra_stats'].get('CSR', None)
            if value is not None:
                self._vect_CSR[self._dict_map_name[self._current.grid_config.map_name]] = value
                info['episode_extra_stats']["mean_CSR"] = np.mean(self._vect_CSR)
        return obs, reward, done, infos

    def reset(self, **kwargs):

        self._current = random.choice(self._env_configs)
        self.env.unwrapped.config = self._current.grid_config
        self.env.config = self._current.grid_config
        return self.env.reset(**kwargs)


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        if all(dones):
            observations = self.env.reset()
        return observations, rewards, dones, infos

class LogPogemaStats(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._ISR = None

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        for agent_idx in range(self.env.config.num_agents):
            infos[agent_idx]['episode_extra_stats'] = infos[agent_idx].get('episode_extra_stats', {})

            if done[agent_idx]:
                if agent_idx not in self._ISR:
                    self._ISR[agent_idx] = float('TimeLimit.truncated' not in infos[agent_idx])

        if is_multiagent_env(self.env) and all(done):
            not_tl_truncated = all(['TimeLimit.truncated' not in info for info in infos])
            infos[0]['episode_extra_stats'].update(CSR=float(not_tl_truncated))

            for agent_idx in range(self.env.config.num_agents):
                infos[agent_idx]['episode_extra_stats'].update(ISR=self._ISR[agent_idx])

        return obs, reward, done, infos

    def reset(self, **kwargs):
        self._ISR = {}
        return self.env.reset(**kwargs)


class MultiTimeLimit(TimeLimit):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            for agent_idx in range(self.env.config.num_agents):
                info[agent_idx]["TimeLimit.truncated"] = not done[agent_idx]
            done = [True] * self.env.config.num_agents
        return observation, reward, done, info


class PogemaStackFramesWrapper(gym.Wrapper):

    def __init__(self, env, framestack):
        super().__init__(env)
        self._frames: list = None

        self.stack_past_frames = framestack

        full_size = self.config.obs_radius * 2 + 1
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3 * framestack, full_size, full_size))

    def _render_stacked_frames(self):
        result = [np.concatenate(self._frames[agent_idx]) for agent_idx in range(len(self._frames))]
        return result

    def step(self, action):
        new_observation, reward, done, info = self.env.step(action)
        for agent_idx, obs in enumerate(new_observation):
            self._frames[agent_idx].popleft()
            self._frames[agent_idx].append(new_observation[agent_idx])
        return self._render_stacked_frames(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        self._frames = []
        for obs in observation:
            self._frames.append(deque([obs] * self.stack_past_frames))
        return self._render_stacked_frames()


class SeedWrapper(gym.Wrapper):
    def __init__(self, env, seeds):
        self._seeds = seeds
        self._seed_cnt = 0
        super().__init__(env)

    def set_seed(self):
        self.env.config.seed = self._seeds[self._seed_cnt]
        self._seed_cnt = (self._seed_cnt + 1) % len(self._seeds)

    def step(self, actions):
        observations, reward, dones, infos = self.env.step(actions)
        for agent_idx, _ in enumerate(infos):
            infos[agent_idx]['seed'] = self.env.config.seed

        return observations, reward, dones, infos

    def reset(self, **kwargs):
        self.set_seed()
        return self.env.reset(**kwargs)


class PogemaEvaluationMonitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._results = {}

    def step(self, actions):
        observations, reward, dones, infos = self.env.step(actions)
        for agent_idx in range(len(infos)):
            info = infos[agent_idx]
            if 'CSR' in info['episode_extra_stats']:
                self._results[info['seed']] = info['episode_extra_stats']['CSR']
                max_len = 999
                if len(self._results) >= max_len:
                    print(self._results)
                    print(sum([value for value in self._results.values()]) / max_len)
                    exit(0)
        return observations, reward, dones, infos

   
class AlwaysNAgents(gym.Wrapper):
    def __init__(self, env, max_num_agents=64) -> None:
        super().__init__(env)
        self.num_agents = max_num_agents
        self.env.num_agents = max_num_agents
        self._max_num_agents = max_num_agents

    def step(self, actions):

        if len(actions) > self._max_num_agents:
            raise KeyError("Number of agents can't exceed max_num_agents")

        observations, reward, done, infos = self.env.step(actions[:self.config.num_agents])
        if len(done) != self._max_num_agents:
            for _ in range(len(done), self._max_num_agents):
                observations.append(observations[0])
                reward.append(reward[0])
                done.append(done[0])
                infos.append({'is_active': False})
        return observations, reward, done, infos

    def reset(self):
        observations = self.env.reset()
        if len(observations) != self._max_num_agents:
            for i in range(len(observations), self._max_num_agents):
                observations.append(observations[0])
        return observations
