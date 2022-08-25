# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import numpy as np
import torch

class RewardCSR:
    def __init__(self, reward_type='num', **params):
        self.reward_type = reward_type
        self.min_reward = params.get('min_reward', None)
        self.max = 1
        self.mean = 0
        self.mean_arr = np.zeros(10)
        self.indx = 0
        self.info = {}

    def reward_info(self, rewards):
        if self.reward_type=='max':
            self.max = np.max(rewards)
            if self.max <= 0:
                self.max = 1
        elif self.reward_type=='part':
            self.max = np.max([np.max(rewards), self.max])
            self.mean_arr[self.indx%10] = np.mean(rewards)
            self.mean = np.max([np.mean(self.mean_arr), self.mean])
            min_reward = (self.max - self.mean)*0.2 + self.mean
            self.min_reward = max(self.min_reward, min_reward)
            self.indx += 1

    def update_info(self, p_metrics):
        self.info['p.max'] = p_metrics.max()
        self.info['p.min'] = p_metrics.min()
        self.info['min_reward'] = self.min_reward
    
    def reward_to_CSR(self, reward):
        if self.reward_type in {'num', 'part'}:
            return int(reward > self.min_reward)
        elif self.reward_type=='max':
            return reward/self.max
        elif self.reward_type=='rew':
            return reward


class LevelSamplerCurriculum():
    def __init__(
        self, seeds, obs_space, action_space, 
        num_envs=1, c_name='SITP-2', **param):
        self.obs_space = obs_space
        self.action_space = action_space

        # Track seeds and scores as in np arrays backed by shared memory
        self._init_seed_index(seeds)

        #AC!
        self._last_metric = 0
        self._tasks_holder = TasksHolderMap(self.seeds, num_envs, **param)        
        self._vect_CSR = [0]*len(self._tasks_holder.tasks)


    def seed_range(self):
        return (int(min(self.seeds)), int(max(self.seeds)))

    def _init_seed_index(self, seeds):
        self.seeds = np.array(seeds, dtype=np.int64)
        self.seed2index = {seed: i for i, seed in enumerate(seeds)}

    def update_with_rollouts(self, rollouts): #TODO train 189
        pass

    # def update_seed_score(self, actor_index, seed_idx, score, num_steps):
    #     pass

    def after_update(self): #TODO train 194
        # Reset partial updates, since weights have changed, and thus logits are now stale
        pass


    def curriculum_update(self, seeds, rewards, reward2CSR): #TODO train 194
        reward2CSR.reward_info(rewards)
        for i in range(len(seeds)):
            indx = self.seed2index[seeds[i]]
            reward = reward2CSR.reward_to_CSR(rewards[i])
            self._tasks_holder.task_reset_update_info(indx, reward)
        self._tasks_holder.p_update()
        self._tasks_holder.reset()
        reward2CSR.update_info(self._tasks_holder.p_metrics)


    def sample(self, strategy=None): #TODO envs 126, 154, 216, 244
        seed = self._tasks_holder.select_task()
        return seed

    def sample_weights(self): #TODO train 250
        pass


class TasksHolderMap:
    def __init__(self, seeds, num_envs, n_repeat=25, a_smooth=0, p_temperature=5, p_more=0.5, max_metric=30, **kwargs):
        self.tasks = seeds
        self._n_tasks = len(self.tasks)

        self._n_repeat = min(n_repeat, num_envs)
        self.num_envs = num_envs
        self._n_parts = num_envs // self._n_repeat
        self._n_repeat = num_envs // self._n_parts

        self._mertrics = [2]*self._n_tasks
        self._metrics_tasks = [0]*self._n_tasks

        # self._task_done = [False]*self._n_parts

        # task_reset_update_info
        self._task_metric = np.zeros((self._n_tasks, self._n_repeat))

        # step and p_update
        self._current_step = 0 #np.zeros(self._n_parts)
        self._max_metric = max_metric

        self._temperature = p_temperature
        self._x_more = p_more
        # self._current_tasks = 0

        self.p_metrics = None
        self._task_indx = np.zeros(self._n_tasks, np.int32)
        self._task_indx_s = 0
        self._smooth = a_smooth

    def reset(self):
        pass

    def task_reset_update_info(self, indx, metric):
        self._task_metric[indx, self._task_indx[indx]] = metric
        
        if self._task_indx[indx] < self._n_repeat-1:
            self._task_indx[indx] += 1
        else:
            self._task_indx[indx] = 0

            # metric update:
            mean_metric = np.mean(self._task_metric[indx])
            # self._mertrics[indx] = mean_metric - self._metrics_tasks[indx]
            self._mertrics[indx] = self._smooth * self._mertrics[indx] + (1 - self._smooth) * (mean_metric - self._metrics_tasks[indx])
            self._metrics_tasks[indx] = mean_metric
            

    def p_update(self):
        x = np.array(self._mertrics)
        # x[x < 0] = 0.1
        x = np.abs(x)*self._temperature
        for i in range(0, self._n_tasks):
            if (self._metrics_tasks[i] > self._max_metric):
                x[i] = -self._x_more
        x = np.e ** x / sum(np.e ** x)
        self.p_metrics = x

    def select_task(self, task_indx=-1):
        if task_indx >= 0 and task_indx < self._n_tasks:
            self._current_task = task_indx
        else:
            if self._task_indx_s%self._n_repeat==0:
                p_metrics = self.p_metrics
                rng = np.random.default_rng()
                self._current_task = rng.choice(range(self._n_tasks), p=p_metrics)
            self._task_indx_s += 1
        return self._current_task
