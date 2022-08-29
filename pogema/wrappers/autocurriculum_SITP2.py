import gym
from sample_factory.utils.utils import log
import numpy as np
from wrappers.task_create import init_task_maps

class AutoCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, map_config=None, c_name = None, param=None):
        super().__init__(env)
        self._last_metric = 0
        self._tasks_holder = TasksHolder(map_config, **param)

        # information about the value of the last CSR for all tasks
        self._vect_CSR = [0]*len(self._tasks_holder.tasks)

    def step(self, actions):
        observations, reward, done, infos = self.env.step(actions)
        self._tasks_holder.step(self)
        
        # logging information about CSR and ISR
        for info in infos:
            for key in ['ISR', 'CSR']:
                value = info['episode_extra_stats'].get(key, None)
                if value is not None:
                    info['episode_extra_stats'][f"{key}: {self._tasks_holder.task_name()}"] = value
            value = info['episode_extra_stats'].get('CSR', None)
            
        if all(done):
            # CSR for the task on which the agents are trained
            csr = float(all(['TimeLimit.truncated' not in info for info in infos]))
            self._last_metric = csr
            self._vect_CSR[self._tasks_holder.get_task()] = csr
            info['episode_extra_stats']["mean_CSR"] = np.mean(self._vect_CSR)

            for info in infos:
                # logging information which task is used
                for task in self._tasks_holder.tasks:
                    if task.task_name == self._tasks_holder.task_name():
                        info['episode_extra_stats'][f"use: {task.task_name}"] = 1
                    else:
                        info['episode_extra_stats'][f"use: {task.task_name}"] = 0
        return observations, reward, done, infos

    def reset(self,**kwargs):
        self._tasks_holder.task_reset_update_info(self._last_metric)
        if self._tasks_holder.is_solved():
            self._tasks_holder.task_change()
            self._tasks_holder.reset(self)

        return self.env.reset(**kwargs)


class TasksHolder:
    def __init__(self, map_grid_configs, n_repeat=25, a_smooth=0, p_temperature=1, p_more=0.5, max_metric=0.9, **kwargs):
        self.tasks = init_task_maps(map_grid_configs)
        self._n_tasks = len(self.tasks)
        self._mertrics = [2]*self._n_tasks
        self._metrics_tasks = [0]*self._n_tasks

        self._current_task = np.random.randint(self._n_tasks)
        self._task_done = False

        # task_reset_update_info
        self._n_repeat = n_repeat
        self._task_indx = 0
        self._task_metric = [0]*self._n_repeat
        self._current_step = 0

        # step and p_update
        self._max_metric = max_metric
        self._temperature = p_temperature
        self._x_more = p_more
        self._smooth = a_smooth
        log.debug(f'Starting multitask param: n_r = {self._n_repeat}')


    def task_reset_update_info(self, metric):
        # saving information about learning results per task
        self._task_metric[self._task_indx] = metric
        
        if self._task_indx < self._n_repeat-1:
            self._task_indx += 1
        else:
            self._task_indx = 0
            self._task_done = True

            # metric update:
            mean_metric = np.mean(self._task_metric)
            self._mertrics[self._current_task] = self._smooth * self._mertrics[self._current_task] + (1 - self._smooth) * (mean_metric - self._metrics_tasks[self._current_task])
            self._metrics_tasks[self._current_task] = mean_metric


    def is_solved(self):
        return self._task_done

    def p_update(self):
        # task distribution
        x = np.abs(np.array(self._mertrics))*self._temperature
        for i in range(0, self._n_tasks):
            if (self._metrics_tasks[i] > self._max_metric):
                x[i] = -self._x_more
        x = np.e ** x / sum(np.e ** x)
        return x

    def select_task(self, task_indx=-1):
        if task_indx >= 0 and task_indx < self._n_tasks:
            self._current_task = task_indx
        else:
            p_metrics = self.p_update()
            rng = np.random.default_rng()
            self._current_task = rng.choice(range(self._n_tasks), p=p_metrics)

    def task_change(self):
        if self._task_done:
            self.select_task()
            self._task_done = False

    def get_task(self):
        return self._current_task

    def get_metrics(self):
        return self._metrics_tasks

    def task_name(self):
        name = self.tasks[self._current_task].task_name
        if name:
            return name
        else:
            return self._current_task

    def reset(self, wrapper):
        self.tasks[self._current_task].reset(wrapper)
    
    def step(self, wrapper):
        self._current_step += 1
        self.tasks[self._current_task].step(wrapper)
