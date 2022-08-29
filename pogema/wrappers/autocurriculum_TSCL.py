import gym
from sample_factory.utils.utils import log
import numpy as np
from wrappers.task_create import init_task_maps

class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def __call__(self, Q):
        # find the best action with random tie-breaking
        idx = np.where(Q == np.max(Q))[0]
        assert len(idx) > 0, str(Q)
        a = np.random.choice(idx)

        # create a probability distribution
        p = np.zeros(len(Q))
        p[a] = 1

        # Mix in a uniform distribution, to do exploration and
        # ensure we can compute slopes for all tasks
        p = p * (1 - self.epsilon) + self.epsilon / p.shape[0]

        assert np.isclose(np.sum(p), 1)
        return p

class BoltzmannPolicy:
    def __init__(self, temperature=1.):
        self.temperature = temperature

    def __call__(self, Q):
        e = np.exp((Q - np.max(Q)) / self.temperature)
        p = e / np.sum(e)

        assert np.isclose(np.sum(p), 1)
        return p

def estimate_slope(x, y):
    assert len(x) == len(y)
    A = np.vstack([x, np.ones(len(x))]).T
    c, _ = np.linalg.lstsq(A, y)[0]
    return c
    

class AutoCurriculumWrapperTSCL(gym.Wrapper):
    def __init__(self, env, map_config=None, param=None):
        super().__init__(env)
        self._last_metric = 0
        if param['policy_type'] == 'b':
            policy = BoltzmannPolicy()
        elif param['policy_type'] == 'eps':
            policy = EpsilonGreedyPolicy()
        else:
            raise NotImplementedError

        if param['curriculum_type'] == 'Online':
            self._tasks_holder = TasksHolder_OnlineSlopeBanditTeacher(map_config, policy)
        elif param['curriculum_type'] == 'Naive':
            self._tasks_holder = TasksHolder_NaiveSlopeBanditTeacher(map_config, policy)
        else:
            raise NotImplementedError

        self._vect_CSR = [0]*len(self._tasks_holder.tasks)
        # log.debug(f'Starting TSCL curriculum')


    def step(self, actions):
        observations, reward, done, infos = self.env.step(actions)
        self._tasks_holder.step(self)
        for info in infos:
            for key in ['ISR', 'CSR']:
                value = info['episode_extra_stats'].get(key, None)
                if value is not None:
                    info['episode_extra_stats'][f"{key}: {self._tasks_holder.task_name()}"] = value
            value = info['episode_extra_stats'].get('CSR', None)
            if value is not None:
                self._vect_CSR[self._tasks_holder.get_task()] = value
                # self._last_metric = value
                info['episode_extra_stats']["mean_CSR"] = np.mean(self._vect_CSR)
            
        
        #### TSCL
        if all(done):
            csr = float(all(['TimeLimit.truncated' not in info for info in infos]))
            self._last_metric = csr

            # self._last_metric = np.mean([info['episode_extra_stats'].get('ISR', None) for info in infos])
            for task in self._tasks_holder.tasks:
                if task.task_name == self._tasks_holder.task_name():
                    info['episode_extra_stats'][f"use: {task.task_name}"] = 1
                else:
                    info['episode_extra_stats'][f"use: {task.task_name}"] = 0
        ####

        return observations, reward, done, infos

    def reset(self,**kwargs):
        self._tasks_holder.task_reset_update_info(self._last_metric)
        if self._tasks_holder.is_solved():
            self._tasks_holder.task_change()
            self._tasks_holder.reset(self)

        return self.env.reset(**kwargs)

class TasksHolder_OnlineSlopeBanditTeacher:
    def __init__(self, map_grid_configs, policy, lr=0.1):
        self.tasks = init_task_maps(map_grid_configs)
        self._n_tasks = len(self.tasks)
        self._current_task = np.random.randint(self._n_tasks)
        self._task_done = True
        self._current_step = 0

        self.policy = policy
        self.lr = lr
        self.Q = np.zeros(self._n_tasks)
        self.prevr = np.zeros(self._n_tasks)
        self.abs = abs


    def task_reset_update_info(self, metric):
        indx = self._current_task
        s = metric - self.prevr[indx]
        s = np.nan_to_num(s)
        self.Q[indx] += self.lr * (s - self.Q[indx])
        self.prevr[indx] = metric

    def is_solved(self):
        return self._task_done

    def select_task(self, task_indx=-1):
        if task_indx >= 0 and task_indx < self._n_tasks:
            self._current_task = task_indx
        else:
            p_metrics = self.policy(np.abs(self.Q) if self.abs else self.Q)
            self._current_task = np.random.choice(range(self._n_tasks), p=p_metrics)

    def task_change(self):
        if self._task_done:
            self.select_task()

    def get_task(self):
        return self._current_task

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

class TasksHolder_NaiveSlopeBanditTeacher:
    def __init__(self, map_grid_configs, policy, lr=0.1, window_size=10):
        self.tasks = init_task_maps(map_grid_configs)
        self._n_tasks = len(self.tasks)
        self._current_task = np.random.randint(self._n_tasks)
        self._task_done = True
        self._current_step = 0
        self._current_task_step = 0

        self.policy = policy
        self.lr = lr
        self.window_size = window_size
        self.Q = np.ones(self._n_tasks)
        self.scores = np.zeros(self.window_size)
        self.abs = abs


    def task_reset_update_info(self, metric):
        self.scores[self._current_task_step] = metric
        if self._current_task_step < self.window_size - 1:
            self._current_task_step += 1
        else:
            indx = self._current_task
            s = estimate_slope(list(range(len(self.scores))), self.scores)
            s = np.nan_to_num(s)
            self.Q[indx] += self.lr * (s - self.Q[indx])
            self._task_done = True

    def is_solved(self):
        return self._task_done

    def select_task(self, task_indx=-1):
        if task_indx >= 0 and task_indx < self._n_tasks:
            self._current_task = task_indx
        else:
            p_metrics = self.policy(np.abs(self.Q) if self.abs else self.Q)
            self._current_task = np.random.choice(range(self._n_tasks), p=p_metrics)

    def task_change(self):
        if self._task_done:
            self.select_task()

    def get_task(self):
        return self._current_task

    def task_name(self):
        name = self.tasks[self._current_task].task_name
        if name:
            return name
        else:
            return self._current_task

    def reset(self, wrapper):
        self.tasks[self._current_task].reset(wrapper)
        self._task_done = False
        self._current_task_step = 0
    
    def step(self, wrapper):
        self._current_step += 1
        self.tasks[self._current_task].step(wrapper)