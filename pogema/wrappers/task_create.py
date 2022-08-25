class Task:
    def __init__(self, func_step, func_reset, task_name = None, param=None):
        self.step = func_step
        self.reset = func_reset
        self._param = param
        self.task_name = task_name
        self.task_step = 0

def t_empty(wrapper):
    pass

def map_reset(wrapper, map_grid_configs):
    i = wrapper._tasks_holder.get_task()
    grid_config = map_grid_configs[i].grid_config
    wrapper.env.unwrapped.config = grid_config
    wrapper.env.config = grid_config
    print(grid_config.map_name)

def init_task_maps(map_grid_configs):
    tasks = [0]*len(map_grid_configs)
    t_reset = lambda wrapper: map_reset(wrapper, map_grid_configs)
    for i in range(len(map_grid_configs)):
        tasks[i] = Task(t_empty, t_reset, task_name=map_grid_configs[i].grid_config.map_name)
    return tasks
