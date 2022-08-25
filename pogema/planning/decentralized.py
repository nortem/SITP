from pogema import GridConfig
import gym
import random

from evaluation.eval_utils import ResultsHolder
from planning.tools import Closed, Open, Node, make_path, manhattan_distance, update_obs
from wrappers.pogema_wrappers import LogPogemaStats, MultiTimeLimit

INF = 1000000007


def has_loop(path, next_step):
    # next_step = Node(next_step[0], next_step[1])
    if len(path) > 1:
        if path[-1] == next_step or path[-2] == next_step:
            return True
    return False


def get_random_move(gridMap, current):
    deltas = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    random.shuffle(deltas)
    for delta in deltas:
        i = current.i + delta[0]
        j = current.j + delta[1]
        if 0 > i or i >= len(gridMap) or 0 > j or j >= len(gridMap[0]):
            continue
        if gridMap[i][j] != 0:
            continue
        return delta
    return 0, 0


def a_star_multiagent(gridMap, iStart, jStart, iGoal, jGoal, other_poses=None, heuristicFunction=manhattan_distance):
    open = Open()
    closed = Closed()
    start_node = Node(iStart, jStart, 0, heuristicFunction(iStart, jStart, iGoal, jGoal))
    good_moves = []
    if gridMap[iStart][jStart] != 0:
        return None
    open.add_node(start_node)
    k = 1
    while not open.is_empty():
        current = open.get_best_node()
        closed.add_node(current)

        if current.i == iGoal and current.j == jGoal:
            return make_path(current)
        deltas = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        for delta in deltas:
            i = current.i + delta[0]
            j = current.j + delta[1]
            if 0 > i or i >= len(gridMap) or 0 > j or j >= len(gridMap[0]):
                continue
            if gridMap[i][j] != 0:
                continue

            new_node = Node(i, j)
            if not closed.was_expanded(new_node):
                new_node.g = current.g + 1
                new_node.h = heuristicFunction(i, j, iGoal, jGoal)
                new_node.F = new_node.g + new_node.h
                new_node.k = k
                new_node.parent = current
                if (new_node.i, new_node.j) not in other_poses:
                    open.add_node(new_node)
                    if current.g == 0:
                        good_moves.append(new_node)
        k += 1
    best_move = None
    for move in good_moves:
        if best_move is None or move.F < best_move.F:
            best_move = move
    if best_move is not None:
        return make_path(best_move)
    else:
        return None


def run_multiagent_decentralized(env, memory_limit=None):
    gc: GridConfig = env.config

    size = gc.size
    obs_radius = gc.obs_radius

    random.seed(env.config.seed)
    obs = env.reset()
    actions = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
    steps = 0
    grids = [[[0] * (size + obs_radius * 2) for _ in range(size + obs_radius * 2)] for j in range(gc.num_agents)]
    for i in range(gc.num_agents):
        for k in range(size):
            grids[i][k + obs_radius][obs_radius - 1] = INF
            grids[i][obs_radius - 1][k + obs_radius] = INF
            grids[i][size + obs_radius][k + obs_radius] = INF
            grids[i][k + obs_radius][size + obs_radius] = INF

    done = [False, ...]
    for k in range(gc.num_agents):
        update_obs(grids[k], obs[k][0], env.grid.positions_xy[k][0] - obs_radius,
                   env.grid.positions_xy[k][1] - obs_radius, steps, memory_limit)
    costs = [INF for _ in range(gc.num_agents)]
    paths = [[] for _ in range(gc.num_agents)]
    results_holder = ResultsHolder()

    while not all(done):
        action = []
        for k in range(gc.num_agents):
            if env.grid.positions_xy[k][0] == env.grid.finishes_xy[k][0] and env.grid.positions_xy[k][1] == \
                    env.grid.finishes_xy[k][1] and costs[k] == INF:
                costs[k] = steps - 1
            other_poses = []
            for i in range(gc.num_agents):
                if i != k and (not gc.disappear_on_goal or costs[i] == INF) \
                        and abs(env.grid.positions_xy[i][0] - env.grid.positions_xy[k][0]) <= obs_radius \
                        and abs(env.grid.positions_xy[i][1] - env.grid.positions_xy[k][1]) <= obs_radius:
                    other_poses.append((env.grid.positions_xy[i][0], env.grid.positions_xy[i][1]))

            path = a_star_multiagent(grids[k],
                                     env.grid.positions_xy[k][0],
                                     env.grid.positions_xy[k][1],
                                     env.grid.finishes_xy[k][0],
                                     env.grid.finishes_xy[k][1],
                                     other_poses)
            if path is not None and len(path) > 1:
                in_loop = has_loop(paths[k], path[1])
                if in_loop and random.randint(0, 1) == 1:
                    action.append(0)
                else:
                    action.append(actions[(path[1].i - path[0].i, path[1].j - path[0].j)])
                paths[k].append(path[0])
            else:
                action.append(0)
                if random.randint(0, 1) == 1:
                    move = get_random_move(grids[k], Node(env.grid.positions_xy[k][0], env.grid.positions_xy[k][1]))
                    action[-1] = actions[move]
                paths[k].append(Node(env.grid.positions_xy[k][0], env.grid.positions_xy[k][1]))
        obs, reward, done, infos = env.step(action)
        steps += 1

        results_holder.after_step(infos)

        for k in range(gc.num_agents):
            update_obs(grids[k], obs[k][0], env.grid.positions_xy[k][0] - obs_radius,
                       env.grid.positions_xy[k][1] - obs_radius, steps, memory_limit)
        if all(done):
            break
    env.reset()

    return results_holder.get_final()


class DecentralizedAgent:
    def __init__(self, env, obs, memory_limit=None):
        self.env = env

        gc: GridConfig = env.config
        self.memory_limit = memory_limit
        size = gc.size
        obs_radius = gc.obs_radius

        random.seed(env.config.seed)
        self.actions = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
        self.steps = 0
        self.grids = [[[0] * (size + obs_radius * 2) for _ in range(size + obs_radius * 2)] for j in
                      range(gc.num_agents)]
        for i in range(gc.num_agents):
            for k in range(size):
                self.grids[i][k + obs_radius][obs_radius - 1] = INF
                self.grids[i][obs_radius - 1][k + obs_radius] = INF
                self.grids[i][size + obs_radius][k + obs_radius] = INF
                self.grids[i][k + obs_radius][size + obs_radius] = INF

        for k in range(gc.num_agents):
            update_obs(self.grids[k], obs[k][0], env.grid.positions_xy[k][0] - obs_radius,
                       env.grid.positions_xy[k][1] - obs_radius, self.steps, memory_limit)
        self.costs = [INF for _ in range(gc.num_agents)]
        self.paths = [[] for _ in range(gc.num_agents)]

    def act(self, obs):

        env = self.env
        cfg = self.env.config

        for k in range(cfg.num_agents):
            update_obs(self.grids[k], obs[k][0], self.env.grid.positions_xy[k][0] - cfg.obs_radius,
                       self.env.grid.positions_xy[k][1] - cfg.obs_radius, self.steps, self.memory_limit)

        action = []
        for k in range(cfg.num_agents):
            if env.grid.positions_xy[k][0] == env.grid.finishes_xy[k][0] and env.grid.positions_xy[k][1] == \
                    env.grid.finishes_xy[k][1] and self.costs[k] == INF:
                self.costs[k] = self.steps - 1
            other_poses = []
            for i in range(cfg.num_agents):
                if i != k and self.costs[i] == INF \
                        and abs(env.grid.positions_xy[i][0] - env.grid.positions_xy[k][0]) <= cfg.obs_radius \
                        and abs(env.grid.positions_xy[i][1] - env.grid.positions_xy[k][1]) <= cfg.obs_radius:
                    other_poses.append((env.grid.positions_xy[i][0], env.grid.positions_xy[i][1]))
            path = a_star_multiagent(self.grids[k],
                                     env.grid.positions_xy[k][0],
                                     env.grid.positions_xy[k][1],
                                     env.grid.finishes_xy[k][0],
                                     env.grid.finishes_xy[k][1],
                                     other_poses)
            if path and len(path) > 1:
                self.paths[k] = path
                action.append(self.actions[(path[1].i - path[0].i, path[1].j - path[0].j)])
            else:
                action.append(None)
        return action

    def get_path(self, agent_idx):
        return self.paths[agent_idx]


class NoPathSoRandomOrStayWrapper:

    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env

    def act(self, obs):
        actions = self.agent.act(obs)
        for idx in range(len(actions)):
            if actions[idx] is None:
                if random.randint(0, 1) == 1:
                    actions[idx] = 0
                else:
                    # actions[idx] = self.env.action_space.sample()
                    actions[idx] = self.get_random_move(self.agent.grids[idx], Node(self.env.grid.positions_xy[idx][0],
                                                                self.env.grid.positions_xy[idx][1]))
        return actions

    @staticmethod
    def get_random_move(gridMap, current):
        deltas = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        random.shuffle(deltas)
        for idx, delta in enumerate(deltas):
            i = current.i + delta[0]
            j = current.j + delta[1]
            if 0 > i or i >= len(gridMap) or 0 > j or j >= len(gridMap[0]):
                continue
            if gridMap[i][j] != 0:
                continue
            return idx
        return 0

    def get_path(self, agent_idx):
        return self.agent.get_path(agent_idx)


class FixLoopsWrapper(NoPathSoRandomOrStayWrapper):
    def __init__(self, agent):
        super().__init__(agent)
        self.previous_positions = [[] for _ in range(self.env.config.num_agents)]

    def act(self, obs):
        cfg: GridConfig = self.env.config
        actions = self.agent.act(obs)
        for idx in range(len(actions)):
            self.previous_positions[idx].append(self.env.grid.positions_xy[idx])
            path = self.previous_positions[idx]
            if len(path) > 1:
                next_step = self.previous_positions[idx][-1]
                dx, dy = cfg.MOVES[actions[idx]]
                next_pos = dx + next_step[0], dy + next_step[1]

                if path[-1] == next_pos or path[-2] == next_pos:
                    if random.randint(0, 1) == 1:
                        actions[idx] = 0
        return actions


def run(env):
    results_holder = ResultsHolder()
    obs = env.reset()
    agent = FixLoopsWrapper(NoPathSoRandomOrStayWrapper(DecentralizedAgent(env, obs)))

    dones = [False]

    while not all(dones):
        obs, rewards, dones, infos = env.step(agent.act(obs))
        results_holder.after_step(infos)

    return results_holder.get_final()


def main():
    grid_config = GridConfig(size=64, density=0.3, num_agents=128)

    env = gym.make('Pogema-v0', config=grid_config)
    # env = AnimationMonitor(env, 'renders')
    env = MultiTimeLimit(env, max_episode_steps=512)
    env = LogPogemaStats(env)
    # env = AnimationMonitor(env, directory='renders')

    for _ in range(10):
        print(run(env))
    print('-----' * 5)
    for _ in range(10):
        print(run_multiagent_decentralized(env))


if __name__ == '__main__':
    main()