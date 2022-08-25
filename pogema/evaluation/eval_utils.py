import time

import numpy as np


class ResultsHolder:
    def __init__(self):
        self.results = dict()

        self.times = []
        self.isr = []
        self.step = 0
        self.start_time = time.monotonic()

    def after_step(self, infos):
        self.step += 1

        for agent_idx in range(len(infos)):
            if 'ISR' in infos[agent_idx]['episode_extra_stats']:
                self.isr.append(infos[agent_idx]['episode_extra_stats']['ISR'])
                self.times.append(self.step)

            if 'CSR' in infos[agent_idx]['episode_extra_stats']:
                self.results['CSR'] = infos[agent_idx]['episode_extra_stats']['CSR']

    def get_final(self):
        self.results['FPS'] = self.step / (time.monotonic() - self.start_time)
        self.results['flowtime'] = sum(self.times)
        self.results['makespan'] = self.step
        self.results['ISR'] = float(np.mean(self.isr))

        return self.results
