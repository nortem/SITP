import json
import random
from os.path import join

import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict

from evaluation.eval_utils import ResultsHolder
from planning.decentralized import FixLoopsWrapper, NoPathSoRandomOrStayWrapper, DecentralizedAgent
from training_run import validate_config, register_custom_components


class APPOHolder:
    def __init__(self, path, device='cuda'):
        register_custom_components()

        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        exp, flat_config = validate_config(config['full_config'])
        cfg = flat_config

        env = create_env(cfg.env, cfg=cfg, env_config={})

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        env.close()

        # force cpu workers for parallel evaluation
        # cfg.device = 'cpu'
        if device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        # actor_critic.share_memory()
        actor_critic.model_to_device(device)
        policy_id = cfg.policy_index
        checkpoints = join(path, f'checkpoint_p{policy_id}')
        checkpoints = LearnerWorker.get_checkpoints(checkpoints)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = cfg

        self.rnn_states = None

    def act(self, observations):
        if self.rnn_states is None or len(self.rnn_states) != len(observations):
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)

        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)
            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        return actions.cpu().numpy()

    def after_step(self, dones):
        for agent_i, done_flag in enumerate(dones):
            if done_flag:
                self.rnn_states[agent_i] = torch.zeros([get_hidden_size(self.cfg)], dtype=torch.float32,
                                                       device=self.device)


def run_ppo_experiment(appo: APPOHolder, env):
    obs = env.reset()

    results_holder = ResultsHolder()

    with torch.no_grad():
        while True:
            obs, rew, done, infos = env.step(appo.act(obs))
            results_holder.after_step(infos)
            appo.after_step(done)

            if all(done):
                break

    return results_holder.get_final()


def run_combined(appo, env, plan_offset=3):
    obs = env.reset()
    dec_agent = FixLoopsWrapper(NoPathSoRandomOrStayWrapper(DecentralizedAgent(env, obs)))

    results_holder = ResultsHolder()

    with torch.no_grad():
        while True:
            ppo_action = appo.act(obs)
            plan_action = dec_agent.act(obs)

            actions = []
            for agent_idx in range(len(obs)):
                if obs[agent_idx][1].sum().sum() > plan_offset:
                    actions.append(ppo_action[agent_idx])
                else:
                    actions.append(plan_action[agent_idx])

            obs, rew, done, infos = env.step(actions)
            results_holder.after_step(infos)
            appo.after_step(done)

            if all(done):
                break

    return results_holder.get_final()
