import argparse
import os
import pathlib
from argparse import Namespace
from collections import defaultdict
from random import shuffle

import yaml

from evaluation.appo import run_ppo_experiment, APPOHolder, run_combined
# from planning.decentralized_mapf import run_multiagent_decentralized
from planning.decentralized import run_multiagent_decentralized
# from planning.dstar import run_dstar_experiment
from evaluation.eval_validation import EvaluationConfig, Algorithm
from evaluation.tabulate_utils import get_table, print_tables, log_plots
from training_run import make_pogema
from utils.gs2dict import generate_variants
from concurrent.futures import ProcessPoolExecutor as Pool

from weights.load_weights import load_inference_weights


def add_results_to_dict(results, resolved_vars, config_results, algo):
    if algo not in config_results:
        config_results[algo] = []
    info = {'.'.join(key): value for key, value in resolved_vars.items()}
    config_results[algo].append({'arguments': info, 'results': results})


def split_on_chunks(data, num_chunks):
    offset = int(1.0 * len(data) / num_chunks + 0.5)
    for i in range(0, num_chunks - 1):
        yield data[i * offset:i * offset + offset]
    yield data[num_chunks * offset - offset:]


def evaluation_run_appo(split, runner=run_ppo_experiment):
    algo: Algorithm = split[0].algo
    ppo = APPOHolder(algo.path_to_weights)

    results = []
    for config in split:
        ppo.cfg.full_config['environment'] = config.environment.dict()

        env = make_pogema('', cfg=ppo.cfg, env_config={'remove_seed': False})
        results.append(runner(ppo, env))
        env.close()

    return results


def eval_planning(func, split):
    results = []
    for config in split:
        env = make_pogema('', cfg=Namespace(
            **dict(full_config=dict(environment=config.environment.dict()))),
                          env_config={'remove_seed': False})
        results.append(func(env))
    return results


def run_algo(configs, num_processess=1):
    algo: Algorithm = configs[0].algo
    results = []

    print(f'starting: {algo}')
    with Pool(num_processess) as executor:
        future_to_stuff = []
        for split in split_on_chunks(configs, num_processess):
            if not split:
                continue
            if algo.name == 'APPO':
                future_to_stuff.append(executor.submit(evaluation_run_appo, split))
            elif algo.name == 'Combined':
                future_to_stuff.append(executor.submit(evaluation_run_appo, split, run_combined))
            elif algo.name == 'Decentralized':
                future_to_stuff.append(executor.submit(eval_planning, run_multiagent_decentralized, split))
            else:
                raise KeyError(f'No algo with name: {algo.name}')
        for future in future_to_stuff:
            results += future.result()
    for index in range(len(configs)):
        configs[index].results = results[index]
        configs[index].environment.grid_config.map = None

    print(f'finished: {algo}')
    return configs


def pop_key(key, d):
    to_extract = d
    for part in key[:-1]:
        if part not in to_extract:
            return None
        to_extract = to_extract[part]
    if key[-1] not in to_extract:
        return None
    return to_extract.pop(key[-1])


def evaluate(configs_path, path_for_saving_results=None, num_process=1, num_gpu_process=1, omp_num_threads: int = 1):
    if omp_num_threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # load_inference_weights()

    configs_path = pathlib.Path(configs_path)
    if path_for_saving_results is None:
        path_for_saving_results = configs_path.name + '-results.yaml'

    configs = configs_path.glob('*.yaml')
    if not configs:
        raise FileNotFoundError("No evaluation configs")

    grouped_by_algo = defaultdict(lambda: [])
    id_ = 0
    for filename in sorted(configs):
        with open(filename) as f:
            evaluation_config = yaml.safe_load(f)

        # temporary remove map to speedup config generation, so don't use grid_search with map!
        map_key = ['environment', 'grid_config', 'map']
        map_value = pop_key(map_key, evaluation_config)
        for resolved_vars, eval_config in generate_variants(evaluation_config):
            c = EvaluationConfig(**eval_config)
            shorted_resolved_vars = {key[-1]: value for key, value in resolved_vars.items() if 'algo' not in key}
            c.algo = Algorithm(**c.algo.dict())
            shorted_resolved_vars['algo'] = c.algo.full_name
            if c.environment.grid_config.map_name:
                shorted_resolved_vars['map_name'] = c.environment.grid_config.map_name

            c.resolved_vars = shorted_resolved_vars

            c.environment.grid_config.map = map_value
            c.id = id_
            id_ += 1
            grouped_by_algo[c.algo.full_name].append(c)

    resulted_configs = []
    for configs in grouped_by_algo.values():
        shuffle(configs)
        # exit(0)

        algo_config: Algorithm = configs[0].algo
        if algo_config.device == 'cpu':
            resulted_configs += run_algo(configs, num_processess=num_process)
        elif algo_config.device == 'cuda':
            resulted_configs += run_algo(configs, num_processess=num_gpu_process)

    to_save = list(map(lambda x: x.dict(exclude_unset=True), resulted_configs))
    print(f'Saving results to: ', path_for_saving_results)
    with open(path_for_saving_results, 'w') as f:
        yaml.dump(to_save, f)

    print_tables(to_save)

    log_plots(to_save, project_name=configs_path.name)


def main():
    parser = argparse.ArgumentParser(description='Parallel evaluation over group of EvaluationConfigs.')
    # parser.add_argument('--configs_path', type=str, action="store", default='evaluation_configs/base-vs-pbt-vs-cur',
    #                     help='path folder with *.yaml EvaluationConfigs', required=False)
    parser.add_argument('--configs_path', type=str, action="store", default='evaluation_configs/curriculum',
                        help='path folder with *.yaml EvaluationConfigs', required=False)
    parser.add_argument('--result_file', type=str, action="store",
                        help='Path to file for saving results.', required=False)
    parser.add_argument('--num_process', type=int, action="store", default=1,
                        help='Number of parallel evaluation workers', required=False)
    parser.add_argument('--num_gpu_process', type=int, action="store", default=1,
                        help='Number of parallel evaluation workers for single cuda device', required=False)
    args = parser.parse_args()
    evaluate(configs_path=args.configs_path,
             path_for_saving_results=args.result_file,
             num_process=args.num_process,
             num_gpu_process=args.num_gpu_process)


if __name__ == '__main__':
    main()
