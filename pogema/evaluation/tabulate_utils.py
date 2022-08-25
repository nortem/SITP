from collections import defaultdict

import pandas as pd
import yaml
from pandas import DataFrame
from tabulate import tabulate
import wandb
from evaluation.eval_validation import EvaluationConfig, TabulateConfig


def to_pandas(eval_configs):
    data = {}
    for config in eval_configs:
        data[config.id] = {**config.results, **config.resolved_vars}

    return pd.DataFrame.from_dict(data, orient='index')


def join_by_seed(eval_configs):
    metrics = eval_configs[0].tabulate_config.metrics
    eval_configs.sort(key=lambda x: x.id)

    df = to_pandas(eval_configs)

    for key_to_drop in eval_configs[0].tabulate_config.drop_keys:
        if key_to_drop in df.head():
            df = df.drop(key_to_drop, axis=1)

    group_by = [x for x in df.head() if x not in metrics]
    df = df.groupby(by=group_by, as_index=False).mean()

    df = df.round(eval_configs[0].tabulate_config.round_digits)
    df = df.reindex(sorted(df.columns, key=lambda x: (1 if x in metrics else 0, x)), axis=1)
    return df


def get_table(eval_configs):
    df = join_by_seed(eval_configs)
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        return tabulate(df, headers=df.head(), tablefmt='github')


def by_config_mapping(configs):
    full_results = list(map(lambda x: EvaluationConfig(**x), configs))
    mapping = defaultdict(list)
    for cfg in full_results:
        mapping[cfg.name].append(cfg)
    return mapping


def print_tables(configs):
    mapping = by_config_mapping(configs)
    # full_results = list(map(lambda x: EvaluationConfig(**x), configs))
    #
    # mapping = defaultdict(list)
    # for cfg in full_results:
    #     mapping[cfg.name].append(cfg)
    for name in mapping:
        table = get_table(mapping[name])
        table_length = table.find('\n')
        print('-' * table_length)
        # middle =
        print('|' + name.rjust((table_length + 1) // 2), ' ' * (table_length // 2 - 3) + '|', )
        print('-' * table_length)
        print(table)
        print('-' * table_length)
        print()


def log_algo_curve(df: DataFrame, x_keys, project_name, plot_name=None, ):
    for algo in df['algo'].unique():
        wandb.init(name=algo, project='pogema-eval-' + project_name)
        # for map_name in df['map_name'].unique():

        m = df.loc[(df['algo'] == algo)]
        for metric in df.head():
            if metric in x_keys or metric in ['algo', 'map_name']:
                continue

            for x in x_keys:
                table = wandb.Table(data=list(zip(m[x], m[metric])), columns=[x, metric])
                wandb.log({metric + x: wandb.plot.line(table, x, metric, title=f"{plot_name}")})

        wandb.finish()


def log_plots(configs, project_name):
    mapping = by_config_mapping(configs)
    for config_name, configs in mapping.items():
        cfg = configs[0]

        x_keys = []
        for key in cfg.resolved_vars:
            if key == 'algo' or key in cfg.tabulate_config.drop_keys:
                continue
            x_keys.append(key)
        if len(x_keys) != 1:
            print('skipping graph drawing with multiple x keys')
            continue
        df = join_by_seed(mapping[config_name])
        log_algo_curve(df, x_keys=x_keys, plot_name=config_name, project_name=project_name)


def main():
    # with open("results.yaml", "r") as f:
    with open("base-vs-pbt-vs-cur-results.yaml", "r") as f:
        configs = yaml.safe_load(f)

    log_plots(configs, "base-vs-pbt-vs-cur-results.yaml")


if __name__ == '__main__':
    main()
