import pathlib
import wandb
import os


def load_artifact(name, type_, load_to=None):

    os.environ["WANDB_API_KEY"] = '2493ada22b975bf0409f8240f642b7711e61be40'
    if load_to is None:
        load_to = f'weights/{name}({type_})'
    if pathlib.Path(load_to).exists():
        print(f'Skipping already exist weights: {name} at {pathlib.Path(load_to)}')
        return load_to

    wb_run = wandb.init()
    artifact = wb_run.use_artifact(name, type=type_)

    artifact_dir = artifact.download(root=load_to)
    wandb.finish()
    return artifact_dir


def load_inference_weights():
    load_artifact(name='tviskaron/pogema-appo/pogema-appo-single-agent:v0',
                  type_='moving-obstacles-32',
                  load_to='weights/single_agent_random')

    load_artifact('tviskaron/pogema-appo/pogema-appo-multi-agent:v0',
                  type_='num_agents_64',
                  load_to='weights/multi_agent')

    load_artifact('tviskaron/pogema-appo/pbt:v0',
                  type_='num_agents_64',
                  load_to='weights/pbt')

    load_artifact('tviskaron/pogema-appo/curriculum:v0',
                  type_='num_agents_64',
                  load_to='weights/curriculum')


def main():
    load_inference_weights()


if __name__ == '__main__':
    main()
