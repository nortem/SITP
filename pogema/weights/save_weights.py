import wandb


def save_weights(path, name, type):
    wb_run = wandb.init(project='pogema-appo')
    wb_run.log_artifact(artifact_or_path=path, name=name, type=type)
    wb_run.finish()


def main():
    save_weights(path='weights/curriculum', name='curriculum', type='num_agents_64')


if __name__ == '__main__':
    main()
