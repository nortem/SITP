# APPO approach for Pogema environment

## Installation
Just install all dependencies using:
```bash
pip install -r docker/requirements.txt
```

## Training APPO
Just run ```train.py``` with one of the configs from the experiments folder:
```bash
python main.py --config_path=training_configs/multiagent-02-moving-obstacles.yaml
python main.py --config_path=training_configs/singleagent-03-moving-obstacles-big.yaml
```

The evaluation results can be reproduced using:
```bash
python eval.py --num_process=16 --configs_path evaluation_configs/00-random'
python eval.py --num_process=16 --configs_path evaluation_configs/01-street-maps/configs/city-street-maps'
python eval.py --num_process=16 --configs_path evaluation_configs/02-dragon-age/configs/dragon-age:origins'
python eval.py --num_process=16 --configs_path evaluation_configs/03-dragon-age2/configs/dragon-age-2'
python eval.py --num_process=16 --configs_path evaluation_configs/04-custom-games'
python eval.py --num_process=16 --configs_path evaluation_configs/05-warcraftIII/configs/warcraft-III-(scaled-to-512x512)'
python eval.py --num_process=80 --configs_path evaluation_configs/06-random-multiagent
```

The movingai.com configuration can be downloaded using:
```bash
python evaluation/download.py
```

## Docker 
We use [crafting](https://pypi.org/project/crafting/) to automate our experiments. 
You can find an example of running such a pipeline in ```run.yaml``` file. 
You need to have installed Docker, Nvidia drivers, and crafting package. 

The crafting package is available in PyPI:
```bash
pip install crafting
```


To build the image run the command below in ```docker``` folder:
```bash
sh build.sh
```

To run an experiment specify target command in ```command``` field in ```run.yaml``` file and turn run in crafting:
```bash
crafting run.yaml
```

Example of ```run.yaml``` file:
```yaml
container:
  image: "pogema:latest"
  command: 'python main.py --config_path experiments/singleagent-01-framestack.yaml'
  tty: True
  environment:
    - "WANDB_API_KEY=<YOUR API KEY>"
    - "OMP_NUM_THREADS=1"
    - "MKL_NUM_THREADS=1"
    - "NVIDIA_VISIBLE_DEVICES=0"
code:
  folder: "."

host_config:
  runtime: nvidia
  shm_size: '4096m'
```


