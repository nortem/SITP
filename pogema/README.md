# SITP for Pogema environment

## Installation
Just install all dependencies using:
```bash
pip install -r docker/requirements.txt
```

## Training APPO
Just run ```train.py``` with one of the configs from the experiments folder:
```bash
python main.py --config_path=training_configs/mac5.yaml
```

- mac5.yaml and mac10.yaml -- SITP.
- mac5_b.yaml and mac10_b.yaml -- baseline.
- mac5_t.yaml and mac10_t.yaml -- TSCL.

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
  command: 'python main.py --config_path=training_configs/mac5.yaml'
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
