# Prioritized Level Replay

This is a PyTorch implementation of [Prioritized Level Replay](https://arxiv.org/abs/2010.03934).

## Requirements
```
conda create -n level-replay python=3.8
conda activate level-replay

git clone https://github.com/facebookresearch/level-replay.git
cd level-replay
pip install -r requirements.txt

# Clone a level-replay-compatible version of OpenAI Baselines.
git clone https://github.com/minqi/baselines.git
cd baselines 
python setup.py install
cd ..

# Clone level-replay-compatible versions of Procgen and MiniGrid environments.
git clone https://github.com/minqi/procgen.git
cd procgen 
python setup.py install
cd ..

git clone https://github.com/minqi/gym-minigrid .git
cd gym-minigrid 
pip install -e .
cd ..
```

Note that you may run into cmake finding an incompatible version of g++. You can manually specify the path to a compatible g++ by setting the path to the right compiler in `procgen/procgen/CMakeLists.txt` before the line `project(codegen)`:
```
...
# Manually set the c++ compiler here
set(CMAKE_CXX_COMPILER "/share/apps/gcc-9.2.0/bin/g++")

project(codegen)
...
```

## Examples
### Train PPO with value-based level reply with rank prioritization on BigFish
```
python -m train --env_name bigfish \
--num_processes=64 \
--level_replay_strategy='value_l1' \
--level_replay_score_transform='rank' \
--level_replay_temperature=0.1 \
--staleness_coef=0.1
```


## Acknowledgements
The PPO implementation is largely based on Ilya Kostrikov's excellent implementation (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) and Roberta Raileanu's specific integration with Procgen (https://github.com/rraileanu/auto-drac).
