# Procgen

This is a PyTorch implementation of Success Induced Task Prioritization for Procgen Benchmark based on [Prioritized Level Replay](https://github.com/facebookresearch/level-replay).

## Requirements
```
conda create -n SITP python=3.8
conda activate SITP

git clone 
cd SITP\procgen
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

git clone https://github.com/minqi/gym-minigrid.git
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

### Train PPO with SITP on BigFish

```
python -m train --env_name bigfish \
--num_processes=64 --autocurriculum \
--curriculum_type='SITP' \
--reward_type='num' \
--min_reward=10.0
```
The following table demonstrates minimum reward for different procgen benchmark environments:

|  |  BigFish | Leaper | Plunder | Miner |
| --- | --- | --- | --- | --- |
|min_reward | 10 | 8 | 12 | 10 |