## ADER:Adapting between Exploration and Robustness for Actor-Critic Methods

This is the supplementary code for AAAI2021 review. Submission ID: 4287.

## How to use
### Dependencies:
+ python>=3.6
+ [pytorch>=1.2.0](https://github.com/pytorch/pytorch)
+ [parl>=1.3](https://github.com/PaddlePaddle/PARL)
+ gym
+ mujoco-py
+ roboschool

*mujoco tasks are tested using `mujoco_py==0.5.7` and `gym==0.9.2`* <br/>
*roboschool tasks are tested using `roboschool==1.0.48` and `gym==0.15.4`*

### Start Training:
```
# The experimental results can be reproduced by running:
sh run.sh

# Example: To train an agent for HalfCheetah-v1 game with the random seed
python train.py --env HalfCheetah-v1 --seed 1
```

Hyper-parameters can be modified with different arguments to train.py.