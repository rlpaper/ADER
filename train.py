import gym
import roboschool
import argparse
import numpy as np
import parl
from parl.utils import logger, CSVLogger, ReplayMemory
import os

from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from alg import ADER

import copy
import time

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 1e4
BATCH_SIZE = 256
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise


def run_train_episode(env, agent, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    max_action = float(env.action_space.high[0])
    while True:
        steps += 1

        if rpm.size() < WARMUP_SIZE:
            action = env.action_space.sample()
        else:
            action = np.random.normal(
                agent.predict(np.array(obs)), max_action * EXPL_NOISE).clip(
                    -max_action, max_action)

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)

        obs = next_obs
        total_reward += reward

        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        if done:
            break
    return total_reward, steps


def run_evaluate_episode(env, agent, eval_episodes=10):
    max_steps = env._max_episode_steps 
    total_reward = 0.
    fall = 0.
    total_steps_list = []
    for i in range(eval_episodes):
        obs = env.reset()
        steps = 0
        while True:
            action = agent.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                if steps < max_steps:
                    fall += 1
                total_steps_list.append(steps)
                break
    total_reward /= eval_episodes
    fall /= eval_episodes
    return total_reward, fall, total_steps_list


def main():
    env = gym.make(args.env)
    env.seed(args.seed)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = MujocoModel(obs_dim, act_dim, max_action)
    algorithm = ADER(
        model,
        max_action=max_action,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        kappa=args.kappa,
        epoch=args.epoch,
        alpha=args.alpha)
    agent = MujocoAgent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    test_flag = 0
    total_steps = 0
    while total_steps < args.train_total_steps:
        train_reward, steps = run_train_episode(env, agent, rpm)
        total_steps += steps
        logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))

        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            evaluate_reward, evaluate_fall_rate, total_steps_list = run_evaluate_episode(env, agent)
            mean_steps = np.mean(total_steps_list)
            logger.info('Steps {}, Evaluate reward: {}, Fall rate: {}'.format(
                total_steps, evaluate_reward, evaluate_fall_rate))
            logger.info('Steps {}, Mean episode steps: {}, Steps list: {}'.format(
                total_steps, mean_steps, total_steps_list))
            res = {
                'eval_step': mean_steps,
                'fall_rate': evaluate_fall_rate,
                'Step': total_steps,
                'Value': evaluate_reward
            }
            csv_logger.log_dict(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', help='Mujoco environment name', default='HalfCheetah-v1')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(55e5),
        help='maximum training steps')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='the step interval between two consecutive evaluations')
    parser.add_argument('--kappa', type=float, default=float(5), help='kappa') 
    parser.add_argument('--epoch', type=float, default=float(10000), help='epoch') 
    parser.add_argument('--alpha', type=float, default=float(2), help='alpha') 
    parser.add_argument('--seed', type=int, default=int(1), help='env seed')

    args = parser.parse_args()
    
    logger.set_dir('./train_log/{}_k_{}_e_{}_a_{}_s_{}_{}'.format(args.env, str(args.kappa), str(args.epoch), str(args.alpha), str(args.seed), time.strftime("%H%M%S")))
    csv_logger = CSVLogger(os.path.join(logger.get_dir(), 'ADER_{}_{}.csv'.format(args.env, str(args.seed))))
    main()
