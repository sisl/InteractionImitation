from imitation.algorithms import adversarial, bc
from imitation.util import logger, util
from stable_baselines3 import PPO, DQN, SAC
from soft_q import SQLPolicy
from sqil import SQILReplayBuffer
from stable_baselines3.common import policies
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.rewards import discrim_nets
import numpy as np
import argparse
from utils import make_sa_dataloader, make_sads_dataloader, make_sa_dataset, linear_schedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from adril import AdRILWrapper, AdRILReplayBuffer
import os
from gym.spaces import Discrete
import gym
from advil import advil_training
from stable_baselines3.common.running_mean_std import RunningMeanStd

from advil import AdVILPolicy, AdVILDiscriminator

def train_bc(env, n=0):
    venv = util.make_vec_env(env, n_envs=8)
    if isinstance(venv.action_space, Discrete):
        w = 64
    else:
        w = 256
    for i in range(n):
        mean_rewards = []
        std_rewards = []
        for num_trajs in range(0, 26, 5):
            if num_trajs == 0:
                expert_data = make_sa_dataloader(env, normalize=False)
            else:
                expert_data = make_sa_dataloader(env, max_trajs=num_trajs, normalize=False)
            bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=expert_data,
                               policy_class=policies.ActorCriticPolicy,
                               ent_weight=0., l2_weight=0., policy_kwargs=dict(net_arch=[w, w]))
            if num_trajs > 0:
                bc_trainer.train(n_batches=int(5e5))

            def get_policy(*args, **kwargs):
                return bc_trainer.policy
            model = PPO(get_policy, env, verbose=1)
            model.save(os.path.join("learners", env,
                                    "bc_{0}_{1}".format(i, num_trajs)))
            mean_reward, std_reward = evaluate_policy(
                    model, model.get_env(), n_eval_episodes=10)               
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("{0} Trajs: {1}".format(num_trajs, mean_reward))
            np.savez(os.path.join("learners", env, "bc_rewards_{0}".format(
                i)), means=mean_rewards, stds=std_rewards)


def train_gail(env, n=0):
    venv = util.make_vec_env(env, n_envs=8)
    if isinstance(venv.action_space, Discrete):
        w = 64
    else:
        w = 256
    expert_data = make_sads_dataloader(env, max_trajs=5)
    logger.configure(os.path.join("learners", "GAIL"))

    for i in range(n):
        discrim_net = discrim_nets.ActObsMLP(
                action_space=venv.action_space,
                observation_space=venv.observation_space,
                hid_sizes=(w, w),
                )
        gail_trainer = adversarial.GAIL(venv, expert_data=expert_data, expert_batch_size=32,
                                        gen_algo=PPO("MlpPolicy", venv, verbose=1, n_steps=1024,
                                                     policy_kwargs=dict(net_arch=[w, w])),
                                                     discrim_kwargs={'discrim_net': discrim_net})
        mean_rewards = []
        std_rewards = []
        for train_steps in range(20):
            if train_steps > 0:
                if 'Bullet' in env:
                    gail_trainer.train(total_timesteps=25000)
                else:
                    gail_trainer.train(total_timesteps=16384)

            def get_policy(*args, **kwargs):
                return gail_trainer.gen_algo.policy
            model = PPO(get_policy, env, verbose=1)
            mean_reward, std_reward = evaluate_policy(
                model, model.env, n_eval_episodes=10)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("{0} Steps: {1}".format(train_steps, mean_reward))
            np.savez(os.path.join("learners", env, "gail_rewards_{0}".format(i)),
                     means=mean_rewards, stds=std_rewards)


def train_sqil(env, n=0):
    venv = gym.make(env)
    expert_data = make_sa_dataset(env, max_trajs=5)

    for i in range(n):
        if isinstance(venv.action_space, Discrete):
            model = DQN(SQLPolicy, venv, verbose=1, policy_kwargs=dict(net_arch=[64, 64]), learning_starts=1)
        else:
            model = SAC('MlpPolicy', venv, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
                        learning_rate=linear_schedule(7.3e-4), train_freq=64, gradient_steps=64, gamma=0.98, tau=0.02)

        model.replay_buffer = SQILReplayBuffer(model.buffer_size, model.observation_space,
                                               model.action_space, model.device, 1,
                                               model.optimize_memory_usage, expert_data=expert_data)
        mean_rewards = []
        std_rewards = []
        for train_steps in range(20):
            if train_steps > 0:
                if 'Bullet' in env:
                    model.learn(total_timesteps=25000, log_interval=1)
                else:
                    model.learn(total_timesteps=16384, log_interval=1)
            mean_reward, std_reward = evaluate_policy(
                model, model.env, n_eval_episodes=10)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("{0} Steps: {1}".format(train_steps, mean_reward))
            np.savez(os.path.join("learners", env, "sqil_rewards_{0}".format(i)),
                     means=mean_rewards, stds=std_rewards)


def train_adril(env, n=0, balanced=False):
    num_trajs = 20
    expert_data = make_sa_dataset(env, max_trajs=num_trajs)
    n_expert = len(expert_data["obs"])
    expert_sa = np.concatenate((expert_data["obs"], np.reshape(expert_data["acts"], (n_expert, -1))), axis=1)

    for i in range(0, n):
        venv = AdRILWrapper(gym.make(env))
        mean_rewards = []
        std_rewards = []
        # Create model
        if isinstance(venv.action_space, Discrete):
            model = DQN(SQLPolicy, venv, verbose=1, policy_kwargs=dict(net_arch=[64, 64]), learning_starts=1)
        else:
            model = SAC('MlpPolicy', venv, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
                        learning_rate=linear_schedule(7.3e-4), train_freq=64, gradient_steps=64, gamma=0.98, tau=0.02)
        model.replay_buffer = AdRILReplayBuffer(model.buffer_size, model.observation_space,
                                               model.action_space, model.device, 1,
                                               model.optimize_memory_usage, expert_data=expert_data, N_expert=num_trajs,
                                               balanced=balanced)
        if not balanced:
            for j in range(len(expert_sa)):
                obs = expert_data["obs"][j]
                act = expert_data["acts"][j]
                next_obs = expert_data["next_obs"][j]
                done = expert_data["dones"][j]
                model.replay_buffer.add(obs, next_obs, act, -1, done)    
        for train_steps in range(400):
            # Train policy
            if train_steps > 0:
                if 'Bullet' in env:
                    model.learn(total_timesteps=1250, log_interval=1000)
                else:
                    model.learn(total_timesteps=25000, log_interval=1000)
                if train_steps % 1 == 0: # written to support more complex update schemes
                    model.replay_buffer.set_iter(train_steps)
                    model.replay_buffer.set_n_learner(venv.num_trajs)
            
            # Evaluate policy
            if train_steps % 20 == 0:
                model.set_env(gym.make(env))
                mean_reward, std_reward = evaluate_policy(
                    model, model.env, n_eval_episodes=10)
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
                print("{0} Steps: {1}".format(int(train_steps * 1250), mean_reward))
                np.savez(os.path.join("learners", env, "adril_rewards_{0}".format(i)),
                        means=mean_rewards, stds=std_rewards)
            # Update env
            if train_steps > 0:
                if train_steps % 1  == 0:
                    venv.set_iter(train_steps + 1)
            model.set_env(venv)


def train_advil(env, policy_class=AdVILPolicy, discriminator_class=AdVILDiscriminator,
                iters=int(1e5), lr_pi=8e-6, lr_f=8e-4):
    venv = gym.make(env)
    expert_data = make_sa_dataloader(
        env,
        normalize=False,
        batch_size=1024,
    )
    pi = advil_training(
        expert_data,
        venv,
        iters=iters,
        policy_class=policy_class,
        discriminator_class=discriminator_class, 
        lr_pi=lr_pi,
        lr_f=lr_f,
    )
    return pi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train expert policies.')
    parser.add_argument(
        '-a', '--algo', choices=['bc', 'gail', 'sqil', 'adril', 'advil', 'all'], required=True)
    parser.add_argument('-e', '--env', choices=['cartpole', 'lunarlander', 'acrobot', 'pendulum', 'halfcheetah', 'walker', 'hopper', 'ant'],
                        required=True)
    parser.add_argument('-n', '--num_runs', required=False)
    args = parser.parse_args()
    if args.env == "cartpole":
        envname = 'CartPole-v1'
    elif args.env == "lunarlander":
        envname = 'LunarLander-v2'
    elif args.env == "acrobot":
        envname = 'Acrobot-v1'
    elif args.env == "pendulum":
        envname = 'Pendulum-v0'
    elif args.env == "halfcheetah":
        envname = 'HalfCheetahBulletEnv-v0'
    elif args.env == "walker":
        envname = 'Walker2DBulletEnv-v0'
    elif args.env == "hopper":
        envname = 'HopperBulletEnv-v0'
    elif args.env == "ant":
        envname = 'AntBulletEnv-v0'
    else:
        print("ERROR: unsupported env.")
    if args.num_runs is not None and args.num_runs.isdigit():
        num_runs = int(args.num_runs)
    else:
        num_runs = 1
    if args.algo == 'bc':
        train_bc(envname, num_runs)
    elif args.algo == 'gail':
        train_gail(envname, num_runs)
    elif args.algo == 'sqil':
        train_sqil(envname, num_runs)
    elif args.algo == 'adril':
        train_adril(envname, num_runs)
    elif args.algo == 'advil':
        train_advil(envname, num_runs)
    elif args.algo == 'all':
        train_bc(envname, num_runs)
        train_gail(envname, num_runs)
        train_sqil(envname, num_runs)
        train_adril(envname, num_runs)
        train_advil(envname, num_runs)
    else:
        print("ERROR: unsupported algorithm")
