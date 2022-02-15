import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv as VecEnv

from core.reparam_module import ReparamPolicy
from tqdm import tqdm
from core.gail import Buffer, train_discriminator, roll_buffer, TerminalLogger
from dataclasses import dataclass
from core.trpo import trpo_step
from core.ppo import ppo_step
import torch.nn.functional as F

@dataclass
class OptionsRollout:
    hl: Buffer
    ll: Buffer

def gail(env_fn, expert_data, discriminator, disc_opt, disc_iters, policy, value,
         v_opt, v_iters, epochs, rollout_episodes, rollout_steps, gamma,
         gae_lambda, delta, backtrack_coeff, backtrack_iters, cg_iters=10, cg_damping=0.1, wasserstein=False, wasserstein_c=None, logger=TerminalLogger()):

    policy(torch.zeros(env_fn(0).observation_space.shape))
    policy = ReparamPolicy(policy)

    logger.add_scalar('expert/mean_episode_length', (~expert_data.dones).sum() / expert_data.states.shape[0])
    logger.add_scalar('expert/mean_reward_per_episode', expert_data.rewards[~expert_data.dones].sum() / expert_data.states.shape[0])

    for epoch in tqdm(range(epochs)):
        hl_data, ll_data = rollout(env_fn, policy, rollout_episodes, rollout_steps)
        generator_data = OptionsRollout(Buffer(*hl_data), Buffer(*ll_data))

        generator_data.ll.actions += 0.1 * torch.randn_like(generator_data.ll.actions)

        logger.add_scalar('gen/mean_episode_length', (~generator_data.ll.dones).sum() / generator_data.ll.states.shape[0], epoch)
        logger.add_scalar('gen/mean_reward_per_episode', generator_data.hl.rewards[~generator_data.hl.dones].sum() / generator_data.hl.states.shape[0], epoch)

        discriminator, loss = train_discriminator(expert_data, generator_data.ll, discriminator, disc_opt, disc_iters, wasserstein, wasserstein_c)
        if wasserstein:
            generator_data.ll.rewards = discriminator(generator_data.ll.states, generator_data.ll.actions)
        else:
            generator_data.ll.rewards = -F.logsigmoid(discriminator(generator_data.ll.states, generator_data.ll.actions))
        logger.add_scalar('disc/final_loss', loss, epoch)
        logger.add_scalar('disc/mean_reward_per_episode', generator_data.ll.rewards[~generator_data.ll.dones].sum() / generator_data.ll.states.shape[0], epoch)

        #assert generator_data.ll.rewards.shape == generator_data.ll.dones.shape
        generator_data.hl.rewards = torch.where(~generator_data.ll.dones, generator_data.ll.rewards, torch.tensor(0.)).sum(-1)

        value, policy = trpo_step(value, policy, generator_data.hl.states, generator_data.hl.actions, generator_data.hl.rewards, generator_data.hl.dones, gamma, gae_lambda, delta, backtrack_coeff, backtrack_iters, v_opt, v_iters, cg_iters, cg_damping)
        expert_data = roll_buffer(expert_data, shifts=-3, dims=0)
    
    return value, policy

def gail_ppo(env_fn, expert_data, discriminator, disc_opt, disc_iters, policy, value,
         v_opt, v_iters, epochs, rollout_episodes, rollout_steps, gamma,
         gae_lambda, clip_ratio, pi_opt, pi_iters, target_kl=None, max_grad_norm=None, wasserstein=False, wasserstein_c=None, logger=TerminalLogger()):

    logger.add_scalar('expert/mean_episode_length', (~expert_data.dones).sum() / expert_data.states.shape[0])
    logger.add_scalar('expert/mean_reward_per_episode', expert_data.rewards[~expert_data.dones].sum() / expert_data.states.shape[0])

    for epoch in range(epochs):
        hl_data, ll_data = rollout(env_fn, policy, rollout_episodes, rollout_steps)
        generator_data = OptionsRollout(Buffer(*hl_data), Buffer(*ll_data))

        generator_data.ll.actions += 0.1 * torch.randn_like(generator_data.ll.actions)

        logger.add_scalar('gen/mean_episode_length', (~generator_data.ll.dones).sum() / generator_data.ll.states.shape[0], epoch)
        logger.add_scalar('gen/mean_reward_per_episode', generator_data.hl.rewards[~generator_data.hl.dones].sum() / generator_data.hl.states.shape[0], epoch)

        discriminator, loss = train_discriminator(expert_data, generator_data.ll, discriminator, disc_opt, disc_iters, wasserstein, wasserstein_c)
        if wasserstein:
            generator_data.ll.rewards = discriminator(generator_data.ll.states, generator_data.ll.actions)
        else:
            generator_data.ll.rewards = -F.logsigmoid(discriminator(generator_data.ll.states, generator_data.ll.actions))
        logger.add_scalar('disc/final_loss', loss, epoch)
        logger.add_scalar('disc/mean_reward_per_episode', generator_data.ll.rewards[~generator_data.ll.dones].sum() / generator_data.ll.states.shape[0], epoch)
        
        #assert generator_data.ll.rewards.shape == generator_data.ll.dones.shape
        generator_data.hl.rewards = torch.where(~generator_data.ll.dones, generator_data.ll.rewards, torch.tensor(0.)).sum(-1)

        value, policy = ppo_step(value, policy, generator_data.hl.states, generator_data.hl.actions, generator_data.hl.rewards, generator_data.hl.dones, clip_ratio, gamma, gae_lambda, pi_opt, pi_iters, v_opt, v_iters, target_kl, max_grad_norm)
        expert_data = roll_buffer(expert_data, shifts=-3, dims=0)
    
    return value, policy

def rollout(env_fn, policy, n_episodes, max_steps_per_episode):
    env = env_fn(0)

    states = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.observation_space.shape)
    actions = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.action_space.shape)
    rewards = torch.zeros(n_episodes, max_steps_per_episode + 1)
    dones = torch.ones(n_episodes, max_steps_per_episode + 1, dtype=bool)

    ll_states = torch.zeros(n_episodes, max_steps_per_episode, env.max_plan_length + 1, *env.observation_space.shape)
    ll_actions = torch.zeros(n_episodes, max_steps_per_episode, env.max_plan_length + 1, *env.ll_action_space.shape)
    ll_rewards = torch.zeros(n_episodes, max_steps_per_episode, env.max_plan_length + 1)
    ll_dones = torch.ones(n_episodes, max_steps_per_episode, env.max_plan_length + 1, dtype=bool)

    env = VecEnv(list(map(lambda i: (lambda: env_fn(i)), range(n_episodes))))

    states[:, 0] = torch.tensor(env.reset()).clone().detach()
    dones[:, 0] = False

    for s in tqdm(range(max_steps_per_episode), 'Rollout'):
        actions[:, s] = policy.sample(policy(states[:, s])).clone().detach()

        clipped_actions = actions[:, s]
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_actions = torch.clamp(clipped_actions, torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))

        o, r, d, info = env.step(clipped_actions)
        states[:, s + 1] = torch.tensor(o).clone().detach()
        rewards[:, s] = torch.tensor(r).clone().detach()
        dones[:, s + 1] = torch.tensor(d).clone().detach()

        ll_states[:, s] = torch.from_numpy(np.stack([i['ll']['observations'] for i in info])).clone().detach()
        ll_actions[:, s] = torch.from_numpy(np.stack([i['ll']['actions'] for i in info])).clone().detach()
        ll_rewards[:, s] = torch.from_numpy(np.stack([i['ll']['rewards'] for i in info])).clone().detach()
        ll_dones[:, s] = torch.from_numpy(np.stack([i['ll']['plan_done'] for i in info])).clone().detach()

    dones = dones.cumsum(1) > 0

    states = states[:, :max_steps_per_episode]
    actions = actions[:, :max_steps_per_episode]
    rewards = rewards[:, :max_steps_per_episode]
    dones = dones[:, :max_steps_per_episode]

    return (states, actions, rewards, dones), (ll_states, ll_actions, ll_rewards, ll_dones)

class OptionsEnv(gym.Wrapper):

    def __init__(self, env, options):
        super().__init__(env)
        self.ll_action_space = env.action_space
        self.options = options
        self.action_space = gym.spaces.Discrete(len(options))
        self.max_plan_length = max(t for _, t in options)
    
    def plan(self, option):
        target_v, t = option
        current_v = self.env._env.state[self.env._agent, 1].item()
        dt = self.env._env._dt
        a = (target_v - current_v) / (t * dt)
        a = self.env._normalize(a)
        a = a * np.ones((t,))
        a += 0.01 * np.random.randn(*a.shape)
        a = np.clip(a, self.ll_action_space.low, self.ll_action_space.high)
        return a

    def execute_plan(self, obs, option, render_mode=None):
        observations = np.zeros((self.max_plan_length + 1, *self.env.observation_space.shape))
        actions = np.zeros((self.max_plan_length + 1, *self.ll_action_space.shape))
        rewards = np.zeros((self.max_plan_length + 1,))
        env_done = np.ones((self.max_plan_length + 1,), dtype=bool)
        plan_done = np.ones((self.max_plan_length + 1,), dtype=bool)
        infos = []

        observations[0] = obs
        env_done[0] = False
        for k, u in enumerate(self.plan(option)):
            plan_done[k] = False
            o, r, d, i = super().step(u)
            actions[k] = u
            rewards[k] = r
            env_done[k+1] = d
            infos.append(i)
            observations[k+1] = o

            if render_mode is not None:
                self.env.render(render_mode)

            if d:
                break
        
        n_steps = k + 1
        return observations, actions, rewards, env_done, plan_done, infos, n_steps

    def step(self, action, render_mode=None):
        a = int(action)
        assert a == action
        ll_obs, ll_actions, ll_rewards, ll_env_done, ll_plan_done, ll_infos, ll_steps = self.execute_plan(self.last_obs, self.options[a], render_mode)
        hl_obs = ll_obs[ll_steps]
        hl_reward = (ll_rewards * ~ll_plan_done).sum().item()
        hl_done = ll_env_done[ll_steps].item()
        hl_infos = {
            'll': {
                'observations': ll_obs,
                'actions': ll_actions,
                'rewards': ll_rewards,
                'env_done': ll_env_done,
                'plan_done': ll_plan_done,
                'infos': ll_infos,
                'steps': ll_steps,
            }
        }
        self.last_obs = hl_obs
        return hl_obs, hl_reward, hl_done, hl_infos
    
    def reset(self, *args, **kwargs):
        self.last_obs = super().reset(*args, **kwargs)
        return self.last_obs
