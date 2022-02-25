import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv as VecEnv

from src.core.reparam_module import ReparamPolicy
from tqdm import tqdm
from src.core.gail import train_discriminator, roll_buffer, TerminalLogger
from dataclasses import dataclass
from src.safe_options.policy_gradient import trpo_step, ppo_step
import torch.nn.functional as F

from src.options.envs import OptionsEnv
from src.safe_options.collisions import feasible

from intersim.envs import IntersimpleLidarFlatIncrementingAgent
from src.util.wrappers import Setobs, TransformObservation

@dataclass
class Buffer:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

@dataclass
class HLBuffer:
    states: torch.Tensor
    safe_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

@dataclass
class OptionsRollout:
    hl: HLBuffer
    ll: Buffer

def gail(env_fn, expert_data, discriminator, disc_opt, disc_iters, policy, value,
         v_opt, v_iters, epochs, rollout_episodes, rollout_steps, gamma,
         gae_lambda, delta, backtrack_coeff, backtrack_iters, cg_iters=10, cg_damping=0.1, wasserstein=False, wasserstein_c=None, logger=TerminalLogger(), callback=None):

    policy(torch.zeros(env_fn(0).observation_space['observation'].shape), torch.zeros(env_fn(0).observation_space['safe_actions'].shape))
    policy = ReparamPolicy(policy)

    logger.add_scalar('expert/mean_episode_length', (~expert_data.dones).sum() / expert_data.states.shape[0])
    logger.add_scalar('expert/mean_reward_per_episode', expert_data.rewards[~expert_data.dones].sum() / expert_data.states.shape[0])

    for epoch in tqdm(range(epochs)):
        hl_data, ll_data = rollout(env_fn, policy, rollout_episodes, rollout_steps)
        generator_data = OptionsRollout(HLBuffer(*hl_data), Buffer(*ll_data))

        generator_data.ll.actions += 0.1 * torch.randn_like(generator_data.ll.actions)

        logger.add_scalar('gen/mean_episode_length', (~generator_data.ll.dones).sum() / generator_data.ll.states.shape[0], epoch)
        gen_mean_reward_per_episode = generator_data.hl.rewards[~generator_data.hl.dones].sum() / generator_data.hl.states.shape[0]
        logger.add_scalar('gen/mean_reward_per_episode', gen_mean_reward_per_episode, epoch)
        logger.add_scalar('gen/unsafe_probability_mass', policy.unsafe_probability_mass(policy(generator_data.hl.states[~generator_data.hl.dones], generator_data.hl.safe_actions[~generator_data.hl.dones])).mean(), epoch)

        discriminator, loss = train_discriminator(expert_data, generator_data.ll, discriminator, disc_opt, disc_iters, wasserstein, wasserstein_c)
        if wasserstein:
            generator_data.ll.rewards = discriminator(generator_data.ll.states, generator_data.ll.actions)
        else:
            generator_data.ll.rewards = -F.logsigmoid(discriminator(generator_data.ll.states, generator_data.ll.actions))
        logger.add_scalar('disc/final_loss', loss, epoch)
        logger.add_scalar('disc/mean_reward_per_episode', generator_data.ll.rewards[~generator_data.ll.dones].sum() / generator_data.ll.states.shape[0], epoch)

        #assert generator_data.ll.rewards.shape == generator_data.ll.dones.shape
        generator_data.hl.rewards = torch.where(~generator_data.ll.dones, generator_data.ll.rewards, torch.tensor(0.)).sum(-1)

        value, policy = trpo_step(value, policy, generator_data.hl.states, generator_data.hl.safe_actions, generator_data.hl.actions, generator_data.hl.rewards, generator_data.hl.dones, gamma, gae_lambda, delta, backtrack_coeff, backtrack_iters, v_opt, v_iters, cg_iters, cg_damping)
        expert_data = roll_buffer(expert_data, shifts=-3, dims=0)

        if callback is not None:
            callback({
                'epoch': epoch,
                'value': value,
                'policy': policy,
                'gen/mean_reward_per_episode': gen_mean_reward_per_episode,
            })
    
    return value, policy

def gail_ppo(env_fn, expert_data, discriminator, disc_opt, disc_iters, policy, value,
         v_opt, v_iters, epochs, rollout_episodes, rollout_steps, gamma,
         gae_lambda, clip_ratio, pi_opt, pi_iters, target_kl=None, max_grad_norm=None, wasserstein=False, wasserstein_c=None, logger=TerminalLogger(), callback=None, lr_schedulers=[]):

    logger.add_scalar('expert/mean_episode_length', (~expert_data.dones).sum() / expert_data.states.shape[0])
    logger.add_scalar('expert/mean_reward_per_episode', expert_data.rewards[~expert_data.dones].sum() / expert_data.states.shape[0])

    for epoch in range(epochs):
        hl_data, ll_data = rollout(env_fn, policy, rollout_episodes, rollout_steps)
        generator_data = OptionsRollout(HLBuffer(*hl_data), Buffer(*ll_data))

        generator_data.ll.actions += 0.1 * torch.randn_like(generator_data.ll.actions)

        logger.add_scalar('gen/mean_episode_length', (~generator_data.ll.dones).sum() / generator_data.ll.states.shape[0], epoch)
        gen_mean_reward_per_episode = generator_data.hl.rewards[~generator_data.hl.dones].sum() / generator_data.hl.states.shape[0]
        logger.add_scalar('gen/mean_reward_per_episode', gen_mean_reward_per_episode, epoch)
        logger.add_scalar('gen/unsafe_probability_mass', policy.unsafe_probability_mass(policy(generator_data.hl.states[~generator_data.hl.dones], generator_data.hl.safe_actions[~generator_data.hl.dones])).mean(), epoch)

        discriminator, loss = train_discriminator(expert_data, generator_data.ll, discriminator, disc_opt, disc_iters, wasserstein, wasserstein_c)
        if wasserstein:
            generator_data.ll.rewards = discriminator(generator_data.ll.states, generator_data.ll.actions)
        else:
            generator_data.ll.rewards = -F.logsigmoid(discriminator(generator_data.ll.states, generator_data.ll.actions))
        logger.add_scalar('disc/final_loss', loss, epoch)
        logger.add_scalar('disc/mean_reward_per_episode', generator_data.ll.rewards[~generator_data.ll.dones].sum() / generator_data.ll.states.shape[0], epoch)
        
        #assert generator_data.ll.rewards.shape == generator_data.ll.dones.shape
        generator_data.hl.rewards = torch.where(~generator_data.ll.dones, generator_data.ll.rewards, torch.tensor(0.)).sum(-1)

        value, policy = ppo_step(value, policy, generator_data.hl.states, generator_data.hl.safe_actions, generator_data.hl.actions, generator_data.hl.rewards, generator_data.hl.dones, clip_ratio, gamma, gae_lambda, pi_opt, pi_iters, v_opt, v_iters, target_kl, max_grad_norm)
        expert_data = roll_buffer(expert_data, shifts=-3, dims=0)

        if callback is not None:
            callback({
                'epoch': epoch,
                'value': value,
                'policy': policy,
                'gen/mean_reward_per_episode': gen_mean_reward_per_episode,
            })
    
    for lr_scheduler in lr_schedulers:
        lr_scheduler.step()
    
    return value, policy

def rollout(env_fn, policy, n_episodes, max_steps_per_episode):
    env = env_fn(0)

    states = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.observation_space['observation'].shape)
    safe_actions = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.observation_space['safe_actions'].shape)
    actions = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.action_space.shape)
    rewards = torch.zeros(n_episodes, max_steps_per_episode + 1)
    dones = torch.ones(n_episodes, max_steps_per_episode + 1, dtype=bool)

    ll_states = torch.zeros(n_episodes, max_steps_per_episode, env.max_plan_length + 1, *env.observation_space['observation'].shape)
    ll_actions = torch.zeros(n_episodes, max_steps_per_episode, env.max_plan_length + 1, *env.ll_action_space.shape)
    ll_rewards = torch.zeros(n_episodes, max_steps_per_episode, env.max_plan_length + 1)
    ll_dones = torch.ones(n_episodes, max_steps_per_episode, env.max_plan_length + 1, dtype=bool)

    env = VecEnv(list(map(lambda i: (lambda: env_fn(i)), range(n_episodes))))

    obs = env.reset()
    states[:, 0] = torch.tensor(obs['observation']).clone().detach()
    safe_actions[:, 0] = torch.tensor(obs['safe_actions']).clone().detach()
    dones[:, 0] = False

    for s in tqdm(range(max_steps_per_episode), 'Rollout'):
        actions[:, s] = policy.sample(policy(states[:, s], safe_actions[:, s])).clone().detach()

        clipped_actions = actions[:, s]
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_actions = torch.clamp(clipped_actions, torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))

        o, r, d, info = env.step(clipped_actions)
        states[:, s + 1] = torch.tensor(o['observation']).clone().detach()
        safe_actions[:, s + 1] = torch.tensor(o['safe_actions']).clone().detach()
        rewards[:, s] = torch.tensor(r).clone().detach()
        dones[:, s + 1] = torch.tensor(d).clone().detach()

        ll_states[:, s] = torch.from_numpy(np.stack([i['ll']['observations'] for i in info])).clone().detach()
        ll_actions[:, s] = torch.from_numpy(np.stack([i['ll']['actions'] for i in info])).clone().detach()
        ll_rewards[:, s] = torch.from_numpy(np.stack([i['ll']['rewards'] for i in info])).clone().detach()
        ll_dones[:, s] = torch.from_numpy(np.stack([i['ll']['plan_done'] for i in info])).clone().detach()

    dones = dones.cumsum(1) > 0

    states = states[:, :max_steps_per_episode]
    safe_actions = safe_actions[:, :max_steps_per_episode]
    actions = actions[:, :max_steps_per_episode]
    rewards = rewards[:, :max_steps_per_episode]
    dones = dones[:, :max_steps_per_episode]

    return (states, safe_actions, actions, rewards, dones), (ll_states, ll_actions, ll_rewards, ll_dones)

class SafeOptionsEnv(OptionsEnv):

    def __init__(self, env, options, safe_actions_collision_method=None, abort_unsafe_collision_method=None):
        super().__init__(env, options)
        self.safe_actions_collision_method = safe_actions_collision_method
        self.abort_unsafe_collision_method = abort_unsafe_collision_method
        self.observation_space = gym.spaces.Dict({
            'observation': self.observation_space,
            'safe_actions': gym.spaces.Box(low=0., high=1., shape=(self.action_space.n,)),
        })

    def safe_actions(self):
        if self.safe_actions_collision_method is None:
            return np.ones(len(self.options), dtype=bool)

        plans = [self.plan(o) for o in self.options]
        plans = [np.pad(p, (0, self.max_plan_length - len(p)), constant_values=np.nan) for p in plans]
        plans = np.stack(plans)
        safe = feasible(self.env, plans, method=self.safe_actions_collision_method)
        if not safe.any():
            # action 0 is considered safe fallback
            safe[0] = True

        return safe
    
    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        obs = {
            'observation': obs,
            'safe_actions': self.safe_actions(),
        }
        return obs

    def step(self, action, render_mode=None):
        obs, reward, done, info = super().step(action, render_mode)
        obs = {
            'observation': obs,
            'safe_actions': self.safe_actions(),
        }
        return obs, reward, done, info

    def execute_plan(self, obs, option, render_mode=None):
        observations = np.zeros((self.max_plan_length + 1, *self.env.observation_space.shape))
        actions = np.zeros((self.max_plan_length + 1, *self.ll_action_space.shape))
        rewards = np.zeros((self.max_plan_length + 1,))
        env_done = np.ones((self.max_plan_length + 1,), dtype=bool)
        plan_done = np.ones((self.max_plan_length + 1,), dtype=bool)
        infos = []

        plan = self.plan(option)
        observations[0] = obs
        env_done[0] = False
        for k, u in enumerate(plan):
            plan_done[k] = False
            o, r, d, i = self.env.step(u)
            actions[k] = u
            rewards[k] = r
            env_done[k] = d
            infos.append(i)
            observations[k+1] = o

            if render_mode is not None:
                self.env.render(render_mode)

            if d:
                break

            if self.abort_unsafe_collision_method is not None and \
                not feasible(self.env, plan[k:], method=self.abort_unsafe_collision_method):
                break
        
        n_steps = k + 1
        return observations, actions, rewards, env_done, plan_done, infos, n_steps

obs_min = np.array([
    [-1000, -1000, 0, -np.pi, -1e-1, 0.],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
]).reshape(-1)

obs_max = np.array([
    [1000, 1000, 20, np.pi, 1e-1, 0.],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
]).reshape(-1)

def NormalizedSafeOptionsEvalEnv(safe_actions_collision_method=None, abort_unsafe_collision_method=None, **kwargs):
    return SafeOptionsEnv(Setobs(
        TransformObservation(IntersimpleLidarFlatIncrementingAgent(
            n_rays=5,
            **kwargs,
        ), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
    ), options=[(0, 5), (1, 5), (2, 5), (4, 5), (6, 5), (8, 5), (10, 5)], safe_actions_collision_method=safe_actions_collision_method, abort_unsafe_collision_method=abort_unsafe_collision_method)
