import gym
import torch
from src.util.collisions import feasible
import numpy as np
from collections import deque
import itertools

def imitation_discriminator(discriminator):
    return lambda obs, action, next_obs, done: discriminator.discrim_net.predict_reward_train(
        state=torch.tensor(obs).unsqueeze(0).to(discriminator.discrim_net.device()),
        action=torch.tensor([[action]]).to(discriminator.discrim_net.device()),
        next_state=torch.tensor(next_obs).unsqueeze(0).to(discriminator.discrim_net.device()), # unused
        done=torch.tensor(done).unsqueeze(0).to(discriminator.discrim_net.device()), # unused
    ).item()

class OptionsEnv(gym.Wrapper):

    def __init__(self, env, options, discriminator, ll_buffer_capacity, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self.options = options
        num_hl_options = len(self.options)
        self.action_space = gym.spaces.Discrete(num_hl_options)
        self.observation_space = gym.spaces.Dict({
            'obs': env.observation_space,
            'mask': gym.spaces.Box(low=0, high=1, shape=(num_hl_options,)),
        })

        self.discriminator = discriminator
        self.ll_buffer_capacity = ll_buffer_capacity
        self.ll_buffer = deque(maxlen=ll_buffer_capacity)
    
    @staticmethod
    def _hl_observation(obs, mask):
        return {
            'obs': obs,
            'mask': mask,
        }
    
    def reset(self):
        self.done = False
        self.obs = self.env.reset()
        self.m = available_actions(self.env, self.options)
        return self._hl_observation(self.obs, self.m)
    
    def _ll_step(self, action):
        return self.env.step(action)
    
    def step(self, action):
        assert self.m[action]
        assert not self.done

        plan = list(map(float, generate_plan(self.env, action, self.options)))
        reward = 0
        steps = 0

        while not self.done and plan and \
            (feasible(self.env, safety_plan(self.env, plan)) or self.m.sum() == 1):
            
            a, plan = plan[0], plan[1:]
            a = self.env._normalize(a)

            next_obs, _, self.done, info = self._ll_step(a)

            reward += self.discount**steps * self.discriminator(self.obs, a, next_obs, self.done)

            self.ll_buffer.append({
                'obs': self.obs,
                'next_obs': next_obs,
                'acts': np.array((a,)),
                'dones': np.array(self.done),
            })

            steps += 1
            self.obs = next_obs

        self.m = available_actions(self.env, self.options)

        return self._hl_observation(self.obs, self.m), reward, self.done, info
    
    def sample_ll(self, n):
        assert n <= self.ll_buffer_capacity, f'Sample size of {n} exceeds buffer capacity of {self.ll_buffer_capacity}'
        assert n <= len(self.ll_buffer), f'Sample size of {n} exceeds buffer size of {len(self.ll_buffer)}'
        return list(itertools.islice(self.ll_buffer, n))

class RenderOptions(OptionsEnv):

    def __init__(self, options, *args, **kwargs):
        super().__init__(options, discriminator=lambda s, a, n, d: 0, ll_buffer_capacity=0, *args, **kwargs)

    def _ll_step(self):
        out = super()._ll_step()
        self.env.render()
        return out
    
    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

def safety_plan(env, plan):
    return np.concatenate((plan, np.array(5 * [env._env._min_acc])), axis=0)

def available_actions(env, options):
    """Return mask of available actions given current `env` state.
    Action 0 is considered safe fallback.
    """
    plans = [generate_plan(env, i, options) for i, _ in enumerate(options)]
    # is emergency braking still possible?
    plans = list(map(lambda p: safety_plan(env, p), plans))

    T = max(len(p) for p in plans)
    plans = [np.pad(p, ((0, T-len(p)),), constant_values=np.nan) for p in plans]
    plans = np.stack(plans, axis=0)
    
    valid = feasible(env, plans)
    if not valid.any():
        valid[0] = True

    return valid

def target_velocity_plan(current_v: float, target_v: float, t: int, dt: float):
    """Smoothly target a velocity in a given number of steps"""
    # for now, constant acceleration
    a = (target_v - current_v)  / (t * dt)
    return a*np.ones((t,))

def generate_plan(env, i, options):
    """Generate input profile for high-level action `i`."""
    assert i < len(options), "Invalid option index {i}"
    target_v, t = options[i]
    current_v = env._env.state[env._agent, 1].item() # extract from env
    plan = target_velocity_plan(current_v, target_v, t, env._env._dt)
    assert len(plan) == t, "incorrect plan length"
    return plan
