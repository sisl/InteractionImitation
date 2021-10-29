import gym
import torch
from src.util.collisions import feasible
import numpy as np

class OptionsEnv(gym.Wrapper):
    
    def __init__(self, env, options=[(v,t) for v in [0,2,4,6,8] for t in [5]], *args, **kwargs):
        """option 0 is treated as safe fallback"""

        super().__init__(env, *args, **kwargs)
        self.options = options
        num_hl_options = len(self.options)
        self.action_space = gym.spaces.Discrete(num_hl_options)
        self.observation_space = gym.spaces.Dict({
            'obs': env.observation_space,
            'mask': gym.spaces.Box(low=0, high=1, shape=(num_hl_options,)),
        })

    def _after_choice(self):
        pass
    
    def _after_step(self):
        pass

    def _transitions(self):
        raise NotImplementedError('Use `LLOptions` or `HLOptions` for sampling.')
    
    def sample(self, generator):
        self.done = True
        while True:
            self.episode_start = False
            if self.done:
                self.s = self.env.reset()
                self.done = False
                self.episode_start = True

            self.m = available_actions(self.env, self.options)
            self.ch, self.value, self.log_prob = generator.policy.forward({
                'obs': torch.tensor(self.s).unsqueeze(0).to(generator.policy.device),
                'mask': torch.tensor(self.m).unsqueeze(0).to(generator.policy.device),
            })
            self.plan = list(map(float, generate_plan(self.env, self.ch, self.options)))

            self._after_choice()

            assert not self.done
            assert self.plan
            #assert feasible(self.env, self.plan, self.ch)

            while not self.done and self.plan and feasible(self.env, self.plan, self.ch.to('cpu')):
                self.a, self.plan = self.plan[0], self.plan[1:]
                self.a = self.env._normalize(self.a)
                self.nexts, _, self.done, _ = self.env.step(self.a)

                self._after_step()

                self.s = self.nexts
            
            yield from self._transitions()

class LLOptions(OptionsEnv):
    """Sample low-level (state, action) tuples for discriminator training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space['obs']

    def _after_choice(self):
        self._transition_buffer = []

    def _after_step(self):
        self._transition_buffer.append({
            'obs': self.s,
            'next_obs': self.nexts,
            'acts': np.array((self.a,)),
            'dones': np.array(self.done),
        })

    def _transitions(self):
        yield from self._transition_buffer
    
    def sample_ll(self, policy):
        return self.sample(policy)

class HLOptions(OptionsEnv):
    """Sample high-level (state, action, reward) tuples for generator training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _after_choice(self):
        self.obs = {'obs': np.copy(self.s), 'mask': np.copy(self.m)}
        self.r = 0
        self.steps = 0

    def _after_step(self):
        self.r += self.discount**self.steps * self.discriminator.discrim_net.reward_train(
            state=torch.tensor(self.s).unsqueeze(0).to(self.discriminator.discrim_net.device()),
            action=torch.tensor([[self.a]]).to(self.discriminator.discrim_net.device()),
            next_state=torch.tensor(self.s).unsqueeze(0).to(self.discriminator.discrim_net.device()), # unused
            done=torch.tensor(self.done).unsqueeze(0).to(self.discriminator.discrim_net.device()), # unused
        )
        self.steps += 1
    
    def _transitions(self):
        yield {
            'obs': self.obs,
            'action': self.ch,
            'reward': self.r.detach(),
            'episode_start': self.episode_start,
            'value': self.value.detach(),
            'log_prob': self.log_prob.detach(),
            'done': self.done,
        }
    
    def sample_hl(self, policy, discriminator):
        self.discriminator = discriminator
        return self.sample(policy)

class RenderOptions(LLOptions):

    def _after_step(self):
        super()._after_step()
        self.env.render()
    
    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

def available_actions(env, options):
    """Return mask of available actions given current `env` state."""
    plan_indices = list(range(len(options)))
    plans = [generate_plan(env, i, options) for i in plan_indices]
    T = max(len(p) for p in plans)
    plans = [np.pad(p, ((0, T-len(p)),), constant_values=np.nan) for p in plans]
    plans = np.stack(plans, axis=0)
    valid = feasible(env, plans, plan_indices)
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
