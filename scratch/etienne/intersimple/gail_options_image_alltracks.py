# %%
import sys
sys.path.append('../../../')
from src.discriminator import CnnDiscriminator, CnnDiscriminatorFlatAction
from src.policies import OptionsCnnPolicy
from src.util import feasible
from src.data import load_experts

from imitation.algorithms import adversarial
from imitation.util import logger
import imitation.data.rollout as rollout

import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env

import torch
import torch.utils.data
import numpy as np
import itertools
import gym
import pickle
import tempfile
import pathlib
from tqdm import tqdm

from intersim.envs.intersimple import NRasterized, NRasterizedRandomAgent, NRasterizedIncrementingAgent

ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5, 10, 20]] # option 0 is safe fallback

class OptionsEnv(gym.Wrapper):
    """
    Wrap an intersimple environment with an options generator
    """
    def __init__(self, env, *args, **kwargs):
        """
        Initialize wrapped environment and set high-level action and observation spaces
        """
        super().__init__(env, *args, **kwargs)
        num_hl_options = len(ALL_OPTIONS)
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
        """
        yield transitions using a generator
        Args:
            generator (sb3.PPO)
        Yields:

        """
        self.done = True
        while True:
            self.episode_start = False
            if self.done:
                self.s = self.env.reset()
                self.done = False
                self.episode_start = True

            self.m = available_actions(self.env)
            self.ch, self.value, self.log_prob = generator.policy.predict({
                'obs': torch.tensor(self.s).unsqueeze(0).to(generator.policy.device),
                'mask': torch.tensor(self.m).unsqueeze(0).to(generator.policy.device),
            })
            self.plan = list(map(float, generate_plan(self.env, self.ch)))

            self._after_choice()

            assert not self.done
            assert self.plan
            #assert feasible(self.env, self.plan, self.ch)

            while not self.done and self.plan and feasible(self.env, self.plan, self.ch):
                self.a, self.plan = self.plan[0], self.plan[1:]
                self.a = self.env._normalize(self.a)
                self.nexts, _, self.done, _ = self.env.step(self.a)

                self._after_step()

                self.s = self.nexts
            
            yield from self._transitions()

class LLOptions(OptionsEnv):
    """Sample low-level (state, action) tuples for discriminator training."""
    
    def __init__(self, *args, **kwargs):
        """
        LLOption uses the true LL observations
        """
        super().__init__(*args, **kwargs)
        # overwrite observation space to just output obs directly
        self.observation_space = self.observation_space['obs']

    def _after_choice(self):
        """
        After each option choice, initialize/reset the transition buffer
        """
        self._transition_buffer = []

    def _after_step(self):
        """
        After each ll action, append s, s', a, done to transition buffer
        """
        self._transition_buffer.append({
            'obs': self.s,
            'next_obs': self.nexts,
            'acts': np.array((self.a,)),
            'dones': np.array(self.done),
        })

    def _transitions(self):
        """
        Yield from the transition buffer
        """
        yield from self._transition_buffer
    
    def sample_ll(self, policy):
        """
        Args:
            policy
        Returns:
            gen: iterable which samples low-level transitions from the environment
        """
        return self.sample(policy)

class HLOptions(OptionsEnv):
    """Sample high-level (state, action, reward) tuples for generator training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _after_choice(self):
        """
        After an option selection, initialize total reward and number of steps
        """
        self.obs = {'obs': np.copy(self.s), 'mask': np.copy(self.m)}
        self.r = 0
        self.steps = 0

    def _after_step(self):
        """
        After each low-level action, add the discounted discriminated reward score (given a discriminator)
        """
        self.r += self.discount**self.steps * self.discriminator.discrim_net.reward_train(
            state=torch.tensor(self.s).unsqueeze(0).to(self.discriminator.discrim_net.device()),
            action=torch.tensor([[self.a]]).to(self.discriminator.discrim_net.device()),
            next_state=torch.tensor(self.s).unsqueeze(0).to(self.discriminator.discrim_net.device()), # unused
            done=torch.tensor(self.done).unsqueeze(0).to(self.discriminator.discrim_net.device()), # unused
        )
        self.steps += 1
    
    def _transitions(self):
        """
        Yield a single dictionary per high-level selected action
        Fields:
            obs: high-level state and mask at selection
            action: chosen high-level action
            reward: accumulated option reward
            episode_start: whether the action was chosen at the episode start
            value: the value estimate from the starting state
            log_prob: the log_prob of the selected action from the starting state
            done: whether the episode has ended

        """
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
        """
        Args:
            policy
            discriminator: function with which to score rewards
        Returns:
            gen: iterable which samples high-level transitions from the environment
        """
        self.discriminator = discriminator
        return self.sample(policy)

class RenderOptions(LLOptions):

    def _after_step(self):
        """
        Render the environment after each low-level step
        """
        super()._after_step()
        self.env.render()
    
    def close(self, *args, **kwargs):
        """
        On 'close', close the environment
        """
        self.env.close(*args, **kwargs)

def available_actions(env):
    """Return mask of available actions given current `env` state."""
    valid = np.array([feasible(env, generate_plan(env, i), i) for i in range(len(ALL_OPTIONS))])
    return valid

def target_velocity_plan(current_v: float, target_v: float, t: int, dt: float):
    """Smoothly target a velocity in a given number of steps"""
    # for now, constant acceleration
    a = (target_v - current_v)  / (t * dt)
    return a*np.ones((t,))

def generate_plan(env, i):
    """Generate input profile for high-level action `i`.

    Args:
        env (gym.Env): current environment state
        i (int): high-level action `i`
    Returns:
        plan (np.array): length T array of acceleration values
    """
    assert i < len(ALL_OPTIONS), "Invalid option index {i}"
    target_v, t = ALL_OPTIONS[i]
    current_v = env._env.state[env._agent, 1].item() # extract from env
    plan = target_velocity_plan(current_v, target_v, t, env._env._dt)
    assert len(plan) == t, "incorrect plan length"
    return plan

def flatten_transitions(transitions):
    return {
        'obs': np.stack(list(t['obs'] for t in transitions), axis=0),
        'next_obs': np.stack(list(t['next_obs'] for t in transitions), axis=0),
        'acts': np.stack(list(t['acts'] for t in transitions), axis=0),
        'dones': np.stack(list(t['dones'] for t in transitions), axis=0),
    }

def train_discriminator(env, generator, discriminator, num_samples):
    transitions = list(itertools.islice(env.sample_ll(generator), num_samples))
    generator_samples = flatten_transitions(transitions)
    discriminator.train_disc(gen_samples=generator_samples)

def train_generator(env, generator, discriminator, num_samples):
    generator_samples = list(itertools.islice(env.sample_hl(generator, discriminator), num_samples+1))
    
    generator.rollout_buffer.reset()
    for s in generator_samples[:-1]:
        generator.rollout_buffer.add(
            obs=s['obs'],
            action=s['action'].cpu(),
            reward=s['reward'].cpu(),
            episode_start=s['episode_start'],
            value=s['value'],
            log_prob=s['log_prob'],
        )
    
    generator.rollout_buffer.compute_returns_and_advantage(
        last_values=generator_samples[-1]['value'],
        dones=generator_samples[-1]['done'],
    )
    
    generator.train()

def train(expert_data, env_class=NRasterizedRandomAgent, env_settings={}, epochs=10, discrim_batch_size=32, generator_steps=2048, discount=0.99):
    """
    Args: 
        expert_data: list of transitions
        env_class: environment class
        env_settings: environment settings
        epochs: number of epochs to train for
        discrim_batch_size: discriminator batch size
        generator_steps: number of steps taken in generator
        discount: discount factor
    Returns:
        generator (stable_baselines3.PPO): options policy
    """
    env = env_class(**env_settings)
    env.discount = discount

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    logger.configure(tempdir_path / "GAIL/")
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv = make_vec_env(env_class, n_envs=1, env_kwargs=env_settings)
    discriminator = adversarial.GAIL(
        expert_data=expert_data,
        expert_batch_size=discrim_batch_size,
        discrim_kwargs={'discrim_net': CnnDiscriminatorFlatAction(venv)},
        #discrim_kwargs={'discrim_net': CnnDiscriminator(venv)},
        venv=venv, # unused
        gen_algo=stable_baselines3.PPO("CnnPolicy", venv), # unused
    )

    generator = stable_baselines3.PPO(
        OptionsCnnPolicy,
        OptionsEnv(env),
        verbose=1,
        n_steps=generator_steps,
    )

    # PPO.train requires logger as set up in
    # PPO._setup_learn (called by PPO.learn)
    generator._logger = stable_baselines3.common.utils.configure_logger(
        generator.verbose,
        generator.tensorboard_log,
    )

    for _ in tqdm(range(epochs)):
        train_discriminator(LLOptions(env), generator, discriminator, num_samples=discrim_batch_size)
        train_generator(HLOptions(env), generator, discriminator, num_samples=generator_steps)
    
    return generator

# %%
if __name__ == '__main__':
    # %%
    model_name = 'gail_options_image'
    env_class = NRasterizedRandomAgent
    env_settings = {'width': 36, 'height': 36, 'm_per_px': 2}
    
    #env_class = NRasterized
    #env_settings = {'agent': 51, 'width': 36, 'height': 36, 'm_per_px': 2}
    files = ['../../../expert_data/DR_USA_Roundabout_FT/track%04i/expert.pkl'%(i) for i in range(5)]
    transitions=load_experts(files)

    generator = train(
        transitions,
        env_class=env_class,
        env_settings=env_settings,
        epochs=10, 
        discrim_batch_size=32, 
        generator_steps=2048, 
        discount=0.99
        )

    generator.save(model_name)

    # %%
    model = stable_baselines3.PPO.load(model_name)

    env = RenderOptions(NRasterizedRandomAgent(**env_args))

    for s in env.sample_ll(model):
        if s['dones']:
            break

    env.close(filestr='render/'+model_name)

# %% Tests

def test_ll_expert_data():
    with open("data/NormalizedIntersimpleExpertMu.001_NRasterizedAgent51w36h36mppx2.pkl", "rb") as f:
        expert_trajectories = pickle.load(f)
    expert_transitions = rollout.flatten_trajectories(expert_trajectories)

    env = LLOptions(NRasterized(agent=51, width=36, height=36, m_per_px=2))

    gen_transitions = list(itertools.islice(env.sample_ll(
        policy=stable_baselines3.PPO(
            OptionsCnnPolicy,
            OptionsEnv(env),
            verbose=1,
        )
    ), 10))
    gen_transitions = flatten_transitions(gen_transitions)

    assert expert_transitions[:10].obs.shape == gen_transitions['obs'].shape
    assert expert_transitions[:10].next_obs.shape == gen_transitions['next_obs'].shape
    assert expert_transitions[:10].acts.shape == gen_transitions['acts'].shape
    assert expert_transitions[:10].dones.shape == gen_transitions['dones'].shape

def test_ll_states():
    env = NRasterized()
    policy = stable_baselines3.PPO(
        OptionsCnnPolicy,
        OptionsEnv(env),
        verbose=1,
    )
    llenv = LLOptions(env)
    transitions = list(itertools.islice(llenv.sample_ll(policy=policy), 100))

    env2 = NRasterized()
    s2 = env2.reset()
    for i, t in enumerate(transitions):
        assert i == 0 or np.array_equal(t['obs'], transitions[i-1]['next_obs'])
        assert np.array_equal(t['obs'], s2)
        assert t['acts'].shape == (1,)

        nexts2, _, done2, _ = env2.step(t['acts'])
        assert np.array_equal(t['next_obs'], nexts2)
        assert np.array_equal(t['dones'], done2)

        if done2:
            break

        s2 = nexts2

def test_hl_transitions():
    pass
