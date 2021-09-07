from gail.discriminator import MlpDiscriminator
from imitation.algorithms import adversarial
import stable_baselines3
import torch.utils.data
import numpy as np
from intersim.envs.intersimple import Intersimple
import itertools
from torch.distributions import Categorical
import gym

class OptionsMlpPolicy:

    def __init__(self, *args, **kwargs):
        self._policy = stable_baselines3.common.policies.ActorCriticPolicy(
            *args, **kwargs
        )

    def _prior_distribution(self, s):
        latent_pi, _, latent_sde = self._policy._get_latent(s)
        distribution = self._policy._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.distribution

    def predict(self, obs):
        s, m = obs
        prior = self._prior_distribution(s)
        posterior = Categorical(prior.probs * m)
        ch = posterior.sample()
        return ch

    def evaluate_actions(self, obs, ch):
        s, m = obs
        values = self._policy.value_net(s)
        prior = self._prior_distribution(s)
        posterior = Categorical(prior.probs * m)
        return values, posterior.logprob(ch), posterior.entropy() # additional values used by PPO.train

def available_actions(env):
    """Return mask of available actions given current `env` state."""
    return np.ones((env.num_hl_actions,))

def generate_plan(env, i):
    """Generate input profile for high-level action `i`."""
    return np.zeros((env.num_hl_steps,))

def feasible(env, plan):
    """Check if input profile is feasible given current `env` state."""
    return True

def sample_ll(env, generator):
    """Sample low-level (state, action) pairs for discriminator training."""
    done = True
    while True:
        if done:
            s = env.reset()

        m = available_actions(env)
        ch = generator.policy.predict((s, m))
        plan = list(generate_plan(env, ch))

        while not done and plan and feasible(env, plan):
            a = plan.pop()
            yield (s, a)
            s, _, done, _ = env.step(a)

def train_discriminator(env, expert_data, generator, discriminator, generator_batch_size):
    expert_samples = next(expert_data)
    generator_samples = itertools.islice(sample_ll(env, generator), generator_batch_size)
    discriminator.train_disc(expert_samples, generator_samples)

def sample_hl(env, generator, discriminator):
    """Sample high-level (state, action, reward) tuples for generator training."""
    done = True
    while True:
        if done:
            s = env.reset()

        m = available_actions(env)
        obs = (s, m)
        ch = generator.policy.predict((s, m))
        plan = list(generate_plan(env, ch))
        r = 0
        discount = 1

        while not done and plan and feasible(env, plan):
            a = plan.pop()
            r += discount * discriminator.discrim_net(s, a)
            discount *= env.discount
            s, _, done, _ = env.step(a)
        
        yield (obs, ch, r)

def train_generator(env, generator, discriminator, generator_batch_size):
    generator_samples = itertools.islice(sample_hl(env, generator, discriminator), generator_batch_size)
    generator.rollout_buffer.reset()
    generator.rollout_buffer.add(generator_samples)
    generator.train()

class OptionsEnv(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(env.num_hl_options)

def train(expert_data, epochs=10, generator_batch_size=1024, expert_batch_size=1024, num_hl_options=10, num_hl_steps=10, discount=0.99):
    env = Intersimple()
    env.num_hl_options = num_hl_options
    env.num_hl_steps = num_hl_steps
    env.discount = discount

    discriminator = adversarial.GAIL(discrim_kwargs={'discrim_net': MlpDiscriminator()})
    generator = stable_baselines3.PPO(OptionsMlpPolicy, OptionsEnv(env))
    expert_data = torch.utils.data.DataLoader(expert_data, expert_batch_size)

    for _ in range(epochs):
        train_discriminator(env, expert_data, generator, discriminator, generator_batch_size)
        train_generator(env, generator, discriminator, generator_batch_size)
