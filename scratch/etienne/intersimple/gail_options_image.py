# %%
from gail.discriminator import CnnDiscriminator
from imitation.algorithms import adversarial
import stable_baselines3
import torch.utils.data
import numpy as np
from intersim.envs.intersimple import NRasterized
import itertools
from torch.distributions import Categorical
import gym
import torch
import pickle
import imitation.data.rollout as rollout
import tempfile
import pathlib
from imitation.util import logger
from stable_baselines3.common.env_util import make_vec_env

model_name = 'gail_options_image'
env_settings = {'agent': 51, 'width': 36, 'height': 36, 'm_per_px': 2}

ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5, 10, 20]] # option 0 is safe fallback

class OptionsCnnPolicy(stable_baselines3.common.policies.ActorCriticCnnPolicy):

    def __init__(self, observation_space, *args, **kwargs):
        super().__init__(observation_space['obs'], *args, **kwargs)

    def _prior_distribution(self, s):
        latent_pi, latent_vf, latent_sde = self._get_latent(s)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        values = self.value_net(latent_vf)
        return values, distribution.distribution

    def predict(self, obs):
        s, m = obs['obs'], obs['mask']
        values, prior = self._prior_distribution(s)
        posterior = Categorical(prior.probs * m)
        ch = posterior.sample()
        return ch, values, posterior.log_prob(ch)

    def evaluate_actions(self, obs, ch):
        s, m = obs['obs'], obs['mask']
        values, prior = self._prior_distribution(s)
        posterior = Categorical(prior.probs * m)
        return values, posterior.log_prob(ch), posterior.entropy() # additional values used by PPO.train

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
    """Generate input profile for high-level action `i`."""
    assert i < len(ALL_OPTIONS), "Invalid option index {i}"
    target_v, t = ALL_OPTIONS[i]
    current_v = env._env.state[env._agent, 1].item() # extract from env
    plan = target_velocity_plan(current_v, target_v, t, env._env._dt)
    assert len(plan) == t, "incorrect plan length"
    return plan

def check_future_collisions_fast(env, actions):
    """Checks whether `env._agent` would collide with other agents assuming `actions` as input.
    
    Vehicles are (over-)approximated by single circles.

    Args:
        env (gym.Env): current environment state
        actions (list of torch.Tensor): list of B (T, nv, adims) T-length action profiles
    Returns:
        feasible (torch.Tensor): tensor of shape (B,) indicating whether the respective action profiles are collision-free
    """
    B, (T, nv, _) = len(actions), actions[0].shape

    states = torch.stack(env._env.propagate_action_profile(actions), axis=0)
    assert states.shape == (B, T, nv, 5)

    distance = ((states[:, :, :, :2] - states[:, :, env._agent:env._agent+1, :2])**2).sum(-1).sqrt()
    distance = torch.where(distance.isnan(), np.inf*torch.ones_like(distance), distance) # only collide with spawned agents
    distance[:, :, env._agent] = np.inf # cannot collide with itself
    assert distance.shape == (B, T, nv)

    radius = (env._env._lengths**2 + env._env._widths**2).sqrt() / 2
    min_distance = radius[env._agent] + radius
    min_distance = min_distance.unsqueeze(0).unsqueeze(0)
    print('min_distance', min_distance.shape)
    assert min_distance.shape == (1, 1, nv)

    return (distance > min_distance).all(-1).all(-1)

def feasible(env, plan, ch):
    """Check if input profile is feasible given current `env` state. Action `ch=0` is safe fallback."""
    
    # zero pad plan - Take (T,) np plan and convert it to (T, nv, 1) torch.Tensor
    full_plan = torch.zeros(len(plan), env._env._nv, 1)
    full_plan[:, env._agent, 0] = torch.tensor(plan)
    valid = check_future_collisions_fast(env, [full_plan]) # check_future_collisions_fast takes in B-list and outputs (B,) bool tensor
    return ch == 0 or valid.item()

def sample_ll(env, generator):
    """Sample low-level (state, action) pairs for discriminator training."""
    done = True
    while True:
        if done:
            s = env.reset()
            done = False

        m = available_actions(env)
        ch, _, _ = generator.policy.predict({
            'obs': torch.tensor(s).unsqueeze(0).to(generator.policy.device),
            'mask': torch.tensor(m).unsqueeze(0).to(generator.policy.device),
        })
        plan = list(map(float, generate_plan(env, ch)))

        assert not done
        assert plan
        assert feasible(env, plan, ch), f'Infeasible hl action {ch}'

        while not done and plan and feasible(env, plan, ch):
            a, plan = env._normalize(plan[0]), plan[1:]
            nexts, _, done, _ = env.step(a)
            yield {
                'obs': s,
                'next_obs': nexts,
                'acts': np.array((a,)),
                'dones': np.array(done),
            }
            s = nexts

def sample_hl(env, generator, discriminator):
    """Sample high-level (state, action, reward) tuples for generator training."""
    done = True
    while True:
        episode_start = False
        if done:
            s = env.reset()
            m = available_actions(env)
            done = False
            episode_start = True

        obs = {'obs': s, 'mask': m}
        ch, value, log_prob = generator.policy.predict({
            'obs': torch.tensor(s).unsqueeze(0).to(generator.policy.device),
            'mask': torch.tensor(m).unsqueeze(0).to(generator.policy.device),
        })
        plan = list(map(float, generate_plan(env, ch)))
        r = 0
        steps = 0

        while not done and plan and feasible(env, plan, ch):
            a, plan = env._normalize(plan[0]), plan[1:]
            r += discriminator.discrim_net.discriminator(
                torch.tensor(s).unsqueeze(0).to(discriminator.discrim_net.device()),
                torch.tensor([[a]]).to(discriminator.discrim_net.device()),
            )
            steps += 1
            s, _, done, _ = env.step(a)
            m = available_actions(env)

        yield {
            'obs': obs,
            'action': ch,
            'reward': r.detach() / steps,
            'episode_start': episode_start,
            'value': value.detach(),
            'log_prob': log_prob.detach(),
            'done': done,
        }

def flatten_transitions(transitions):
    return {
        'obs': np.stack(list(t['obs'] for t in transitions), axis=0),
        'next_obs': np.stack(list(t['next_obs'] for t in transitions), axis=0),
        'acts': np.stack(list(t['acts'] for t in transitions), axis=0),
        'dones': np.stack(list(t['dones'] for t in transitions), axis=0),
    }

def train_discriminator(env, generator, discriminator, num_samples):
    transitions = list(itertools.islice(sample_ll(env, generator), num_samples))
    generator_samples = flatten_transitions(transitions)
    discriminator.train_disc(gen_samples=generator_samples)

def train_generator(env, generator, discriminator, num_samples):
    generator_samples = list(itertools.islice(sample_hl(env, generator, discriminator), num_samples+1))
    
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

class OptionsEnv(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        num_hl_options = len(ALL_OPTIONS)
        self.action_space = gym.spaces.Discrete(num_hl_options)
        self.observation_space = gym.spaces.Dict({
            'obs': env.observation_space,
            'mask': gym.spaces.Box(low=0, high=1, shape=(num_hl_options,)),
        })

def train(expert_data, epochs=10, expert_batch_size=32, generator_steps=2048):
    env = NRasterized(**env_settings)

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    logger.configure(tempdir_path / "GAIL/")
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv = make_vec_env(NRasterized, n_envs=1, env_kwargs=env_settings)
    discriminator = adversarial.GAIL(
        expert_data=expert_data,
        expert_batch_size=expert_batch_size,
        discrim_kwargs={'discrim_net': CnnDiscriminator(venv)},
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

    for _ in range(epochs):
        train_discriminator(env, generator, discriminator, num_samples=expert_batch_size)
        train_generator(env, generator, discriminator, num_samples=generator_steps)
    
    return generator

# %%
if __name__ == '__main__':
    # %%
    with open("data/NormalizedIntersimpleExpertMu.001_NRasterizedAgent51w36h36mppx2.pkl", "rb") as f:
        trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(trajectories)
    generator = train(transitions, generator_steps=200)

    generator.save(model_name)

    # %%
    model = stable_baselines3.PPO.load(model_name)

    env = NRasterized(**env_settings)

    s = env.reset()
    done = False
    env.render()
    while not done:
        m = available_actions(env)
        ch, _, _ = generator.policy.predict({
            'obs': torch.tensor(s).unsqueeze(0).to(generator.policy.device),
            'mask': torch.tensor(m).unsqueeze(0).to(generator.policy.device),
        })
        plan = list(map(float, generate_plan(env, ch)))

        while not done and plan and feasible(env, plan, ch):
            a, plan = env._normalize(plan[0]), plan[1:]
            s, _, done, _ = env.step(a)
            env.render()

    env.close(filestr='render/'+model_name)

# %% Tests

def test_ll_transitions_vs_expert_data():
    with open("data/NormalizedIntersimpleExpertMu.001_NRasterizedAgent51w36h36mppx2.pkl", "rb") as f:
        expert_trajectories = pickle.load(f)
    expert_transitions = rollout.flatten_trajectories(expert_trajectories)

    env = NRasterized(agent=51, width=36, height=36, m_per_px=2)

    gen_transitions = list(itertools.islice(sample_ll(
        env=NRasterized(**env_settings),
        generator=stable_baselines3.PPO(
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


def test_hl_transitions():
    pass
