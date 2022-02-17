import torch
import torch.nn.functional as F
from dataclasses import dataclass
from src.core.reparam_module import ReparamPolicy
from src.core.sampling import rollout
from src.core.trpo import trpo_step
from src.core.ppo import ppo_step
from tqdm import tqdm

class TerminalLogger:
    def add_scalar(self, key, scalar, i=None):
        if i is not None:
            print('Iteration', i, end=' ')
        print(key, scalar)

@dataclass
class Buffer:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

def roll_buffer(buffer, *args, **kwargs):
    return Buffer(
        torch.roll(buffer.states, *args, **kwargs),
        torch.roll(buffer.actions, *args, **kwargs),
        torch.roll(buffer.rewards, *args, **kwargs),
        torch.roll(buffer.dones, *args, **kwargs),
    )

def gail(env_fn, expert_data, discriminator, disc_opt, disc_iters, policy, value,
         v_opt, v_iters, epochs, rollout_episodes, rollout_steps, gamma,
         gae_lambda, delta, backtrack_coeff, backtrack_iters, cg_iters=10, cg_damping=0.1, wasserstein=False, wasserstein_c=None, logger=TerminalLogger()):

    policy(torch.zeros(env_fn(0).observation_space.shape))
    policy = ReparamPolicy(policy)

    logger.add_scalar('expert/mean_episode_length', (~expert_data.dones).sum() / expert_data.states.shape[0])
    logger.add_scalar('expert/mean_reward_per_episode', expert_data.rewards[~expert_data.dones].sum() / expert_data.states.shape[0])

    for epoch in tqdm(range(epochs)):
        generator_data = Buffer(*rollout(env_fn, policy, rollout_episodes, rollout_steps))

        logger.add_scalar('gen/mean_episode_length', (~generator_data.dones).sum() / generator_data.states.shape[0], epoch)
        logger.add_scalar('gen/mean_reward_per_episode', generator_data.rewards[~generator_data.dones].sum() / generator_data.states.shape[0], epoch)

        discriminator, loss = train_discriminator(expert_data, generator_data, discriminator, disc_opt, disc_iters, wasserstein, wasserstein_c)
        if wasserstein:
            generator_data.rewards = discriminator(generator_data.states, generator_data.actions)
        else:
            generator_data.rewards = -F.logsigmoid(discriminator(generator_data.states, generator_data.actions))
        logger.add_scalar('disc/final_loss', loss, epoch)
        logger.add_scalar('disc/mean_reward_per_episode', generator_data.rewards[~generator_data.dones].sum() / generator_data.states.shape[0], epoch)

        value, policy = trpo_step(value, policy, generator_data.states, generator_data.actions, generator_data.rewards, generator_data.dones, gamma, gae_lambda, delta, backtrack_coeff, backtrack_iters, v_opt, v_iters, cg_iters, cg_damping)
        expert_data = roll_buffer(expert_data, shifts=-3, dims=0)
    
    return value, policy

def gail_ppo(env_fn, expert_data, discriminator, disc_opt, disc_iters, policy, value,
         v_opt, v_iters, epochs, rollout_episodes, rollout_steps, gamma,
         gae_lambda, clip_ratio, pi_opt, pi_iters, target_kl=None, max_grad_norm=None, wasserstein=False, wasserstein_c=None, logger=TerminalLogger()):

    logger.add_scalar('expert/mean_episode_length', (~expert_data.dones).sum() / expert_data.states.shape[0])
    logger.add_scalar('expert/mean_reward_per_episode', expert_data.rewards[~expert_data.dones].sum() / expert_data.states.shape[0])

    for epoch in range(epochs):
        generator_data = Buffer(*rollout(env_fn, policy, rollout_episodes, rollout_steps))

        logger.add_scalar('gen/mean_episode_length', (~generator_data.dones).sum() / generator_data.states.shape[0], epoch)
        logger.add_scalar('gen/mean_reward_per_episode', generator_data.rewards[~generator_data.dones].sum() / generator_data.states.shape[0], epoch)

        discriminator, loss = train_discriminator(expert_data, generator_data, discriminator, disc_opt, disc_iters, wasserstein, wasserstein_c)
        if wasserstein:
            generator_data.rewards = discriminator(generator_data.states, generator_data.actions)
        else:
            generator_data.rewards = -F.logsigmoid(discriminator(generator_data.states, generator_data.actions))
        logger.add_scalar('disc/final_loss', loss, epoch)
        logger.add_scalar('disc/mean_reward_per_episode', generator_data.rewards[~generator_data.dones].sum() / generator_data.states.shape[0], epoch)

        value, policy = ppo_step(value, policy, generator_data.states, generator_data.actions, generator_data.rewards, generator_data.dones, clip_ratio, gamma, gae_lambda, pi_opt, pi_iters, v_opt, v_iters, target_kl, max_grad_norm)
        expert_data = roll_buffer(expert_data, shifts=-3, dims=0)
    
    return value, policy

def train_discriminator(expert_data, generator_data, discriminator, disc_opt, disc_iters, wasserstein, wasserstein_c=None):

    n_expert_samples = (~expert_data.dones).sum()
    n_generator_samples = (~generator_data.dones).sum()
    n_samples = torch.minimum(n_expert_samples, n_generator_samples)

    gen_states = generator_data.states[~generator_data.dones][:n_samples]
    gen_actions = generator_data.actions[~generator_data.dones][:n_samples]
    exp_states = expert_data.states[~expert_data.dones][:n_samples]
    exp_actions = expert_data.actions[~expert_data.dones][:n_samples]

    states = torch.cat((exp_states, gen_states), dim=0).detach()
    actions = torch.cat((exp_actions, gen_actions), dim=0).detach()
    labels = torch.cat((torch.zeros(n_samples), torch.ones(n_samples))).detach()

    # print('Batch augmentation on')
    # random_states = torch.rand_like(gen_states)
    # random_actions = torch.rand_like(gen_actions)
    # states = torch.cat((exp_states, gen_states, random_states), dim=0).detach()
    # actions = torch.cat((exp_actions, gen_actions, random_actions), dim=0).detach()
    # labels = torch.cat((torch.zeros(n_samples), torch.ones(n_samples), torch.ones(n_samples))).detach()

    for _ in range(disc_iters):
        disc_opt.zero_grad()
        pred = discriminator(states, actions)

        if wasserstein:
            loss = -(pred * (1 - labels) - pred * labels).mean()
        else:
            loss = F.binary_cross_entropy(torch.sigmoid(pred), labels)
        
        loss.backward()
        disc_opt.step()

        if wasserstein_c is not None:
            with torch.no_grad():
                for param in discriminator.parameters():
                    param.clamp_(-wasserstein_c, wasserstein_c)
    
    return discriminator, loss
