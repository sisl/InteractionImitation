import torch
from core.sampling import rollout
from core.value_estimation import gae

def ppo(env_fn, value, policy, epochs, rollout_episodes, rollout_steps, gamma, gae_lambda, clip_ratio, pi_opt, pi_iters, v_opt, v_iters, target_kl=None, max_grad_norm=None):

    for epoch in range(epochs):
        policy.eval()
        states, actions, rewards, dones = rollout(env_fn, policy, rollout_episodes, rollout_steps)

        print('mean', states[~dones].mean(0))
        print('std', states[~dones].std(0))

        print(f'Iteration {epoch} mean episode length {(~dones).sum() / states.shape[0]}')
        print(f'Iteration {epoch} mean reward per episode {rewards[~dones].sum() / states.shape[0]}')

        policy.train()
        value.train()
        value, policy = ppo_step(value, policy, states, actions, rewards, dones, clip_ratio, gamma, gae_lambda, pi_opt, pi_iters, v_opt, v_iters, target_kl, max_grad_norm)

    return value, policy

def ppo_step(value, policy, states, actions, rewards, dones, clip_ratio, gamma, gae_lambda, pi_opt, pi_iters, v_opt, v_iters, target_kl, max_grad_norm):

    states = states.detach()
    actions = actions.detach()
    rewards = rewards.detach()
    dones = dones.detach()

    advantages, returns, valid = gae(states, rewards, value(states), dones, gamma, gae_lambda)
    advantages = advantages.detach()
    returns = returns.detach()

    # update value function

    for _ in range(v_iters):
        v_opt.zero_grad()
        value_loss = (value(states) - returns).pow(2)[valid].mean()
        value_loss.backward()
        v_opt.step()

    # update policy

    old_dist = policy(states).detach()
    old_logprob = policy.log_prob(old_dist, actions).detach()

    def g(advantages, clip_ratio):
        return torch.where(advantages >= 0, (1 + clip_ratio) * advantages, (1 - clip_ratio) * advantages)

    def L(states, actions, advantages, clip_ratio):
        return torch.minimum(
            (policy.log_prob(policy(states), actions) - old_logprob).exp() * advantages,
            g(advantages, clip_ratio)
        )[valid].mean()

    for _ in range(pi_iters):
        pi_opt.zero_grad()
        ppo_loss = -L(states, actions, advantages, clip_ratio)
        ppo_loss.backward()

        if max_grad_norm:
            torch.nn.utils.clip_grad_norm(policy.parameters(), max_grad_norm)
        
        pi_opt.step()

        kl = policy.kl_divergence(policy(states), old_dist)[valid].mean()
        if target_kl and kl > target_kl:
            break
    
    print('KL', kl.item())

    return value, policy
