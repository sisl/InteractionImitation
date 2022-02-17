import torch
from src.core.value_estimation import gae
from src.core.optimization import conjugate_gradient, line_search

def trpo_step(value, policy, states, safe_actions, actions, rewards, dones, gamma, gae_lambda, delta, backtrack_coeff, backtrack_iters, v_opt, v_iters, cg_iters=10, cg_damping=0.1):

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

    # compute policy gradient

    plogprob = policy.log_prob(policy(states, safe_actions), actions)
    surrogate_advantage = (plogprob * advantages)[valid].sum() / states.shape[0]
    g = torch.cat(torch.autograd.grad(surrogate_advantage, policy.flat_param)).detach()
    
    def Hx(x):
        kl = policy.kl_divergence(policy(states, safe_actions), policy(states, safe_actions).detach())[valid].mean()
        dKL = torch.cat(torch.autograd.grad(kl, policy.flat_param, create_graph=True))
        H_x = torch.cat(torch.autograd.grad(dKL.T @ x, policy.flat_param)).detach()
        return H_x + cg_damping * x

    x = conjugate_gradient(Hx, g, cg_iters)
    npg = torch.sqrt(2 * delta / (x.T @ Hx(x))) * x

    # perform line search

    def L(theta):
        rplogprob = policy.log_prob(policy(states, safe_actions, flat_param=theta), actions)
        return ((rplogprob - plogprob.detach()).exp() * advantages)[valid].sum() / advantages.shape[0]

    condition = lambda theta: policy.kl_divergence(policy(states, safe_actions, flat_param=theta), policy(states, safe_actions))[valid].mean() < delta

    x0 = policy.flat_param
    g0 = torch.cat(torch.autograd.grad(L(x0), x0))
    theta = line_search(L, x0, npg, g0, backtrack_coeff, condition, max_steps=backtrack_iters)

    # update policy parameters
    
    with torch.no_grad():
        policy.flat_param.copy_(theta)

    return value, policy

def ppo_step(value, policy, states, safe_actions, actions, rewards, dones, clip_ratio, gamma, gae_lambda, pi_opt, pi_iters, v_opt, v_iters, target_kl, max_grad_norm):

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

    old_dist = policy(states, safe_actions).detach()
    old_logprob = policy.log_prob(old_dist, actions).detach()

    def g(advantages, clip_ratio):
        return torch.where(advantages >= 0, (1 + clip_ratio) * advantages, (1 - clip_ratio) * advantages)

    def L(states, actions, advantages, clip_ratio):
        return torch.minimum(
            (policy.log_prob(policy(states, safe_actions), actions) - old_logprob).exp() * advantages,
            g(advantages, clip_ratio)
        )[valid].mean()

    for _ in range(pi_iters):
        pi_opt.zero_grad()
        ppo_loss = -L(states, actions, advantages, clip_ratio)
        ppo_loss.backward()

        if max_grad_norm:
            torch.nn.utils.clip_grad_norm(policy.parameters(), max_grad_norm)
        
        pi_opt.step()

        kl = policy.kl_divergence(policy(states, safe_actions), old_dist)[valid].mean()
        if target_kl and kl > target_kl:
            break
    
    print('KL', kl.item())

    return value, policy
