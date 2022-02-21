import torch
from src.core.reparam_module import ReparamPolicy
from src.core.sampling import rollout
from src.core.value_estimation import gae
from src.core.optimization import conjugate_gradient, line_search

def trpo(env_fn, value, policy, epochs, rollout_episodes, rollout_steps, gamma, gae_lambda, delta, backtrack_coeff, backtrack_iters, v_opt, v_iters, cg_iters=10, cg_damping=0.1):

    policy(torch.zeros(env_fn(0).observation_space.shape))
    policy = ReparamPolicy(policy)

    for epoch in range(epochs):
        policy.eval()
        states, actions, rewards, dones = rollout(env_fn, policy, rollout_episodes, rollout_steps)

        print('mean', states[~dones].mean(0))
        print('std', states[~dones].std(0))

        print(f'Iteration {epoch} mean episode length {(~dones).sum() / states.shape[0]}')
        print(f'Iteration {epoch} mean reward per episode {rewards[~dones].sum() / states.shape[0]}')

        policy.train()
        value.train()
        value, policy = trpo_step(value, policy, states, actions, rewards, dones, gamma, gae_lambda, delta, backtrack_coeff, backtrack_iters, v_opt, v_iters, cg_iters, cg_damping)

    return value, policy

def trpo_step(value, policy, states, actions, rewards, dones, gamma, gae_lambda, delta, backtrack_coeff, backtrack_iters, v_opt, v_iters, cg_iters=10, cg_damping=0.1):

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

    plogprob = policy.log_prob(policy(states), actions)
    surrogate_advantage = (plogprob * advantages)[valid].sum() / states.shape[0]
    g = torch.cat(torch.autograd.grad(surrogate_advantage, policy.flat_param)).detach()
    
    def Hx(x):
        kl = policy.kl_divergence(policy(states), policy(states).detach())[valid].mean()
        dKL = torch.cat(torch.autograd.grad(kl, policy.flat_param, create_graph=True))
        H_x = torch.cat(torch.autograd.grad(dKL.T @ x, policy.flat_param)).detach()
        return H_x + cg_damping * x

    x = conjugate_gradient(Hx, g, cg_iters)
    npg = torch.sqrt(2 * delta / (x.T @ Hx(x))) * x

    # perform line search

    def L(theta):
        rplogprob = policy.log_prob(policy(states, flat_param=theta), actions)
        return ((rplogprob - plogprob.detach()).exp() * advantages)[valid].sum() / advantages.shape[0]

    condition = lambda theta: policy.kl_divergence(policy(states, flat_param=theta), policy(states))[valid].mean() < delta

    x0 = policy.flat_param
    g0 = torch.cat(torch.autograd.grad(L(x0), x0))
    theta = line_search(L, x0, npg, g0, backtrack_coeff, condition, max_steps=backtrack_iters)

    # update policy parameters
    
    with torch.no_grad():
        policy.flat_param.copy_(theta)

    return value, policy
