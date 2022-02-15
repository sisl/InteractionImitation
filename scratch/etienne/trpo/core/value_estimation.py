from operator import index
import torch

def gae(states, rewards, values, dones, gamma, gae_lambda):
    assert rewards.shape == values.shape == dones.shape
    n_episodes, n_steps = rewards.shape

    valid = ~dones
    valid[..., -1] = False

    td = rewards + gamma * torch.roll(values, shifts=-1, dims=1) - values
    adv = td.repeat(n_steps, 1, 1).transpose(0, 1)
    assert adv.shape == (n_episodes, n_steps, n_steps)

    step_start, step = torch.meshgrid(torch.arange(n_steps), torch.arange(n_steps), indexing='ij')
    past = step < step_start

    # add up discounted temporal differences 
    discount = torch.minimum(torch.tensor(gamma).log() * (step - step_start), torch.tensor(0.)).exp()
    discount = discount * ~past
    discount = discount * valid.unsqueeze(1)

    adv = adv * discount
    adv = adv.cumsum(2) # eq. (14)
    assert adv.shape == (n_episodes, n_steps, n_steps)

    # add up discounted k-advantages
    lambda_discount = torch.minimum(torch.tensor(gae_lambda).log() * (step - step_start), torch.tensor(0.)).exp()
    lambda_discount = lambda_discount * ~past
    lambda_discount = lambda_discount * valid.unsqueeze(1)

    adv = adv * lambda_discount
    adv = adv.sum(2) / (lambda_discount.sum(2) + 1e-10) # eq. (16)

    adv = (adv - adv[valid].mean()) / adv[valid].std()
    assert adv.shape == rewards.shape == values.shape

    returns = adv + values

    return adv, returns, valid
