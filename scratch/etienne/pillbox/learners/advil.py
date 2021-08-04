import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete
import gym
from stable_baselines3.common.preprocessing import get_action_dim
from tqdm import tqdm
from torch.autograd import Variable
from itertools import repeat
from torch.autograd import grad as torch_grad
from typing import List, Type
import types

# Infinite dataloader
def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def create_mlp(
    input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU) -> List[nn.Module]:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    return modules

def init_ortho(layer):
    if type(layer) == nn.Linear:
        nn.init.orthogonal_(layer.weight)

        
class AdVILPolicy(nn.Module):
    def __init__(self, env, mean=None, std=None):
        super(AdVILPolicy, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
            self.discrete = True
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
            self.low = torch.as_tensor(env.action_space.low)
            self.high = torch.as_tensor(env.action_space.high)
            self.discrete = False
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = env.observation_space
        net = create_mlp(self.obs_dim, self.action_dim, self.net_arch, nn.ReLU)
        if self.discrete:
            net.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
            self.is_normalized = True
        else:
            self.is_normalized = False
    def forward(self, obs):
        action = self.net(obs)
        return action
    def predict(self, obs, state, mask, deterministic):
        obs = obs.reshape((-1,) + (self.obs_dim,))
        if self.is_normalized:
            obs = (obs - self.mean) / self.std
        obs = torch.as_tensor(obs)
        with torch.no_grad():
            actions = self.forward(obs)
            if self.discrete:
                actions = actions.argmax(dim=1).reshape(-1)
            else:
                actions = self.low + ((actions + 1.0) / 2.0) * (self.high - self.low)
                actions = torch.max(torch.min(actions, self.high), self.low)
            actions = actions.cpu().numpy()
        return actions, state


class AdVILDiscriminator(nn.Module):
    def __init__(self, env):
        super(AdVILDiscriminator, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        net = create_mlp(self.obs_dim + self.action_dim, 1, self.net_arch, nn.ReLU)
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)

    def forward(self, inputs):
        output = self.net(inputs)
        return output.view(-1)

def pi_update(obs, acts, pi, f, pi_opt, prog):
    pi_opt.zero_grad()
    obs_v = Variable(obs)
    pi_acts = pi(obs_v)
    #learner_sa = torch.cat((obs, pi_acts), axis=1)
    f_learner = f(obs, acts)
    pi_loss = f_learner.mean() + orthogonal_reg(pi) + 2e-1 * (pi_acts - acts).square().mean()
    pi_loss.backward()
    if prog > 0.1:
        torch.nn.utils.clip_grad_norm(pi.parameters(), 40.0)
    pi_opt.step()
    return pi_loss.item(), (2e-1 * (pi_acts - acts).square().mean()).item()

def orthogonal_reg(pi):
    with torch.enable_grad():
        reg = 1e-4
        orth_loss = torch.zeros(1)
        for name, param in pi.named_parameters():
            if 'bias' not in name:
                x = torch.mm(torch.t(param), param)
                x = x * (1. - torch.eye(param.shape[-1]))
                orth_loss = orth_loss + reg * (x.square().sum())
        return orth_loss

def f_update(obs, acts, pi, f, f_opt, prog):
    obs_v = Variable(obs)
    pi_acts = pi(obs_v)
    #learner_sa = torch.cat((obs, pi_acts), axis=1)
    #expert_sa = Variable(torch.cat((obs, acts), axis=1))
    f_learner = f(obs, pi_acts)
    f_expert = f(obs, acts)
    #gp = gradient_penalty((obs, pi_acts), (obs, acts), f)
    f_opt.zero_grad()
    f_loss = f_expert.mean() - f_learner.mean()# + 10 * gp
    f_loss.backward()
    if prog > 0.1:
        torch.nn.utils.clip_grad_norm(f.parameters(), 40.0)
    f_opt.step()
    return f_loss.item()

def gradient_penalty(learner_sa, expert_sa, f):
    batch_size = expert_sa[0].size()[0]

    #alpha = torch.rand(batch_size, 1)
    #alpha = alpha.expand_as(expert_sa)

    salpha = torch.rand(batch_size, 1, 1)
    salpha = salpha.expand_as(expert_sa[0])

    aalpha = torch.rand(batch_size, 1)
    aalpha = aalpha.expand_as(expert_sa[1])

    #interpolated = alpha * expert_sa.data + (1 - alpha) * learner_sa.data
    #interpolated = Variable(interpolated, requires_grad=True)
    #f_interpolated = f(interpolated.float())

    sinterpolated = salpha * expert_sa[0].data + (1 - salpha) * learner_sa[0].data
    sinterpolated = Variable(sinterpolated, requires_grad=True)

    ainterpolated = aalpha * expert_sa[1].data + (1 - aalpha) * learner_sa[1].data
    ainterpolated = Variable(ainterpolated, requires_grad=True)
    
    f_interpolated = f(sinterpolated, ainterpolated)

    #gradients = torch_grad(outputs=f_interpolated, inputs=interpolated,
    #                       grad_outputs=torch.ones(f_interpolated.size()),
    #                       create_graph=True, retain_graph=True)[0]
    
    sgradients = torch_grad(outputs=f_interpolated, inputs=sinterpolated,
                           grad_outputs=torch.ones(f_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    agradients = torch_grad(outputs=f_interpolated, inputs=ainterpolated,
                           grad_outputs=torch.ones(f_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    #gradients = gradients.view(batch_size, -1)
    sgradients = sgradients.view(batch_size, -1)
    agradients = agradients.view(batch_size, -1)
    #norm = gradients.norm(2, dim=1).mean().item()
    #gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradients_norm = torch.sqrt(torch.sum(sgradients ** 2, dim=1) + torch.sum(agradients ** 2, dim=1) + 1e-12)
    # 2 * |f'(x_0)|
    return ((gradients_norm - 0.4) ** 2).mean()

def advil_training(data_loader, env, iters=int(1e5), policy_class=AdVILPolicy, discriminator_class=AdVILDiscriminator, lr_pi=8e-6, lr_f=8e-4):
    if not isinstance(env.action_space, Discrete):
        low = torch.as_tensor(env.action_space.low)
        high = torch.as_tensor(env.action_space.high)
    if data_loader.dataset.is_normalized:
        pi = policy_class(env, data_loader.dataset.mean, data_loader.dataset.std)
    else:
        pi = policy_class(env)
    f = discriminator_class(env)
    pi_opt = optim.Adam(pi.parameters(), lr=lr_pi)

    last_loss = 0
    f_opt = optim.Adam(f.parameters(), lr=lr_f)
    data_loader = repeater(data_loader)
    for t in tqdm(range(iters)):
        data = next(data_loader)
        obs = data['obs']
        acts = data['acts']
        #if isinstance(env.action_space, Discrete):
        #    acts = nn.functional.one_hot(acts, env.action_space.n)
        #else:
        #    acts = (((acts - low) / (high - low)) * 2.0) - 1.0
        pi_loss, mse_reg = pi_update(obs, acts, pi, f, pi_opt, t/iters)
        f_loss = f_update(obs, acts, pi, f, f_opt, t/iters)
        if t % 100 == 0:
            print("pi loss:", pi_loss)
            print("mse reg:", mse_reg)
            print("f loss:", f_loss)
    return pi
