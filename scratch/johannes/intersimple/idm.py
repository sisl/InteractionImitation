# %%
import torch
from src.baselines.rule_policies import IDMRulePolicy
from tqdm import tqdm

# expert_data = torch.load('intersimple-expert-data-setobs2.pt')
# states, actions, _, dones = expert_data

# policy = SetPolicy(actions.shape[-1])

# policy = policy.cuda()
# optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
# states = states[~dones].cuda()
# actions = actions[~dones].cuda()

# for _ in tqdm(range(10000)):
#     optim.zero_grad()
#     loss = -policy.log_prob(policy(states), actions).mean()
#     loss.backward()
#     optim.step()

#     print('Loss', loss)

# torch.save(policy.state_dict(), 'bc-intersimple-setobs2.pt')

# %%
# import numpy as np
# from core.policy import SetPolicy
# from util.wrappers import Setobs, TransformObservation, CollisionPenaltyWrapper
from intersim.envs import NRasterizedIncrementingAgent, NRasterizedRandomAgent
from intersim.envs.intersimple import speed_reward
import functools


env = NRasterizedRandomAgent(
    # agent = 4,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
    stop_on_collision=False,
)
policy = IDMRulePolicy(env)

obs = env.reset()
env.render(mode='post')
for i in range(1000):
    action, _ = policy.predict(torch.tensor(obs))
    # action = policy.sample(policy(torch.tensor(obs, dtype=torch.float32)))
    obs, reward, done, _ = env.step(action)
    env.render(mode='post')
    print('step', i, 'reward', reward)
    if done:
        obs = env.reset()
env.close()

# %%
