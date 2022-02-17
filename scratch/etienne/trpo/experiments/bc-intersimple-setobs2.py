# %%
import torch
from core.policy import SetPolicy
from tqdm import tqdm

expert_data = torch.load('intersimple-expert-data-setobs2.pt')
states, actions, _, dones = expert_data

policy = SetPolicy(actions.shape[-1])

policy = policy.cuda()
optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
states = states[~dones].cuda()
actions = actions[~dones].cuda()

for _ in tqdm(range(10000)):
    optim.zero_grad()
    loss = -policy.log_prob(policy(states), actions).mean()
    loss.backward()
    optim.step()

    print('Loss', loss)

torch.save(policy.state_dict(), 'bc-intersimple-setobs2.pt')

# %%
import numpy as np
from core.policy import SetPolicy
from util.wrappers import Setobs, TransformObservation, CollisionPenaltyWrapper
from intersim.envs import IntersimpleLidarFlatRandom
from intersim.envs.intersimple import speed_reward
import functools

policy = SetPolicy(actions.shape[-1])
policy.load_state_dict(torch.load('bc-intersimple-setobs2.pt'))

obs_min = np.array([
    [-1000, -1000, 0, -np.pi, -1e-1, 0.],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
]).reshape(-1)

obs_max = np.array([
    [1000, 1000, 20, np.pi, 1e-1, 0.],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
]).reshape(-1)

env = Setobs(
    TransformObservation(CollisionPenaltyWrapper(IntersimpleLidarFlatRandom(
        n_rays=5,
        reward=functools.partial(
            speed_reward,
            collision_penalty=0
        ),
        stop_on_collision=False,
    ), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
)

obs = env.reset()
env.render(mode='post')
for i in range(300):
    #action, _ = policy.predict(torch.tensor(obs))
    action = policy.sample(policy(torch.tensor(obs, dtype=torch.float32)))
    obs, reward, done, _ = env.step(action)
    env.render(mode='post')
    print('step', i, 'reward', reward)
    if done:
        break
env.close()

# %%
