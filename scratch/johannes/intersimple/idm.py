# %%
import torch
from src.baselines.rule_policies import IDMRulePolicy
from tqdm import tqdm

from intersim.envs import NRasterizedIncrementingAgent, NRasterizedRandomAgent, NRasterized,IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools


env = IntersimpleLidarFlat(
    agent = 51,
    n_rays=5,
    reward=functools.partial(
        speed_reward,
        collision_penalty=1000
    ),
    stop_on_collision=True,
)
policy = IDMRulePolicy(env)

colliding_agents = []

# for agent in range(151):
agent = env._agent
print("Start agent", agent)
obs = env.reset()
env.render(mode='post')
for i in range(300):
    action, _ = policy.predict(torch.tensor(obs))
    # action = policy.sample(policy(torch.tensor(obs, dtype=torch.float32)))
    obs, reward, done, _ = env.step(action)
    env.render(mode='post')
    # print('step', i, 'reward', reward)
    if done:
        if reward < -500:
            colliding_agents.append(agent)
            print("  Collision")
        break
env.close(filestr='idm3/agent_{}'.format(agent))

print(len(colliding_agents), "colliding_agents")
print(colliding_agents)
# %%
