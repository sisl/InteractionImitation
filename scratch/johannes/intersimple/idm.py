# %%
import torch
from src.baselines.rule_policies import IDMRulePolicy
from tqdm import tqdm

from intersim.envs import NRasterizedIncrementingAgent, NRasterizedRandomAgent, NRasterized
from intersim.envs.intersimple import speed_reward
import functools


env = NRasterizedIncrementingAgent(
    # agent = 4,
    reward=functools.partial(
        speed_reward,
        collision_penalty=1000
    ),
    stop_on_collision=True,
)
policy = IDMRulePolicy(env)

colliding_agents = []

for agent in range(151):
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
                collising_agents.append(agent)
                print("  Collision")
            break
    env.close(filestr='idm/agent_{}'.format(agent))

print(len(colliding_agents), "colliding_agents")
print(colliding_agents)
# %%
