# %%
from stable_baselines3 import PPO
from intersim.envs.intersimple import IntersimpleFlatRandomAgent, Reward, RewardVisualization, speed_reward
import functools

model_name = "ppo_speed_random"

class IntersimpleRewardRandom(RewardVisualization, Reward, IntersimpleFlatRandomAgent):
    """`IntersimpleFlatAgent` with rewards."""
    pass

env = IntersimpleRewardRandom(
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    )
)

# %%
model = PPO(
    "MlpPolicy", env,
    verbose=1,
    batch_size=2048,
)
model.learn(total_timesteps=2e5)
model.save(model_name)

print('Done training.')

del model # remove to demonstrate saving and loading

# %%
model = PPO.load(model_name)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='post')
    if done:
        break

env.close(filestr='render/'+model_name)
