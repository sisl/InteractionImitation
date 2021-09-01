# %%
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from intersim.envs.intersimple import IntersimpleReward, speed_reward

model_name = "ppo_const"

env = IntersimpleReward(
    agent=51,
    #reward=speed_reward,
)

# %%
model = PPO(
    "MlpPolicy", env,
    verbose=1,
)
model.learn(total_timesteps=100000)
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
