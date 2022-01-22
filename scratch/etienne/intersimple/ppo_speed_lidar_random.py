# %%
from stable_baselines3 import PPO
from intersim.envs import IntersimpleLidarFlatRandom
from intersim.envs.intersimple import speed_reward
import functools

model_name = "ppo_speed_lidar_random"

#def reward(state, action, info):
#    speed = state[2].item()
#    r = speed if speed < 10 else (10 - 5 * (speed - 10))
#    return 0.1 * r

env = IntersimpleLidarFlatRandom(
    n_rays=5,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
)

# %%
model = PPO(
    "MlpPolicy", env,
    learning_rate=1e-4,
    verbose=1,
    tensorboard_log='runs/'
)
model.learn(total_timesteps=1000000)
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

# %%
