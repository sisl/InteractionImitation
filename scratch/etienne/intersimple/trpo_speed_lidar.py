# %%
from sb3_contrib import TRPO
from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools

model_name = "trpo_speed_lidar"

#def reward(state, action, info):
#    speed = state[2].item()
#    r = speed if speed < 10 else (10 - 5 * (speed - 10))
#    return 0.1 * r

env = IntersimpleLidarFlat(
    n_rays=5,
    agent=51,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
)

# %%
model = TRPO(
    "MlpPolicy", env,
    learning_rate=1e-4,
    verbose=1,
    tensorboard_log='runs/',
    #use_sde=True,
    #sde_sample_freq=4,
)
model.learn(total_timesteps=1000000)
model.save(model_name)

print('Done training.')

del model # remove to demonstrate saving and loading

# %%
model = TRPO.load(model_name)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='post')
    if done:
        break

env.close(filestr='render/'+model_name)

# %%
