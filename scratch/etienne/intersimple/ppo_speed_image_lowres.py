# %%
from stable_baselines3 import PPO
from intersim.envs.intersimple import NRasterized, speed_reward
import functools

model_name = "ppo_speed_image_lowres"

#def reward(state, action, info):
#    speed = state[2].item()
#    r = speed if speed < 10 else (10 - 5 * (speed - 10))
#    return 0.1 * r

env = NRasterized(
    agent=51,
    height=36,
    width=36,
    m_per_px=2,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
)

# %%
model = PPO(
    "CnnPolicy", env,
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

# %%
