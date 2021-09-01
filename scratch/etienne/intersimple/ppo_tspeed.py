# %%
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from intersim.envs.intersimple import IntersimpleTargetSpeedAgent

model_name = "ppo_tspeed"

env = IntersimpleTargetSpeedAgent(
    agent=51,
    target_speed=10,
    speed_penalty_weight=0.001,
    collision_penalty=1000
)

# %%
model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-6,
    verbose=1,
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
