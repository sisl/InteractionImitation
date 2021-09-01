from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from intersim.envs.intersimple import IntersimpleTargetSpeed

env = IntersimpleTargetSpeed()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_intersimple")

print('Done training.')

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_intersimple")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='post')
    if done:
        break

env.close()
