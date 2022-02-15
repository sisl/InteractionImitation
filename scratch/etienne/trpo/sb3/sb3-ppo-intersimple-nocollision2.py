from stable_baselines3 import PPO
from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools
from gym import Wrapper

model_name = "ppo_speed_lidar_nocollision"

env = IntersimpleLidarFlat(
    n_rays=5,
    agent=51,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
    stop_on_collision=False,
)

class CollisionPenaltyWrapper(Wrapper):

    def __init__(self, env, collision_distance, collision_penalty, last_reward_weight, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.penalty = collision_penalty
        self.distance = collision_distance
        self.last_reward = -collision_penalty
        self.last_reward_weight = last_reward_weight
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = -self.penalty if (obs.reshape(-1, 6)[1:, 0] < self.distance).any() else reward
        reward = self.last_reward_weight * self.last_reward + (1 - self.last_reward_weight) * self.last_reward

        self.env._rewards.pop()
        self.env._rewards.append(reward)

        return obs, reward, done, info

env = CollisionPenaltyWrapper(env, collision_distance=6, collision_penalty=10, last_reward_weight=0.9)

model = PPO(
    "MlpPolicy", env,
    learning_rate=1e-4,
    verbose=1,
)
model.learn(total_timesteps=100000)
model.save(model_name)

model = PPO.load(model_name)
obs = env.reset()
env.render(mode='post')
for i in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render(mode='post')
    print('step', i, 'front distance', obs.reshape(-1, 6)[3, 0], 'reward', reward)
    if done:
        break
env.close()
