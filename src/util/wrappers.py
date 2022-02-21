import numpy as np
import gym

class Wrapper(gym.Wrapper):
    def __getattr__(self, name):
        return getattr(self.env, name)

class TransformObservation(gym.wrappers.TransformObservation):
    def __getattr__(self, name):
        return getattr(self.env, name)

class OptionsTimeLimit(gym.wrappers.TimeLimit):
    def __getattr__(self, name):
        return getattr(self.env, name)

class CollisionPenaltyWrapper(Wrapper):

    def __init__(self, env, collision_distance, collision_penalty, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.penalty = collision_penalty
        self.distance = collision_distance
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = -self.penalty if (obs.reshape(-1, 6)[1:, 0] < self.distance).any() else reward

        self.env._rewards.pop()
        self.env._rewards.append(reward)

        return obs, reward, done, info

class Minobs(Wrapper):
    """ Meant to be used as wrapper around LidarObservation """

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        n_rays = int(self.observation_space.shape[0] / 6) - 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=((1 + n_rays) * 2,))
    
    def minobs(self, obs):
        """ ego v, psidot ; (for each ray,) rel. distance, rel. velocity in ego forward direction """
        obs = obs.reshape(-1, 6)
        obs = np.concatenate((obs[:1, [2, 4]], obs[1:, [0, 2]]), axis=0)
        return obs.reshape(-1)

    def reset(self):
        return self.minobs(super().reset())
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.minobs(obs), reward, done, info

class Setobs(Wrapper):
    """ Meant to be used as wrapper around LidarObservation """

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.n_rays = int(self.observation_space.shape[0] / 6) - 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_rays, 6))
    
    def obs(self, obs):
        obs = obs.reshape(-1, 6)

        ego = obs[:1, [2, 4]] # v, psidot
        ego = np.tile(ego, (self.n_rays, 1))
        
        other = obs[1:, [0, 1, 2]] # distance, angle, velocity component in ego forward direction
        other = np.stack((other[:, 0], np.cos(other[:, 1]), np.sin(other[:, 1]), other[:, 2]), axis=-1)
        
        obs = np.concatenate((ego, other), axis=-1) 
        return obs

    def reset(self):
        return self.obs(super().reset())
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.obs(obs), reward, done, info
