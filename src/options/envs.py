import gym
import numpy as np
from src.gail2.wrappers import Wrapper, Setobs, TransformObservation
from intersim.envs import IntersimpleLidarFlatIncrementingAgent

obs_min = np.array([
    [-1000, -1000, 0, -np.pi, -1e-1, 0.],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
]).reshape(-1)

obs_max = np.array([
    [1000, 1000, 20, np.pi, 1e-1, 0.],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
]).reshape(-1)

def NormalizedOptionsEvalEnv(**kwargs):
    return OptionsEnv(Setobs(
        TransformObservation(IntersimpleLidarFlatIncrementingAgent(
            n_rays=5,
            **kwargs,
        ), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
    ), options=[(0, 5), (1, 5), (2, 5), (4, 5), (6, 5), (8, 5)])

def NormalizedContinuousEvalEnv(**kwargs):
    return Setobs(
        TransformObservation(IntersimpleLidarFlatIncrementingAgent(
            n_rays=5,
            **kwargs,
        ), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
    )

class OptionsEnv(Wrapper):

    def __init__(self, env, options):
        super().__init__(env)
        self.ll_action_space = env.action_space
        self.options = options
        self.action_space = gym.spaces.Discrete(len(options))
        self.max_plan_length = max(t for _, t in options)
    
    def plan(self, option):
        target_v, t = option
        current_v = self.env._env.state[self.env._agent, 1].item()
        dt = self.env._env._dt
        a = (target_v - current_v) / (t * dt)
        a = self.env._normalize(a)
        a = a * np.ones((t,))
        a += 0.01 * np.random.randn(*a.shape)
        a = np.clip(a, self.ll_action_space.low, self.ll_action_space.high)
        return a

    def execute_plan(self, obs, option, render_mode=None):
        observations = np.zeros((self.max_plan_length + 1, *self.env.observation_space.shape))
        actions = np.zeros((self.max_plan_length + 1, *self.ll_action_space.shape))
        rewards = np.zeros((self.max_plan_length + 1,))
        env_done = np.ones((self.max_plan_length + 1,), dtype=bool)
        plan_done = np.ones((self.max_plan_length + 1,), dtype=bool)
        infos = []

        observations[0] = obs
        env_done[0] = False
        for k, u in enumerate(self.plan(option)):
            plan_done[k] = False
            o, r, d, i = super().step(u)
            actions[k] = u
            rewards[k] = r
            env_done[k] = d
            infos.append(i)
            observations[k+1] = o

            if render_mode is not None:
                self.env.render(render_mode)

            if d:
                break
        
        n_steps = k + 1
        return observations, actions, rewards, env_done, plan_done, infos, n_steps

    def step(self, action, render_mode=None):
        a = int(action)
        assert a == action
        ll_obs, ll_actions, ll_rewards, ll_env_done, ll_plan_done, ll_infos, ll_steps = self.execute_plan(self.last_obs, self.options[a], render_mode)
        hl_obs = ll_obs[ll_steps]
        hl_reward = (ll_rewards * ~ll_plan_done).sum().item()
        hl_done = ll_env_done[ll_steps-1].item()
        hl_infos = {
            'll': {
                'observations': ll_obs,
                'actions': ll_actions,
                'rewards': ll_rewards,
                'env_done': ll_env_done,
                'plan_done': ll_plan_done,
                'infos': ll_infos,
                'steps': ll_steps,
            }
        }
        self.last_obs = hl_obs
        return hl_obs, hl_reward, hl_done, hl_infos
    
    def reset(self, *args, **kwargs):
        self.last_obs = super().reset(*args, **kwargs)
        return self.last_obs
