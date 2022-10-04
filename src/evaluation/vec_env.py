from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from copy import deepcopy
import numpy as np

class CallbackWhenDoneVecEnv(DummyVecEnv):
    """DummyVecEnv that calls `done_callback` before resetting the wrapped environment."""

    def __init__(self, env_fns, done_callback):
        assert len(env_fns) == 1 # for now
        super().__init__(env_fns)
        self.done_callback = done_callback

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs

                self.done_callback(deepcopy(self.buf_infos[env_idx]))
                
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
    
    def render(self, mode='post'):
        super().render(mode)
