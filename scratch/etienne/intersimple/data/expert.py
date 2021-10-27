from intersim.envs.intersimple import Intersimple
from stable_baselines3.common.policies import BasePolicy
import gym
import intersim.envs.intersimple
import imitation.data.rollout as rollout
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper

class IntersimExpert(BasePolicy):
    
    def __init__(self, intersim_env, mu=0, *args, **kwargs):
        super().__init__(
            observation_space=gym.spaces.Space(),
            action_space=gym.spaces.Space(),
            *args, **kwargs
        )
        self._intersim = intersim_env
        self._mu = mu

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

    def _action(self):
        target_t = min(self._intersim._ind + 1, len(self._intersim._svt.simstate) - 1)
        target_state = self._intersim._svt.simstate[target_t]
        return self._intersim.target_state(target_state, mu=self._mu)

    def predict(self, *args, **kwargs):
        return self._action(), None

class IntersimpleExpert(BasePolicy):

    def __init__(self, intersimple_env, mu=0, *args, **kwargs):
        super().__init__(
            observation_space=intersimple_env.observation_space,
            action_space=intersimple_env.action_space,
            *args, **kwargs
        )
        self._intersimple = intersimple_env
        self._intersim_expert = IntersimExpert(intersimple_env._env, mu=mu)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

    def _action(self):
        # RandomLocation mixin re-initializes the intersim sub-env
        self._intersim_expert._intersim = self._intersimple._env
        return self._intersim_expert._action()[self._intersimple._agent]

    def predict(self, *args, **kwargs):
        return self._action(), None

class NormalizedIntersimpleExpert(IntersimpleExpert):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs):
        action, _ = super().predict(*args, **kwargs)
        return self._intersimple._normalize(action), None

class DummyVecEnvPolicy(BasePolicy):

    def __init__(self, experts):
        self._experts = [e() for e in experts]

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()
    
    def predict(self, *args, **kwargs):
        predictions = [e.predict() for e in self._experts]
        actions = [p[0] for p in predictions]
        states = [p[1] for p in predictions]
        return actions, states
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

def save_video(env, expert):
    env.reset()
    env.render()
    done = False
    while not done:
        actions, _ = expert.predict()
        _, _, done, _ = env.step(actions)
        env.render()
    env.close()

def demonstrations(expert='NormalizedIntersimpleExpert', env='NRasterizedRandomAgent', path=None, min_timesteps=25000, min_episodes=None, video=False, env_args={}, policy_args={}):
    """Rollout and save expert demos.
    
    Usage:
        python -m intersimple.expert <flags>

    """
    Env = intersim.envs.intersimple.__dict__[env]
    Expert = globals()[expert]

    env = Env(**env_args)
    info_env = RolloutInfoWrapper(env)
    venv = DummyVecEnv([lambda: info_env])

    policy = Expert(env, **policy_args)
    venv_policy = DummyVecEnvPolicy([lambda: policy])

    if video:
        save_video(env, policy)
    
    path = path or (policy.__class__.__name__ + '_' + env.__class__.__name__ + '.pkl')

    rollout.rollout_and_save(
        path=path,
        policy=venv_policy,
        venv=venv,
        sample_until=rollout.make_sample_until(
            min_timesteps=min_timesteps,
            min_episodes=min_episodes,
        )
    )

if __name__ == '__main__':
    import fire
    fire.Fire(demonstrations)
