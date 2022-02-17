from intersim.envs import IntersimpleLidarFlat
from options import OptionsEnv
import gym
import numpy as np

def test_obs_shape():
    options = [(0, 5), (5, 5), (10, 5)]
    env = OptionsEnv(IntersimpleLidarFlat(n_rays=5), options)
    assert env.reset().shape == (36,)

def test_act_space():
    options = [(0, 5), (5, 5), (10, 5)]
    env = OptionsEnv(IntersimpleLidarFlat(n_rays=5), options)
    assert env.action_space == gym.spaces.Discrete(3)

def test_plan():
    options = [(0, 5), (5, 5), (10, 5)]
    env = OptionsEnv(IntersimpleLidarFlat(n_rays=5), options)
    env.reset()
    plan = env.plan(options[0])
    assert np.allclose(plan, -13.998268127441406 * np.ones((5,)))

def test_plan2():
    options = [(0, 5), (5, 5), (10, 5)]
    env = OptionsEnv(IntersimpleLidarFlat(n_rays=5), options)
    obs = env.reset()
    states, actions, rewards, dones, plan_done, infos, n_steps = env.execute_plan(obs, options[0])
    assert states.shape == (6, 36)
    assert rewards.shape == (6,)
    assert dones.shape == (6,)
    assert len(infos) == 5

def test_step():
    options = [(0, 5), (5, 5), (10, 5)]
    env = OptionsEnv(IntersimpleLidarFlat(n_rays=5), options)
    env.reset()
    obs, reward, done, _ = env.step(0)
    assert obs.shape == (36,)
    assert reward == 5.0
    assert done == False

def test_ll_step():
    options = [(0, 5), (5, 5), (10, 5)]
    env = OptionsEnv(IntersimpleLidarFlat(n_rays=5), options)
    env.reset()
    _, _, _, info = env.step(0)
    assert info['ll']['observations'].shape == (6, 36)
    assert info['ll']['actions'].shape == (6, 1)
    assert info['ll']['rewards'].shape == (6,)
    assert info['ll']['env_done'].shape == (6,)
    assert info['ll']['plan_done'].shape == (6,)
    assert info['ll']['plan_done'][5] == True
    assert info['ll']['steps'] == 5
    assert len(info['ll']['infos']) == 5
