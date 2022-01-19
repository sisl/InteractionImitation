# Borrowed heavily from https://github.com/HumanCompatibleAI/imitation/tree/master/src/imitation/data

import pickle
import numpy as np
import logging
import os
import pathlib
from typing import Optional, List, Dict

def generate_trajectories(
    policy,
    env,
    sample_until,
    rng: Optional[np.random.RandomState] = None, # np.random to shuffle
) -> List[dict]:
    """Generate trajectory dictionaries from a policy and an environment.

    Args:
      policy: a stable_baselines3 policy or algorithm trained on the gym environment
      env: The environment to interact with.
      sample_until: A function determining the termination condition.
          It takes a sequence of trajectories, and returns a bool.
          Most users will want to use one of `min_episodes` or `min_timesteps`.
      rng: used for shuffling trajectories.

    Returns:
      Sequence of trajectories, satisfying `sample_until`. 
    """
    trajectories = []
    while not sample_until(trajectories):
        
        # sample a trajectory
        ob, done = env.reset(), False
        ob_list, rew_list, act_list, info_list = [], [], [], []
        while not done:
            act, _ = policy.predict(ob)
            next_ob, rew, done, info = env.step(act) # ignore infos
            ob_list.append(ob)
            act_list.append(act)
            rew_list.append(rew)
            info_list.append(info)
            ob = next_ob
        ob_list.append(ob)

        traj = {
            'obs':np.stack(ob_list),
            'acts':np.stack(act_list),
            'rews':np.stack(rew_list),
            'infos': info_list,
            'terminal': True
        }
        trajectories.append(traj)
    
    # Shuffle trajectories
    if rng:
        rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory['acts'])
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + env.observation_space.shape
        real_obs = trajectory['obs'].shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + env.action_space.shape
        real_act = trajectory['acts'].shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory['rews'].shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories


def flatten_trajectories(trajectory_list: list) -> list:
    """
    Turn a list of trajectories into a (longer) list of transitions with appropriate fields

    Args:
        trajectory_list (list): list of trajectory dicts with keys:
            obs (np.ndarray): (T, *O) tensor of all observations in T-step trajectory
            acts (np.ndarray): (T-1, *A) tensor of all actions in T-step trajectory
            infos (list[dict]): (T-1)-length list of all information dictionaries
            terminal (bool): True if the trajectory ends at the last step
            rews (np.ndarray): (T-1, 1) tensor of rewards along trajectory
    Returns:
        transition_list (list): list of all transition dicts with keys:
            obs (np.ndarray): (*O) tensor of single-step observation
            acts (np.ndarray): (*A) tensor of single-step action
            infos (dict): single-step information dictionary
            next_obs (np.ndarray): (*O) tensor of next observation
            rews (np.ndarray): (1) tensor of single-step reward
            dones (bool): whether state is terminal
    """
    transition_list = []
    for traj in trajectory_list:
        T = traj['obs'].shape[0]
        if traj['infos']:
            infos = traj['infos'] 
        else: 
            infos = [{}] * T
        for i in range(T-1):
            transition_list.append({
               'obs': traj['obs'][i],
               'acts': traj['acts'][i],
               'next_obs': traj['obs'][i+1],
               'rews': traj['rews'][i],
               'dones': (i==T-2),
               'infos': infos[i],
            })
    return transition_list


def make_sample_until(min_timesteps: Optional[int]=None, min_episodes: Optional[int]=None):
    """Returns a termination condition sampling for a number of timesteps and episodes.
    
    Args:
        min_timesteps: Sampling will not stop until there are at least this many
            timesteps.
        min_episodes: Sampling will not stop until there are at least this many
            episodes.

    Returns:
        A termination condition which given a list of trajectories returns true if the condition is met.

    Raises:
        ValueError if neither of n_timesteps and n_episodes are set, or if either are
            non-positive.
    """
    if min_timesteps is None and min_episodes is None:
        raise ValueError(
            "At least one of min_timesteps and min_episodes needs to be non-None"
        )

    conditions = []
    if min_timesteps is not None:
        if min_timesteps <= 0:
            raise ValueError(
                f"min_timesteps={min_timesteps} if provided must be positive"
            )

        def timestep_cond(trajectories):
            if len(trajectories) == 0:
                return False
            timesteps = sum(len(t['obs']) - 1 for t in trajectories)
            return timesteps >= min_timesteps   
        conditions.append(timestep_cond)

    if min_episodes is not None:
        if min_episodes <= 0:
            raise ValueError(
                f"min_episodes={min_episodes} if provided must be positive"
            )
        conditions.append(lambda trajectories: len(trajectories) >= min_episodes)

    def sample_until(trajs: List[dict]) -> bool:
        for cond in conditions:
            if not cond(trajs):
                return False
        return True

    return sample_until

def rollout_and_save(
    path: str,
    policy,
    env,
    sample_until,
    *,
    exclude_infos: bool = True,
    **kwargs,
) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      policy: a stable_baselines3 policy or algorithm trained on the gym environment
      env: The environment to interact with.
      sample_until: End condition for rollout sampling.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      **kwargs: Passed through to `generate_trajectories`.
    """
    trajs = generate_trajectories(policy, env, sample_until, **kwargs)
    if exclude_infos:
        [traj.update(infos=None) for traj in trajs]
    save(path, trajs)

def save(path: str, trajectories: List[dict]) -> None:
    """Save a sequence of Trajectories to disk.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(trajectories, f)
    # Ensure atomic write
    os.replace(tmp_path, path)
    logging.info(f"Dumped demonstrations to {path}.")