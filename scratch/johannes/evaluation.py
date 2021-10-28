
def evaluate_policy_simple(
    model,
    env: gym.Env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback = None,
    reward_threshold = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    episode_rewards = []
    episode_lengths = []

    episode_counts = 0

    current_rewards = 0
    current_lengths = 0
    observations = env.reset()
    states = None
    while (episode_counts < n_eval_episodes):
        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        print(env._env.t)
        current_rewards += rewards
        current_lengths += 1

        # unpack values so that the callback can access the local variables
        reward = rewards
        done = dones
        info = infos
        if info['collision']:
            print("COLLISION")

        if callback is not None:
            callback(locals(), globals())

        if dones:
            episode_rewards.append(current_rewards)
            episode_lengths.append(current_lengths)
            episode_counts += 1
            current_rewards = 0
            current_lengths = 0
            if states is not None:
                states *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
