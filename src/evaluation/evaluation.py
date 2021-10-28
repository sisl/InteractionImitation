from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from intersim.envs.intersimple import Intersimple
from src.evaluation.metrics import nanmean, divergence, visualize_distribution

class Evaluation:
    def __init__(self, eval_env, n_eval_episodes=10):
        # if env is a VecEnv, the code needs to be adapted, since the callback will be called after each step, 
        # so transitions of different envs will be mixed and the total number of episodes could be larger than n_eval_episodes!
        assert not isinstance(eval_env, VecEnv)
        self.env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.reset()

    def reset(self):
        self._n_collisions = 0
        self._trajectories = []
        self._episode_done = True
        self._accelerations = []

    def evaluate(self, epoch, generator, discriminator, expert_data):
        self.reset()
        metrics = {}
        
        episode_rewards, episode_lengths = evaluate_policy(
            generator, 
            self.env,
            n_eval_episodes=self.n_eval_episodes,
            callback=self.evaluate_policy_callback, 
            return_episode_rewards=True
        )

        collision_rate = self._n_collisions / self.n_eval_episodes
        metrics['collision_rate'] = collision_rate

        assert len(self._trajectories) >= self.n_eval_episodes

        # average velocity of each episode
        # this first averages velocity over single trajectories and then averages over trajectories
        # avg_velocities = [nanmean(torch.stack(t)[:,2]) for t in self._trajectories]
        # avg_velocity = np.mean(avg_velocities)
        
        # velocities produced by generator
        policy_velocities = torch.cat([torch.stack(t)[:,2] for t in self._trajectories])
        # if episodes terminate without collisions, then the state is fully nan
        policy_velocities = policy_velocities[~torch.isnan(policy_velocities)]

        # expert velocities
        extract_state = lambda info: info['projected_state'][info['agent']]
        expert_velocities = torch.stack([extract_state(info) for info in expert_data.infos])[:,2]
        expert_velocities = expert_velocities[~torch.isnan(expert_velocities)]

        metrics['avg_velocity_loss'] = (expert_velocities.mean() - policy_velocities.mean()).item()
        metrics['velocity_divergence'] = divergence(policy_velocities, expert_velocities, type='js')
        

        # accelerations produced by generator
        policy_accelerations = torch.tensor(self._accelerations)
        # expert accelerations
        extract_accel = lambda info: info['action_taken'][info['agent']]
        expert_accelerations = torch.cat([extract_accel(info) for info in expert_data.infos])

        metrics['acceleration_divergence'] = divergence(policy_accelerations, expert_accelerations, type='js')
        visualize_distribution(expert_accelerations, policy_accelerations, 'output/_action_viz{:02}'.format(epoch)) 

        print(metrics)
        return metrics

    def evaluate_policy_callback(self, local_vars, global_vars):
        venv_i = local_vars['i']
        info = local_vars['info']
        done = local_vars['done']
        _agent = info['agent']
        env = local_vars['env'].envs[venv_i]
        assert isinstance(env, Intersimple)

        # Increase collision counter if episode terminated with a collision
        if info['collision']:
            assert done
            self._n_collisions += 1

        # if last episode is done, start new trajectory
        if self._episode_done:
            self._trajectories.append([])
        agent_state = info['projected_state'][_agent]
        self._trajectories[-1].append(agent_state)
        # this does not work since agent_action are high level options
        # agent_action = local_vars['actions'][venv_i] # normalized intersimple action
        # acceleration = env._unnormalize(agent_action) if isinstance(env, NormalizedActionSpace) else agent_action
        self._accelerations.append(info['action_taken'][_agent])
        self._episode_done = done
