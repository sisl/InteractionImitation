from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from torch.distributions import Categorical

class OptionsCnnPolicy(ActorCriticPolicy):
    """
    Class for high-level options policy (generator)
    """
    def __init__(self, observation_space, *args, eps=0, **kwargs):
        super().__init__(observation_space, *args, **kwargs)
        self.cnn_policy = ActorCriticCnnPolicy(observation_space['obs'], *args, **kwargs)
        self.eps = eps

    def _prior_distribution(self, s):
        """
        Return prior distribution over high-level options (before masking)
        Args:
            s (torch.tensor): observation
        Returns:
            values (torch.tensor): values from critic
            dist (torch.distributions): prior distribution over actions
        """
        latent_pi, latent_vf, latent_sde = self.cnn_policy._get_latent(s)
        distribution = self.cnn_policy._get_action_dist_from_latent(latent_pi, latent_sde)
        values = self.cnn_policy.value_net(latent_vf)
        return values, distribution.distribution

    def forward(self, obs):
        """
        Will mask invalid states before making action selections
        Args:
            obs: dict with keys:
                obs (torch.tensor): (*,o)  true observations
                mask (torch.tensor): (*,m) mask over valid actions
        Returns:
            ch (torch.tensor): (*,a) sampled actions
            values (torch.tensor): (*,) predicted value at observation
            log_probs (torch.tensor): (*,) log probabilities of selected actions
        """
        s, m = obs['obs'], obs['mask']
        values, prior = self._prior_distribution(s)
        posterior = Categorical((prior.probs + self.eps) * m)
        ch = posterior.sample()
        return ch, values, posterior.log_prob(ch)
    
    def _predict(self, obs, deterministic=False):
        action, _, _ = self.forward(obs)
        return action

    def evaluate_actions(self, obs, ch):
        """
        Evaluate particular actions
        Args:
            obs: dict with keys:
                obs (torch.tensor): (*,o) true observations
                mask (torch.tensor): (*,m) masks over valid actions
            ch (torch.tensor): (*,a) selected actions
        Returns:
            values (torch.tensor): (*,) predicted value at observation
            log_probs (torch.tensor): (*,) log probabilities of selected actions
            ent (torch.tensor): (*,) entropy of each distribution over actions
        """
        s, m = obs['obs'], obs['mask']
        values, prior = self._prior_distribution(s)
        posterior = Categorical((prior.probs + self.eps) * m)
        return values, posterior.log_prob(ch), posterior.entropy() # additional values used by PPO.train
