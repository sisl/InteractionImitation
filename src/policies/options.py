import stable_baselines3
from torch.distributions import Categorical

class OptionsCnnPolicy(stable_baselines3.common.policies.ActorCriticCnnPolicy):
    """
    Class for high-level options policy (generator)
    """
    def __init__(self, observation_space, *args, **kwargs):
        super().__init__(observation_space['obs'], *args, **kwargs)

    def _prior_distribution(self, s):
        """
        Return prior distribution over high-level options (before masking)
        Args:
            s (torch.tensor): observation
        Returns:
            values (torch.tensor): values from critic
            dist (torch.distributions): prior distribution over actions
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(s)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        values = self.value_net(latent_vf)
        return values, distribution.distribution

    def predict(self, obs, eps=1e-6):
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
        posterior = Categorical((prior.probs + eps) * m)
        ch = posterior.sample()
        return ch, values, posterior.log_prob(ch)

    def evaluate_actions(self, obs, ch, eps=1e-6):
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
        posterior = Categorical((prior.probs + eps) * m)
        return values, posterior.log_prob(ch), posterior.entropy() # additional values used by PPO.train
