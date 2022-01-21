from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from torch.distributions import Categorical

import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

class CustomCNN(BaseFeaturesExtractor):
    """
    Smaller version of `stable_baselines3.common.torch_layers.NatureCNN`

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use CustomCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 4, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class OptionsCnnPolicy(ActorCriticPolicy):
    """
    Class for high-level options policy (generator)
    """
    def __init__(self, observation_space, *args, eps=0, **kwargs):
        super().__init__(observation_space, *args, **kwargs)
        self.cnn_policy = ActorCriticCnnPolicy(observation_space['obs'], *args, features_extractor_class=CustomCNN, **kwargs)
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
