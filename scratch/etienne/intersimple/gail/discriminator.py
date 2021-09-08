import torch

# imitation.rewards.discrim_nets.DiscrimNetGAIL is composed of self.discriminator (nn.Module),
# which gets called with inputs (state, action) when needed.

class CnnDiscriminator(torch.nn.Module):
    """ConvNet similar to stable_baselines3.common.policies.ActorCriticCnnPolicy."""

    def __init__(self, env):
        super().__init__()

        obs_channels, _, _ = env.observation_space.shape
        (action_size,) = env.action_space.shape
        in_channels = obs_channels + action_size

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=(8, 8), stride=(4, 4)), # 5+1 -> 32
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)), # 32 -> 64
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)), # 64 -> 64
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.LazyLinear(512), # 28224 -> 512
            torch.nn.ReLU(),
            torch.nn.LazyLinear(1), # 512 -> 1
        )
    
    @staticmethod
    def _concatenate(state, action):
        b, _, h, w = state.shape
        _, a = action.shape
        act = action.unsqueeze(-1).unsqueeze(-1).expand((b, a, h, w))
        sa = torch.cat((state, act), -3)
        return sa

    def forward(self, state, action):
        sa = self._concatenate(state, action)
        return self.cnn(sa).squeeze()

class MlpDiscriminator(torch.nn.Module):
    """MLP similar to stable_baselines3.common.policies.ActorCriticPolicy."""

    def __init__(self, env=None):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.mlp = torch.nn.Sequential(
            torch.nn.LazyLinear(64), # 42 -> 64
            torch.nn.Tanh(),
            torch.nn.LazyLinear(64), # 64 -> 64
            torch.nn.Tanh(),
            torch.nn.LazyLinear(1),  # 64 -> 1
        )
    
    def forward(self, state, action):
        flat = self.flatten(state)
        sa = torch.cat((action, flat), -1)
        return self.mlp(sa).squeeze()
