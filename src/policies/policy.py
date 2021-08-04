
import torch
from torch import nn

from src.nets.deepsets import DeepSetsModule, Phi

class Policy:
    pass

class DeepSetsPolicy(Policy, nn.Module):
    def __init__(self, config):
        """
        Args:
            config (dict): dictionary for configuring the deep sets policy
        """
        super(DeepSetsPolicy, self).__init__()
        ego_config = config['ego_encoder']
        deepsets_config = config['deepsets']
        pathnet_config = config['path_encoder']

        self.ego_net = Phi.from_config(ego_config)
        self.deepsets_net = DeepSetsModule.from_config(deepsets_config)
        self.path_net = Phi.from_config(pathnet_config)

        cat_dim = self.ego_net.output_dim + self.deepsets_net.output_dim + self.path_net.output_dim
        # head has number of concatenated features as input
        head_config = config['head']
        head_config["input_dim"] = cat_dim
        self.head = Phi.from_config(head_config)

    def forward(self, sample):
        """
        Args:
            sample (dict): sample dictionary with the following entries:
                state (torch.tensor): (B, 5) raw state
                relative_state (torch.tensor): (B, max_nv, d) relative state (padded with nans)
                path (torch.tensor): (B, P, 2) tensor of P future path x and y positions
                action (torch.tensor): (B, 1) actions taken from each state
        Returns:
            x (torch.tensor): (head_output_dim,) output of common head network
        """
        ego = self.ego_net(sample["ego_state"])
        relative = self.deepsets_net(sample["relative_state"])
        path = self.path_net(sample["path"].reshape((sample["path"].shape[0], -1)))
        x = torch.cat([ego, relative, path], dim=-1)
        x = self.head(x)
        return x
