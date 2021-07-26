
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
        ego_config = config['ego_state']
        deepsets_config = config['deepsets']
        pathnet_config = config['path_encoder']

        self.ego_net = Phi.from_config(ego_config) if ego_config else lambda x: x
        self.deepsets_net = DeepSetsModule.from_config(deepsets_config) if deepsets_config else lambda x: x
        self.path_net = Phi.from_config(pathnet_config) if pathnet_config else lambda x: x

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
                path_x (torch.tensor): (B, P) tensor of P future path x positions
                path_y (torch.tensor): (B, P) tensor of P future path y positions
                action (torch.tensor): (B, 1) actions taken from each state
        Returns:
            x (torch.tensor): (head_output_dim,) output of common head network
        """
        ego = self.ego_net(sample["state"])
        relative = self.deepsets_net(sample["relative_state"])
        # cat path_x, path_y to tensor of dim (B, 2*P)
        path = torch.cat([sample["path_x"], sample["path_y"]], dim=-1)
        path = self.path_net(path)
        x = torch.cat([ego, relative, path], dim=-1)
        x = self.head(x)
        return x
