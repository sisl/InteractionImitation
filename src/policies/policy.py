
import torch
from torch import nn

from src.nets.deepsets import DeepSetsModule, Phi

class Policy:
    pass

class DeepSetsPolicy(Policy, nn.Module):
    def __init__(self, ego_config, deepsets_config, path_config, head_config):
        """
        Args:
            ego_config (dict): dictionary for configuring the ego network
            deepsets_config (dict): dictionary for configuring the deepsets network
            path_config (dict): dictionary for configuring the path network
            head_config (dict): dictionary for configuring the common head network
        """
        super(DeepSetsPolicy, self).__init__()
        self.ego_net = Phi.from_config(ego_config)
        self.deepsets_net = DeepSetsModule.from_config(deepsets_config)
        self.path_net = Phi.from_config(path_config)
        cat_dim = self.ego_net.output_dim + self.deepsets_net.output_dim + self.path_net.output_dim
        # head has number of concatenated features as input
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
        x = torch.cat([ego, relative, path])
        x = self.head(x)
        return x
