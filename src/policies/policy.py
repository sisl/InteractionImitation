
import torch
from torch import nn

from src.nets.deepsets import DeepSetsModule, Phi

class Policy:
    pass

class DeepSetsPolicy(Policy, nn.Module):
    def __init__(self, ego_config, dynamic_config, path_config, head_config):
        """
        Args:
            ego_config (dict): dictionary for configuring the ego network
            dynamic_config (dict): dictionary for configuring the dynamic input (deepsets) network
            path_config (dict): dictionary for configuring the path network
            head_config (dict): dictionary for configuring the common head network
        """
        self.ego_net = Phi.from_config(ego_config)
        self.deepsets = DeepSetsModule.from_config(dynamic_config)
        self.path_net = Phi.from_config(path_config)
        self.head = Phi.from_config(path_config)
        output_dim = self.ego_net.output_dim + self.deepsets.output_dim + self.path_net.output_dim
        assert output_dim == head_config["input_dim"]

    def forward(self, ego_state, relative_states, path):
        """
        Args:
            ego_state (torch.tensor): (ns,) state of ego vehicle
            relative_states (torch.tensor): (nv, ns) relative states of other vehicles (dynamic size)
            path (torch.tensor): (path_length, 2) coordinates (x,y) of path
        Returns:
            x (torch.tensor): (head_output_dim,) output of common head network
        """
        x_ego = self.ego_net(ego_state)
        x_relative = self.deepsets(relative_states)
        x_path = self.path_net(path.flatten())
        x = torch.cat([x_ego, x_relative, x_path])
        x = self.head(x)
        return x




