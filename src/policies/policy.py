
import torch
from torch import nn

from src.nets.deepsets import DeepSetsModule, Phi
from src.util.transform import MinMaxScaler

class IntersimStateNet(nn.Module):
    def __init__(self, config):
        """
        Args:
            config (dict): dictionary for configuring the deep sets policy
        """
        super(IntersimStateNet, self).__init__()
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


class IntersimStateActionNet(nn.Module):
    def __init__(self, config):
        """
        Args:
            config (dict): dictionary for configuring the deep sets policy
        """
        super(IntersimStateActionNet, self).__init__()
        ego_config = config['ego_encoder']
        deepsets_config = config['deepsets']
        pathnet_config = config['path_encoder']

        self.ego_net = Phi.from_config(ego_config)
        self.deepsets_net = DeepSetsModule.from_config(deepsets_config)
        self.path_net = Phi.from_config(pathnet_config)
        self.action_dim = config["action_dim"]

        cat_dim = self.ego_net.output_dim + self.deepsets_net.output_dim + self.path_net.output_dim + self.action_dim
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
        ego = self.ego_net(sample["ego_state"])
        relative = self.deepsets_net(sample["relative_state"])
        action = sample["action"]
        path = self.path_net(sample["path"].reshape((sample["path"].shape[0], -1)))
        x = torch.cat([ego, relative, path, action], dim=-1)
        x = self.head(x)
        return x


class IntersimPolicy():
    """
    Base class for intersim policies
    """
    def __init__(self, config, transforms):
        super(IntersimPolicy, self).__init__()
        self._config = config
        self._transforms = transforms

    @property
    def transforms(self):
        return self._transforms
    
    @transforms.setter
    def transforms(self, transforms):
        self._transforms=transforms

    @property 
    def policy(self):
        return self._policy
    
    @policy.setter
    def policy(self, policy):
        self._policy = policy

    def transform_observation(self, ob):
        # run observation through transforms
        transformed_ob = {}
        for key in ['ego_state', 'relative_state', 'path', 'action']:             
            if key in self._transforms.keys() and key in ob.keys():
                transformed_ob[key] = self._transforms[key].transform(ob[key])
        return transformed_ob

    def __call__(self, ob):

        if 'ego_state' in ob.keys():
            # extract state from dataloader samples  
            pass
        else:
            # extract state from observation (using simulator)
            ob['ego_state'] = ob['state']
            ob['path'] = torch.stack(ob['paths'],dim=-1)

        ob = self.transform_observation(ob)

        # run transformed state through model
        action = self._policy(ob)
        assert action.ndim == 2, 'action has incorrect shape'

        # untransform action
        if 'action' in self._transforms.keys():
            action = self._transforms['action'].inverse_transform(action)
        return action


def generate_transforms(dataset):
    """
    Generate transform dictionary from dataset
    Args:
        dataset (Dataset): dataset of demo observations and actions
    """
    transforms = {
        'action': MinMaxScaler(),
        'ego_state': MinMaxScaler(),
        'relative_state': MinMaxScaler(reduce_dim=2),
        'path': MinMaxScaler(reduce_dim=2),
    }
    for key in transforms.keys():
        if key == 'action':
            transforms[key].fit(dataset[:][key])
        else:
            transforms[key].fit(dataset[:]['state'][key])

    return transforms
