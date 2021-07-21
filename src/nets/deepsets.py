import torch
from torch import nn

from src.nets.util import parse_functional

class DeepSetsModule(nn.Module):
    def __init__(self, input_dim, phi_hidden_n, phi_hidden_dim, latent_dim, rho_hidden_n, rho_hidden_dim, output_dim):
        """
        Args:
            input_dim (int): input size of one instance of the set; input size of phi
            phi_hidden_n (int): number of hidden layers in phi
            phi_hidden_dim (int): size of hidden layers in phi
            latent_dim (int): output size of phi network, where sum is taken over instances; input size of rho
            rho_hidden_n (int): number of hidden layers in rho
            rho_hidden_dim (int): size of hidden layers in rho
            output_dim (int): output size of rho
        """
        super(DeepSetsModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi = Phi(self.input_dim, phi_hidden_n, phi_hidden_dim, latent_dim)
        self.rho = Phi(latent_dim, rho_hidden_n, rho_hidden_dim, self.output_dim)
        self.pooling = torch.sum  # torch.max # torch.mean

    @staticmethod
    def from_config(config):
        """
        Args:
            config (dict): dictionary with network parameters in the form
                {
                    "input_dim": 5,
                    "phi": {
                        "hidden_n": 1,
                        "hidden_dim": 10,
                    },
                    "latent_dim": 8,
                    "rho": {
                        "hidden_n": 1,
                        "hidden_dim": 10,
                    },
                    "output_dim" : 1,
                }
        Returns:
            m (nn.Module): deep sets module
        """
        input_dim = config["input_dim"]
        phi = config["phi"]
        latent_dim = config["latent_dim"]
        rho = config["rho"]
        output_dim = config["output_dim"]
        m = DeepSetsModule(input_dim, phi["hidden_n"], phi["hidden_dim"], latent_dim, rho["hidden_n"], rho["hidden_dim"], output_dim)
        return m

    def forward(self, x):
        """
        Args:
            x (torch.tensor): (batch_size, dynamic_size, input_dim)
        Returns:
            y (torch.tensor): (batch_size, output_dim)
        """
        # use negative dynamic_dim since batch dimensions are inserted at the front
        dynamic_dim = -2
        # iterate over dynamic dimension to apply phi to every instance
        latent = tuple(self.phi(instance) for instance in x.unbind(dynamic_dim))
        # stack outputs of phi
        latent = torch.stack(latent, dim=dynamic_dim)
        # apply pooling function to reduce dynamic dimension
        latent = self.pooling(latent, dim=dynamic_dim)
        # apply rho network
        y = self.rho(latent)
        return y


class Phi(nn.Module):
    def __init__(self, input_dim, hidden_n, hidden_dim, output_dim, final_activation=None):
        """
        Fully connected feedforward network with same size for all hidden layers and ReLU activation

        Args:
            input_dim (int): input dimension
            hidden_n (int): number of hidden layers
            hidden_dim (int): hidden layer dimension
            output_dim (int): output dimension
        """
        super(Phi, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = [nn.Linear(self.input_dim, hidden_dim)]
        for _ in range(hidden_n - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, self.output_dim))
        # self.in_layer = nn.Linear(input_dim, hidden_dim)
        # self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_n - 1)]
        # self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.functional.relu
        self.final_activation = final_activation if final_activation else lambda x: x

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.final_activation(self.layers[-1](x))
        return x

    @staticmethod
    def from_config(config):
        args = (config["input_dim"], config["hidden_n"], config["hidden_dim"], config["output_dim"])
        if "final_activation" in config:
            kwargs = {"final_activation": parse_functional(config["final_activation"])}
        else:
            kwargs = {}
        return Phi(*args, **kwargs)
