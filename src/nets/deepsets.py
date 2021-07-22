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
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.phi = Phi(self.input_dim, phi_hidden_n, phi_hidden_dim, self.latent_dim)
        self.rho = Phi(self.latent_dim, rho_hidden_n, rho_hidden_dim, self.output_dim)
        self.pooling = torch.sum

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
            x (torch.tensor): ([B, ]max_nv, d)
        Returns:
            y (torch.tensor): ([B, ]output_dim)
        """
        # mask for selecting only those batches and vehicles where all relative states are not nan
        # shape (B, max_nv)
        notnan_mask = torch.all(~torch.isnan(x), dim=-1)
        # create zero tensor of shape (B, max_nv, latent_dim) to store phi evaluations in
        latent = torch.zeros([*x.shape[:-1], self.latent_dim])
        # evaluate phi for all not NaN entries
        # x[batch_dynamic_mask] has shape (notnan_mask.sum(), input_dim)
        latent[notnan_mask] = self.phi(x[notnan_mask])

        # sum over relative state dimension
        latent = self.pooling(latent, dim=-2)
        
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
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, hidden_dim)])
        for _ in range(hidden_n - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, self.output_dim))
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
