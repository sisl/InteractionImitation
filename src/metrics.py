import torch
import pickle
import numpy as np

import intersim.collisions

def metrics(filestr: str, test_dataset, policy):
    """
    Calculate metrics using a) base filestring to a simulation, and b) the test dataset and learned policy
    Args:
        filestr (str): base string to outputs of a simulation
        test_dataset: a dataset held for testing
        policy: policy
    """

    # compute metrics using either
    # a) simulation files that were saved under the trained policy with prefix 'policy'
    # b) applying the policy to observations in the test dataset

    # load trajectory

    # count collisions (from function in intersim.collisions)

    # calculate average velocity

    # calculate divergence between velocity distributions
    
    # calcuate divergence between acceleration distributions
     
    pass

def average_velocity(x):
    """

    """
    pass

def divergence(p, q, type='kl'):
    """
    Calculate a divergence between p and q
    Args:
        p (torch.tensor): (n) samples from p
        q (torch.tensor): (n) samples from q
    Returns:
        d (float): approximate divergence
    """
    pass