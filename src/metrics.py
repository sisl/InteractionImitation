import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from intersim import collisions

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
    states = torch.load(filestr + '_sim_states.pt').detach()
    lengths = torch.load(filestr + '_sim_lengths.pt').detach()
    widths = torch.load(filestr + '_sim_widths.pt').detach()
    xpoly = torch.load(filestr + '_sim_xpoly.pt').detach()
    ypoly = torch.load(filestr + '_sim_ypoly.pt').detach()

    # count collisions (from function in intersim.collisions)
    n_collisions = collisions.count_collisions_trajectory(states, lengths, widths)

    # calculate average velocity
    avg_v = average_velocity(states)

    # calculate divergence between velocity distributions
    
    # calcuate divergence between acceleration distributions
    policy.policy = policy.policy.type(test_dataset[0]['state'].dtype)
    
    # generate actions in test dataset
    true_actions, pred_actions = [], []
    test_loader = DataLoader(test_dataset, batch_size=1024)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            pred_actions.append(policy(batch))
            true_actions.append(batch['action'])

    true_actions, pred_actions = torch.cat(true_actions,dim=0), torch.cat(pred_actions,dim=0)
    visualize_distribution(true_actions[:,0], pred_actions[:,0], filestr+'_action_viz') 


def visualize_distribution(true, pred, filestr):
    """
    Visualize two distributions
    Args:
        true (torch.tensor): (n,)-sized true distribution
        pred (torch.tensor): (m,)-sized pred distribution
        filestr (str): string to save figure to
    """
    nni1 = ~torch.isnan(true)
    nni2 = ~torch.isnan(pred)
    import pdb
    pdb.set_trace()
    plt.figure()
    plt.hist(true[nni1].numpy(), density=True, bins=20)
    plt.hist(pred[nni2].numpy(), density=True, bins=20)
    plt.legend(['True', 'Predicted'])
    plt.savefig(filestr+'.png')
    
def average_velocity(states):
    """
    Compute average of average velocity over all vehicles.
    Args:
        states (torch.tensor): (T,nv,5) vehicle states
    Returns
        avg_v (float): average velocity
    """
    velocities = states[:,:,2]
    vehicle_avg_v = nanmean(velocities, dim=0)
    arg_v = nanmean(vehicle_avg_v)
    return arg_v

def divergence(p, q, type='kl'):
    """
    Calculate a divergence between p and q
    Args:
        p (torch.tensor): (n) samples from p
        q (torch.tensor): (m) samples from q
    Returns:
        d (float): approximate divergence
    """
    pass

def nanmean(v, *args, inplace=False, **kwargs):
    """
    Calculate mean over not nan entries

    To be added to torch as torch.nanmean in the next release
    https://github.com/pytorch/pytorch/issues/61474, https://github.com/pytorch/pytorch/issues/21987

    Args:
        v (torch.tensor): arbitrary tensor
    Returns:
        result (torch.tensor): mean over non nan elements
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    result = v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
    return result