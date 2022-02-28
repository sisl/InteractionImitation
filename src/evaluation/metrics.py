import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from intersim import collisions
from typing import List, Dict
# import tikzplotlib 

def rwse(expert:List[np.ndarray], policy:List[np.ndarray], dt:float=0.1) -> Dict[str,float]:
    """
    Calculate average mean squared displacement error
    
    Args:
        expert (List[np.ndarray]): all position trajectories for all expert rollouts
        policy (List[np.ndarray]): all position trajectories for all policy rollouts

    each trajectory in the list should have shape (2, T). however expert[i] might have a 
    different T than policy[i]
    
    Returns
        rwse_dict (Dict[str,float]): dict of different RWSEs
    """
    assert len(expert) == len(policy)

    # calculate rwse
    times = [1,2,5,10,15,20,25,30]
    time_indices = [int(t/dt) for t in times]
    rwse_dict_keys = [f'rwse_{t}s' for t in times]+['rwse_end']
    se_dict = {key:[] for key in rwse_dict_keys}
    for expert_trajectory, policy_trajectory in zip(expert, policy):
        _, T1 = expert_trajectory.shape
        _, T2 = policy_trajectory.shape
        minT = min(T1, T2)

        crop_expert_trajectory = expert_trajectory[:, :minT]
        crop_policy_trajectory = policy_trajectory[:, :minT]
        
        # square error along every time
        se = ((crop_policy_trajectory - crop_expert_trajectory)**2).sum(0)
        
        # add to dict with appropriate indexing
        for time, idx in zip(times, time_indices):
            if minT >= idx:
                se_dict[f'rwse_{time}s'].append(se[idx-1])
        se_dict['rwse_end'].append(se[-1])

    assert len(se_dict['rwse_end']) == len(expert)

    # print how many trajectories of each time:
    for key in rwse_dict_keys:
        print('%s has %i elements'%(key, len(se_dict[key])))

    rwse_dict = {key:np.mean(np.array(se_dict[key]))**0.5 for key in rwse_dict_keys}

    return rwse_dict

def rwse_basic(expert:List[np.ndarray], policy:List[np.ndarray], dt:float=0.1) -> float:
    """
    Calculate average mean squared displacement error
    
    Args:
        expert (List[np.ndarray]): all position trajectories for all expert rollouts
        policy (List[np.ndarray]): all position trajectories for all policy rollouts

    each trajectory in the list should have shape (2, T). however expert[i] might have a 
    different T than policy[i]
    
    Returns
        rwse (float): rwse of positions
    """
    assert len(expert) == len(policy)

    # calculate rwse
    rwse = []
    for expert_trajectory, policy_trajectory in zip(expert, policy):
        _, T1 = expert_trajectory.shape
        _, T2 = policy_trajectory.shape
        minT = min(T1, T2)

        crop_expert_trajectory = expert_trajectory[:, :minT]
        crop_policy_trajectory = policy_trajectory[:, :minT]
        e = ((crop_policy_trajectory - crop_expert_trajectory)**2).sum(0).mean()

        rwse.append(e)
    
    assert len(rwse) == len(expert)
    rwse = np.array(rwse)
    avg_rwse = rwse.mean()

    return avg_rwse

def visualize_distribution(expert, policy, filestr):
    """
    Visualize two distributions
    Args:
        expert (torch.tensor): (n,)-sized true distribution
        generated (torch.tensor): (m,)-sized pred distribution
        filestr (str): string to save figure to
    """
    nni1 = ~torch.isnan(expert)
    nni2 = ~torch.isnan(policy)
    plt.figure()
    plt.hist(expert[nni1].numpy(), density=True, bins=20)
    plt.hist(policy[nni2].numpy(), density=True, bins=20)
    plt.legend(['Expert', 'Predicted'])
    plt.savefig(filestr+'.png')
    # tikzplotlib.save(filestr+'.tex')
    
def average_velocity(states):
    """
    Compute average of average velocity over all vehicles.
    Args:
        states (torch.tensor): (T,nv,5) vehicle states where T is the number of time steps and nv the number of vehicles
    Returns
        avg_v (float): average velocity
    """
    velocities = states[:,:,2]
    # average velocity per vehicle
    vehicle_avg_v = nanmean(velocities, dim=0)
    arg_v = nanmean(vehicle_avg_v)
    return arg_v

def divergence(p, q, type='js', n_components=-1):
    """
    Calculate a divergence between p and q
    Args:
        p (torch.tensor): (n) samples from p
        q (torch.tensor): (m) samples from q
        type (str): divergence to use
            'kl': Kullback-Leibler divergence KL(p||q)
            'js': Jensen-Shannon divergence (symmetric KLD)
        n_components (int): method to use to compute kl divergence
            n_components < 0: approximate samples with histogram density
            n_components == 0: approximate samples by Gaussian distributions and compute analytically
            n_components > 0: approximate samples as Gaussian mixture models with n_components components
    Returns:
        d (float): approximate divergence
    """
    if type == 'js':
        # Use histogram binning to discretize sampled distributions
        p_hist = np.histogram(p, bins='auto', density=True)
        q_hist = np.histogram(q, bins='auto', density=True)
        m = torch.cat([p, q], dim=0)
        m_weights = torch.cat([torch.full_like(p, 1./len(p)), torch.full_like(q, 1./len(q))], dim=0)
        m_bins = np.sort(np.concatenate([p_hist[1], q_hist[1]]))
        m_hist = np.histogram(m, bins=m_bins, density=True, weights=m_weights)
        d = .5 * kl_histogram(p, p_hist, m_hist) + .5 * kl_histogram(q, q_hist, m_hist)
        return d
    elif type == 'kl':
        if n_components < 0:
            # Use histogram binning to discretize sampled distributions
            p_hist = np.histogram(p, bins='auto', density=True)
            q_hist = np.histogram(q, bins='auto', density=True)
            d = kl_histogram(p, p_hist, q_hist)
            return d
        elif n_components == 0:
            # Assume p and q to be Gaussian
            pm = torch.mean(p)
            qm = torch.mean(q)
            pv = torch.var(p)
            qv = torch.var(q)
            d = kl_normal(pm, pv, qm, qv).item()
            return d
        else:
            from sklearn.mixture import GaussianMixture
            p = p.unsqueeze(-1)
            q = q.unsqueeze(-1)
            p_gmm = GaussianMixture(n_components=n_components).fit(p)
            q_gmm = GaussianMixture(n_components=n_components).fit(q)
            px = p_gmm.score_samples(p)
            qx = q_gmm.score_samples(p)
            d = np.mean(px - qx).item()
            return d
    else:
        raise NotImplementedError("Please implement divergence for type '{}'".format(type))

def kl_histogram(p_sample, p_hist, q_hist):
    """
    Calculate the kl divergence between p and q based on a histogram representation
    Args:
        p_sample (torch.tensor): (n) samples from p
        p_hist (tuple): result of np.histogram(density=True) for samples from p
        q_hist (tuple): result of np.histogram(density=True) for samples from q
    Returns:
        d (float): approximate KL divergence
    """
    p_density, p_edges = p_hist
    q_density, q_edges = q_hist
    px = evaluate_histogram(p_sample, p_density, p_edges)
    qx = evaluate_histogram(p_sample, q_density, q_edges)
    p_supp = ~np.isclose(px, 0.0)
    q_supp = ~np.isclose(qx, 0.0)
    if np.any(np.logical_and(p_supp, ~q_supp)):
        # if not support(p) subset support(q)
        return np.inf
    elif ~np.any(p_supp):
        # if p is zero everywhere
        return 0.
    d = np.mean(np.log(px[p_supp] / qx[p_supp]))
    return d


def kl_normal(pm, pv, qm, qv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(p || q) and
    sum over the last dimension

    Args:
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(qv) - torch.log(pv) + pv / qv + (pm - qm).pow(2) / qv - 1)
    kl = element_wise.sum(-1)
    return kl


def kl_cat(q, log_q, log_p):
    """
    Computes the KL divergence between two categorical distributions

    Args:
        q: tensor: (batch, dim): Categorical distribution parameters
        log_q: tensor: (batch, dim): Log of q
        log_p: tensor: (batch, dim): Log of p

    Return:
        kl: tensor: (batch,) kl between each sample
    """
    element_wise = (q * (log_q - log_p))
    kl = element_wise.sum(-1)
    return kl


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


def evaluate_histogram(x, hist, bin_edges):
    """
    Evaluate a histogram 
    Args:
        x (array) : points at which to evaluate the histogram
        hist (array): histogram values in terms of number of occurrences or probability
        bin_edges (array): edges of histogram bins
            e.g. from hist, bin_edges = np.histogram(p, bins='auto', density=True)
    Return:
        r: tensor: (batch,) kl between each sample
    """
    idx = np.digitize(x, bin_edges)
    mask = np.logical_and(np.less(0, idx), np.less(idx, len(bin_edges)))
    r = np.zeros_like(x)
    r[mask] = hist[idx[mask] - 1]
    return r

