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
    Returns:
        info (dict): metrics in a dictionary
    """
    info = {}

    # compute metrics using either
    # a) simulation files that were saved under the trained policy with prefix 'policy'
    # b) applying the policy to observations in the test dataset

    # load simulated trajectory
    states = torch.load(filestr + '_sim_states.pt').detach()
    lengths = torch.load(filestr + '_sim_lengths.pt').detach()
    widths = torch.load(filestr + '_sim_widths.pt').detach()
    xpoly = torch.load(filestr + '_sim_xpoly.pt').detach()
    ypoly = torch.load(filestr + '_sim_ypoly.pt').detach()

    # count collisions (from function in intersim.collisions)
    n_collisions = collisions.count_collisions_trajectory(states, lengths, widths)
    info['n_collisions'] = n_collisions

    # calculate average velocity
    avg_v = average_velocity(states)
    info['average_velocity'] = avg_v

    # convert policy dtype between float32 and float64
    policy.policy = policy.policy.type(test_dataset[0]['state']['ego_state'].dtype)
    
    # generate actions in test dataset
    true_actions, pred_actions = [], []
    true_velocities = []
    test_loader = DataLoader(test_dataset, batch_size=1024)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            pred_actions.append(policy(batch['state']))
            true_actions.append(batch['action'])
            true_velocities.append(batch['state']['ego_state'][:,2])

    true_actions, pred_actions = torch.cat(true_actions,dim=0), torch.cat(pred_actions, dim=0)
    visualize_distribution(true_actions[:,0], pred_actions[:,0], filestr+'_action_viz') 

    # calculate divergence between acceleration distributions
    acceleration_divergence = divergence(pred_actions, true_actions, type='js')
    info['acceleration_divergence'] = acceleration_divergence

    # calculate divergence between velocity distributions
    sim_velocities = states[:,:,2]
    sim_velocities = sim_velocities[~torch.isnan(sim_velocities)].flatten()
    true_velocities = torch.cat(true_velocities, dim=0)
    velocity_divergence = divergence(sim_velocities, true_velocities, type='js')
    info['velocity_divergence'] = velocity_divergence

    return info


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
    plt.figure()
    plt.hist(true[nni1].numpy(), density=True, bins=20)
    plt.hist(pred[nni2].numpy(), density=True, bins=20)
    plt.legend(['True', 'Predicted'])
    plt.savefig(filestr+'.png')
    
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
