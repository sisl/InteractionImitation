import pickle
import os
import numpy as np
from typing import List,Dict

def save_metrics(metrics:dict, filestr:str):
    """
    Save metric dict to filestr
    Args:
        metrics (dict): dict to save
        filestr (str): strig to save dict to
    
    """
    # make filepath
    if not os.path.isdir(os.path.dirname(filestr)):
        os.makedirs(os.path.dirname(filestr))

    # pickle dump
    with open(filestr, 'wb') as f:
        pickle.dump(metrics, f) 

def load_metrics(filestr:str):
    """
    Load metric dict from filestr
    Args:
        filestr (str): strig to save dict to

    Return:
        metrics (dict): loaded metrics
    
    """
    # pickle load
    with open(filestr, 'rb') as f:
        metrics = pickle.load(f) 
    return metrics

def average_metrics(metric_list:List[Dict[str,float]], verbose:bool=True) ->Dict[str, tuple]:
    """
    Average all the metrics in the list
    
    Args:
        metric_list (list of dicts): list of metric dicts which each map a string to a float
        verbose (bool): whether to print avg metrics
    Returns:
        average_metrics (Dict[str, tuple])
    """
    average_metrics = {}
    if len(metric_list) == 0:
        return average_metrics
    
    keys = list(metric_list[0].keys())
    N = len(metric_list)
    master_dict = {key:[] for key in keys}
    for key in keys:
        for i in range(N):
            master_dict[key].append(metric_list[i][key])
        master_dict[key] = np.array(master_dict[key])
        mu = np.nanmean(master_dict[key])
        std2 = np.nanstd(master_dict[key])*2
        if verbose:
            print(f'{key}: {mu} \pm {std2}')
        average_metrics[key] = (mu, std2)
    return average_metrics


def load_and_average(path:str, verbose:bool=True):
    """
    Load and average all metric files in a particular folder

    Args:
        path (str)
        verbose (bool): whether to print avg metrics
    Returns:
        avg_metrics (Dict[str, tuple])
    """
    assert os.path.isdir(path)

    # summary metrics
    summary_files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('summary.pkl')]
    if verbose: 
        print(*summary_files, sep='\n')
    all_summary_metrics = [load_metrics(f) for f in summary_files]
    avg_metrics = average_metrics(all_summary_metrics, verbose=verbose)

    # comparison metrics
    comp_files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('comparison.pkl')]
    if verbose:
        print(*comp_files, sep='\n') 
    all_comp_metrics = [load_metrics(f) for f in comp_files]
    comp_avg = average_metrics(all_comp_metrics, verbose=verbose)
    avg_metrics.update(comp_avg)
    return avg_metrics

if __name__=='__main__':
    import fire
    fire.Fire()