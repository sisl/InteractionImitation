import pickle
import imitation.data.rollout as rollout
from tqdm import tqdm

def load_experts(expert_files=[]):
    """
    Load expert trajectories from files and combine their transitions into a single RB

    Args:
        expert_files (list): list of expert file strings
    Returns:
        transitions (list): list of combined expert episode transitions 
    """
    transitions = []
    for file in tqdm(expert_files):
        with open(file, "rb") as f:
            trajectories = pickle.load(f)
        transitions = transitions + rollout.flatten_trajectories(trajectories)
    return transitions

if __name__=='__main__':
    import fire
    fire.Fire(load_experts)