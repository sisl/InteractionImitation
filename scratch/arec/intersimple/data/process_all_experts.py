import tqdm
import expert
import copy
import os
import intersim
from tqdm import tqdm

def process_all_experts(filename='expert.pkl',env_args={}, policy_args={}):
    """
    Process all experts in the Interaction Dataset
    For now, using NormalizedIntersimpleExpert with NRasterizedIncrementingAgent environment

    Args:
        filename (str): name for track file
        env_args (dict): default environment kwargs
        policy_args (dict): default policy kwargs
    """
    I, J = len(intersim.LOCATIONS), intersim.MAX_TRACKS
    pbar = tqdm(total=I*J)
    for loc in range(I):
        for track in range(J):
            
            it_env_args = copy.deepcopy(env_args)
            it_env_args.update({
                'loc':loc,
                'track':track,
            })
            out_folder = os.path.join(intersim.LOCATIONS[loc], 'track%04i'%(track))
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder) 
            it_path = os.path.join(out_folder,filename)

            expert.demonstrations(
                expert='NormalizedIntersimpleExpert', 
                env='NRasterizedIncrementingAgent', 
                path=it_path, 
                env_args=it_env_args, 
                policy_args=policy_args,
                )
            pbar.update(1)
    pbar.close()
    

if __name__=='__main__':
    import fire
    fire.Fire(process_all_experts)


