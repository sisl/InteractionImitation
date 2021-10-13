import tqdm
import expert
import copy
import sys, os

def process_all_experts(env_args={}, policy_args={}):
    """
    Process all experts in the Interaction Dataset
    For now, using NormalizedIntersimpleExpert with NRasterizedIncrementingAgent environment

    Args:
        env_args (dict): default environment kwargs
        policy_args (dict): default policy kwargs
    """

    for loc in LOCATIONS:
        for track in TRACKS:
            
            it_env_args = copy.deepcopy(env_args)
            it_env_args.update({
                'loc':loc,
                'track':track,
            })

            it_path = 'newpathname'

            expert.demonstrations(
                expert='NormalizedIntersimpleExpert', 
                env='NRasterizedIncrementingAgent', 
                path=it_path, 
                env_args=it_env_args, 
                policy_args=policy_args,
                )
    

if __name__=='__main__':
    import fire
    fire.Fire(process_all_experts)


