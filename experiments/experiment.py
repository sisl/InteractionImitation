import json5
from functools import partial
import os
opj = os.path.join

from src.main import basestr, main

def parse_args():
    """
    Parse arguments to main
    Returns:
        kwargs: dictionary of arguments:
            train (bool): whether to run train loop
            test (bool): whether to run test loop
            method (str): the method to try for imitation
            loc (int): the location index of the roundabout
            config (str): config path
            seed (int): RNG seed
    """
    import argparse
    parser = argparse.ArgumentParser(description='Save Expert Trajectories')
    parser.add_argument('--loc', default=0, type=int,
        help='location (default 0)')
    parser.add_argument("--train", help="train model",
        action="store_true")
    parser.add_argument("--ray", help="use ray tune to run multiple experiments",
        action="store_true")
    parser.add_argument("--test", help="test model",
        action="store_true") 
    parser.add_argument("--method", help="modeling method",
        choices=['bc', 'gail', 'advil'], default='bc')
    parser.add_argument("--config", help="config file path",
        default=None, type=str)
    parser.add_argument('--seed', default=0, type=int,
        help='seed')    
    parser.add_argument('--nframes', default=500, type=int,
        help='frames for test animation') 
    args = parser.parse_args()
    kwargs = {
        'train':args.train, 
        'test':args.test, 
        'method':args.method,
        'loc':args.loc,
        'config_path':args.config,
        'seed':args.seed,
        'ray':args.ray,
        'nframes':args.nframes,
    }
    return kwargs

def get_full_config(ray_config:dict, method:str)->dict:
    """
    Get full model configuration from ray config and method string
    Args:
        ray_config (dict): ray config
        method (str): method to get full configuration for
    """
    if method == 'bc':
        from src.bc import bc_config
        config = bc_config(ray_config)
    else:
        raise NotImplementedError
    return config

def get_ray_config(method:str)->dict:
    """
    Get configuration for ray based on method.
    Args: 
        method (str): method to get configuration for
    Returns:
        ray_config (dict): configuration for ray
    """
    if method == 'bc':
        ray_config = {
            "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
            "weight_decay": tune.choice([0.001, 0.01, 0.1, 0.5, 0.9]),
            "loss": tune.choice(['huber', 'mse']),
            "train_batch_size": tune.choice([16,32,64]),
            "deepsets_phi_hidden_n": tune.choice([1,2,3]),
            "deepsets_phi_hidden_dim": tune.choice([16,32,64]),
            "deepsets_latent_dim": tune.choice([16,32,64]),
            "deepsets_rho_hidden_n": tune.choice([0,1,2]),
            "deepsets_rho_hidden_dim": tune.choice([16,32,64]),
            "deepsets_output_dim": tune.choice([8,16,32,64]),
            "head_hidden_n": tune.choice([1,2,3]),
            "head_hidden_dim": tune.choice([16,32,64]),
            "head_final_activation": tune.choice(['sigmoid', None]),
        }
    else:
        raise NotImplementedError
    return ray_config

if __name__ == '__main__':
    kwargs = parse_args()
    
    # make prefix of output files
    outdir = opj('output',kwargs['method'],'loc%02i'%(kwargs['loc']))

    if kwargs['config_path']:
        # load config
        with open(kwargs['config_path'], 'r') as cfg:
            config = json5.load(cfg)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)   
        filestr = opj(outdir, basestr(**kwargs)) 
        if kwargs['ray']:
            filestr = kwargs['config_path'].replace('_config.json','')
        main(config, filestr=filestr, **kwargs)

    elif kwargs['ray'] and kwargs['train']:
        # set up ray tune
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

        ray.shutdown() 
        ray.init(log_to_driver=False)

        def ray_train(config, datadir=None):
            full_config = get_full_config(config, kwargs['method'])
            main(full_config, filestr='exp', datadir=datadir, **kwargs)
        
        datadir = os.path.abspath('./expert_data')
        ray_config = get_ray_config(kwargs['method'])
        custom_scheduler = ASHAScheduler(
            metric='cv_loss',
            mode="min",
            grace_period=25,
        )
        analysis = tune.run(
            partial(ray_train, datadir=datadir),
            config=ray_config,
            scheduler=custom_scheduler,
            local_dir=outdir,
            #resources_per_trial={"cpu": 2},
            time_budget_s=45*60,
            num_samples=2,
        )
    elif kwargs['ray'] and kwargs['test']:
        import ray
        from ray.tune import Analysis, ExperimentAnalysis
        analysis = Analysis(outdir, default_metric="cv_loss", default_mode="min")
        config = analysis.get_best_config()
        filepath = analysis.get_best_logdir()
        print("Best ray experiment:", filepath)
        main(None, filestr=opj(filepath, 'exp'), **kwargs)
    else:
        raise Exception('No valid config found')

    
    
    

    
