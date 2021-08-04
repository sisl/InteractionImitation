import json5
from functools import partial
import os
opj = os.path.join

# set up ray tune
import ray
from ray import tune
from ray.tune import Analysis, ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

# get graphs
import intersim
from intersim.graphs import ConeVisibilityGraph


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
        choices=['bc', 'gail', 'advil', 'vd'], default='bc')
    parser.add_argument("--config", help="config file path",
        default=None, type=str)
    parser.add_argument('--seed', default=0, type=int,
        help='seed')    
    parser.add_argument('--nframes', default=500, type=int,
        help='frames for test animation') 
    parser.add_argument('--nsamples', default=200, type=int,
        help='number of ray samples') 
    parser.add_argument('--graph', action='store_true',
        help='whether to mask the relative states based on a ConeVisibilityGraph')
    parser.add_argument('-d', default='./expert_data', type=str,
                       help='data directory')
    parser.add_argument('-o', default=None, type=str,
                       help='output directory')                   
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
        'nsamples':args.nsamples,
        'datadir':os.path.abspath(args.d),
        'graph':None,
        'outdir': opj('output',args.method,'loc%02i'%(args.loc)),
        'train_tracks':[0,1,2],
        'cv_tracks':[3],
        'test_tracks':[4],
    }
    if args.o:
        kwargs['outdir'] = args.o
    if args.graph:
        kwargs['graph'] = ConeVisibilityGraph(r=20, half_angle=120)
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
            "lr": tune.loguniform(1e-5, 1e-3),
            "weight_decay": tune.choice([0, 0.1]),
            "loss": tune.choice(['huber', 'mse']),
            "train_batch_size": tune.choice([16,32,64]),
            "deepsets_phi_hidden_n": tune.randint(1,5),
            "deepsets_phi_hidden_dim": tune.lograndint(8,65),
            "deepsets_latent_dim": tune.lograndint(8,129),
            "deepsets_rho_hidden_n": tune.randint(0,3),
            "deepsets_rho_hidden_dim": tune.lograndint(8,129),
            "deepsets_output_dim": tune.lograndint(4,129),
            "head_hidden_n": tune.randint(1,6),
            "head_hidden_dim": tune.lograndint(16,257),
            "head_final_activation": tune.choice(['sigmoid', None]),
        }
    else:
        raise NotImplementedError
    return ray_config

if __name__ == '__main__':
    kwargs = parse_args()
    
    # make prefix of output files

    if kwargs['config_path']:
        # load config
        with open(kwargs['config_path'], 'r') as cfg:
            config = json5.load(cfg)
        if not os.path.isdir(kwargs['outdir']):
            os.makedirs(kwargs['outdir'])   
        filestr = opj(kwargs['outdir'], basestr(**kwargs)) 
        if kwargs['ray']:
            filestr = kwargs['config_path'].replace('_config.json','')
        main(config, filestr=filestr, **kwargs)

    elif kwargs['ray'] and kwargs['train']:

        ray.shutdown() 
        ray.init(log_to_driver=False)

        def ray_train(config, datadir=None):
            full_config = get_full_config(config, kwargs['method'])
            main(full_config, filestr='exp', **kwargs)
        
        ray_config = get_ray_config(kwargs['method'])
        search = HyperOptSearch(ray_config, max_concurrent=8, metric='cv_loss',mode="min",)
        custom_scheduler = ASHAScheduler(metric='cv_loss', mode="min", grace_period=15)

        analysis = tune.run(
            ray_train, 
            #config=ray_config,
            search_alg=search,
            scheduler=custom_scheduler,
            local_dir=kwargs['outdir'],
            #resources_per_trial={"cpu": 2},
            time_budget_s=120*60,
            num_samples=kwargs['nsamples'],
        )
    elif kwargs['ray'] and kwargs['test']:
        analysis = Analysis(kwargs['outdir'], default_metric="cv_loss", default_mode="min")
        config = analysis.get_best_config()
        filepath = analysis.get_best_logdir()
        filestr = opj(filepath, 'exp')
        config_path = filestr+'_config.json'
        with open(config_path, 'r') as cfg:
            config = json5.load(cfg)
        print("Best ray experiment:", filepath)
        main(config, filestr=filestr, **kwargs)
    else:
        raise Exception('No valid config found')

    
    
    

    
