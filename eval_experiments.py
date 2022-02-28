import os
from src.eval_main import eval_main
from src.evaluation.utils import load_and_average

def main(method:str='expert', folder:str=None, locations=[(0,0)], skip_running=False):
    
    policy_kwargs = {}
    if method in ['expert', 'idm']:
        env, env_kwargs ='NRasterizedRouteIncrementingAgent', {}
    elif method in ['bc','gail']:
        env='NormalizedContinuousEvalEnv' 
        env_kwargs={'stop_on_collision':True, 'max_episode_steps':1000}
    elif method in ['hail']:
        env = 'NormalizedOptionsEvalEnv'
        env_kwargs={'stop_on_collision':True, 'max_episode_steps':1000}
    elif method in ['shail']:
        env = 'NormalizedSafeOptionsEvalEnv'
        env_kwargs={'stop_on_collision':True, 'max_episode_steps':1000}
    else:
        raise NotImplementedError

    files = ['']

    if folder is not None:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        print('%i folders found in %s folder' %(len(files), folder))

    if not skip_running:
        for policy_file in files:
            # run metrics on that file
            outbase = eval_main(locations=locations,
                                method=method, 
                                policy_file=policy_file, 
                                policy_kwargs=policy_kwargs,
                                env=env, 
                                env_kwargs=env_kwargs)
        outfolder = os.path.dirname(outbase)
    else:
        locstr = 'loc_'+'_'.join([f'r{ro}t{tr}' for (ro,tr) in locations])
        if folder is None:
            outfolder = os.path.join('out',method,locstr)
        else:
            path_items = folder.split('/')
            outfolder = os.path.join('out', '/'.join(path_items[1:]), locstr)
    
    # load metrics from save_path
    average_metrics = load_and_average(outfolder)
    if method in ['expert', 'idm']:
        latex_print(average_metrics, light=True)
    else:
        latex_print(average_metrics)

def latex_print(am, light=False):
    """
    print latex line

    am (Dict[str,tuple]): dict mapping metric_name to (mean, std)
    """

    print('success rate, distance travelled, RWSE_10, |DeltaV|, AccelJSD')
    if light:
        if 'rwse_10s' in am.keys():
            print("%2.1f& %2.1f & %1.2f & %2.1f& "
            "%0.3f \\\\" %( 100*am['success rate'][0], am['mean travel distance'][0], am['rwse_10s'][0],
                            am['average absolute average velocity'][0],am['acceleration distribution divergence'][0] ))
            return


        print("%2.1f& %2.1f & $---$ & $---$ & "
            "$---$ \\\\" %( 100*am['success rate'][0], am['mean travel distance'][0]))
        return
    
    print("%2.1f \\scriptstyle\\pm %2.1f & %2.1f \\scriptstyle\\pm %2.1f & "
        "%1.2f \\scriptstyle\\pm %1.2f & %2.1f \\scriptstyle\\pm %1.1f & "
        "%0.3f \\scriptstyle\\pm %0.3f \\\\" %( 100*am['success rate'][0], 100*am['success rate'][1],
               am['mean travel distance'][0] , am['mean travel distance'][1]  ,
               am['rwse_10s'][0] , am['rwse_10s'][1]  ,
               am['average absolute average velocity'][0] , am['average absolute average velocity'][1]  ,
               am['acceleration distribution divergence'][0] , am['acceleration distribution divergence'][1]  ))

if __name__=='__main__':
    import fire 
    fire.Fire(main)