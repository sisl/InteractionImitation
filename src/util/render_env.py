import stable_baselines3 as sb3
import intersim 
from src.gail.options import RenderOptions
from tqdm import tqdm

def render_env(model_name='gail_image_multiagent_nocollision', env='NRasterizedRoute', max_frames=600, options=False, options_list=None,
    **env_kwargs):
    """
    Render a video from an model, agent, and environment
    Args:
        model_name (str): name of the model
        environment (str): gym environment class to render environment on
    """

    model = sb3.PPO.load(model_name)
    Env = intersim.envs.intersimple.__dict__[env]
    
    print(f'Rendering environment with \'{model_name}\' policy')
    if not options:
        env = Env(**env_kwargs)
        obs = env.reset()
        for i in tqdm(range(max_frames)):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render(mode='post')
            if done:
                break
    else:
        assert options_list, "No option list specified"
        env = RenderOptions(Env(**env_kwargs), options=options_list)
        with tqdm(total=max_frames) as pbar:
            for i, s in enumerate(env.sample_ll(model)):
                pbar.update(1)
                if s['dones'] or i >= max_frames:
                    break
    env.close(filestr='render/'+model_name) 

if __name__ == '__main__':
    import fire
    fire.Fire(render_env)