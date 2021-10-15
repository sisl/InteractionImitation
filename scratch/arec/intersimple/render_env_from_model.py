
import stable_baselines3 as sb3
from intersim.envs.intersimple import NRasterized


def render_env(model_name='gail_image_multiagent_nocollision', agent=51, environment=NRasterized):
    """
    Render a video from an model, agent, and environment
    Args:
        model_name (str): name of the model
        agent (int): agent to start the video from
        environment (gym.Env): gym environment class to render environment on
    """

    model = sb3.PPO.load(model_name)

    env = environment(stop_on_collision=False, width=36, height=36, m_per_px=2, agent=agent)

    obs = env.reset()
    i=0
    while True and i < 600:
        i+=1
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='post')
        if done:
            break

    env.close(filestr='render/'+model_name+'_agent%i'%(agent)) 

def render_options_env(model_name='gail_image_multiagent_nocollision', agent=51, environment=NRasterized):
    """
    Render a video from an model, agent, and environment
    Args:
        model_name (str): name of the model
        agent (int): agent to start the video from
        environment (gym.Env): gym environment class to render environment on
    """

    model = sb3.PPO.load(model_name)

    env = environment(stop_on_collision=False, width=36, height=36, m_per_px=2, agent=agent)

    obs = env.reset()
    i=0
    while True and i < 600:
        i+=1
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='post')
        if done:
            break

    env.close(filestr='render/'+model_name+'_agent%i'%(agent)) 

if __name__ == '__main__':
    import fire
    fire.Fire(render_env)