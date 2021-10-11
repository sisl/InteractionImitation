
import stable_baselines3 as sb3
from intersim.envs.intersimple import NRasterized


model_name = 'gail_image_singleagent_nocollision'
model = sb3.PPO.load(model_name)

env = NRasterized(stop_on_collision=False, width=36, height=36, m_per_px=2, agent=51)

obs = env.reset()
i=0
while True and i < 600:
    i+=1
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='post')
    if done:
        break

env.close(filestr='render/'+model_name)