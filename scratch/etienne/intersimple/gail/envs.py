import gym
import numpy as np
from intersim.envs.intersimple import RandomLocation, RandomAgent, RewardVisualization, Reward, \
    ImageObservationAnimation, RasterizedRoute, NObservations, RasterizedObservation, \
    NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple

class RasterizedSpeed:

    def __init__(self, max_speed=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels, height, width = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels+1, height, width),
            dtype=np.uint8
        )
        self._max_speed = max_speed

    def _simple_obs(self, intersim_obs, intersim_info):
        img = super()._simple_obs(intersim_obs, intersim_info)

        ego_speed = intersim_obs['state'][self._agent, 2]
        scaled_speed = (255 * ego_speed) // self._max_speed
        speed_layer = scaled_speed * np.ones_like(img[:1], dtype=np.uint8)
        speed_layer = speed_layer.clamp(0, 255)
        
        obs = np.concatenate((img, speed_layer), axis=0)
        return obs

class NRasterizedRouteSpeedRandomAgentLocation(RandomLocation, RandomAgent, RewardVisualization,
        Reward, ImageObservationAnimation, RasterizedRoute, RasterizedSpeed, NObservations, RasterizedObservation,
        NormalizedActionSpace, ActionVisualization, InteractionSimulatorMarkerViz, ImitationCompat, Intersimple):
    pass