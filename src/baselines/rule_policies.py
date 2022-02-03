from stable_baselines3.common.base_class import BaseAlgorithm
from intersim.envs.intersimple import Intersimple
from typing import Tuple, Optional
import numpy as np

class PControllerPolicy(BaseAlgorithm):


    def __init__(self, env):
        """
        Initialize policy with pointer to environment it will run on
        """
        assert(isinstance(env, Intersimple), 'Environment is not an intersimple environment')
        self._env = env
        self.target_v = 8.94 # m/s
        self.attn_weight = 20
    
    def predict(self, observation: np.ndarray, *args, **kwargs):
        """
        Generate action, state from observation

        (But actually generate next action from underlying environment state)
        
        Args:
            observation (np.ndarray): instantaneous observation from environment

        Returns
            action (np.ndarray): action for controlled agent to take
            state (np.ndarray): hidden state for use in next prediction (null)
        """
        agent = self._env._agent
        ego_state = self._env._env.projected_state[agent].numpy() # (5,) tensor
        
        
        # relative_state = np.delete(self._env._env.relative_state[agent].numpy(), agent, axis=0) #(nv-1, 6) tensor
        
        # calculate front and left distances from ego

        # calculate relative speed in direction of position difference vector
        
        # calculate angle alpha and distance d of vehicle i from ego heading
        
        # attn[i] ~= exp( -(alpha[i])^2 - .01 * d[i] - .1 * vrel[i]


        # Proportional controller
        # action = (self.target_v  - self.attn_weight * attn.sum()) - ego_state[2]
        action = self.target_v  - ego_state[2]
        return action, None

class IDMRulePolicy(BaseAlgorithm):
    """
    IDMRulePolicy returns action predictions based on an IDM policy.

    The front car is chosen as the closer of:
        - closest car within a 45 degree half angle cone of the ego's heading
        - ''' after propagating the environment forward by `t_future' seconds with 
            current headings and velocities
    
    """

    def __init__(self, env: Intersimple, target_speed: float= 8.94, t_future:float=0):
        """
        Initialize policy with pointer to environment it will run on and target speed

        Args:
            env (Intersimple): intersimple environment which IDM runs on
            target_speed (float): target speed in roundabout (default: 8.94=20 mph)
            t_future (float): future time at which to compare closest
        """
        # assert(isinstance(env, Intersimple), 'Environment is not an intersimple environment')
        self._env = env

        
        assert(t_future >=0, 'negative target speed')
        self.t_future = t_future

        self.half_angle = 45 # degrees for finding car to follow

        # Default IDM parameters
        assert(target_speed>0, 'negative target speed')
        self.s_max = target_speed
        self.a_max = np.array([3.]) # nominal acceleration
        self.tau = 0.5 # desired time headway
        self.b_pref = 2.5 # preferred deceleration
        self.d_min = 1 #minimum spacing
        super().__init__()

    def predict(self, observation:np.ndarray, 
        *args, **kwargs) -> Tuple[np.ndarray,Optional[np.ndarray]]:
        """
        Generate action, state from observation

        (But actually generate next action from underlying environment state)
        
        Args:
            observation (np.ndarray): instantaneous observation from environment

        Returns
            action (np.ndarray): action for controlled agent to take
            state (np.ndarray): the index of the chosen vehicle for IDM
        """
        import pdb
        pdb.set_trace()
        agent = self._env._agent
        full_state = self._env._env.projected_state.numpy() #(nv, 5)
        ego_state = full_state[agent] # (5,)
        s = ego_state[2]
        xy = full_state[:,0:2] # (nv, 2)
        v = full_state[:,2:3] # (nv, 1)
        psi = full_state[:,2:3] # (nv, 1)


        d, r, i = self.get_ego_dr(agent, xy, v, psi)

        # propagate environment forward at constant velocity
        if self.t_future > 0:
            xy2 = xy + self.t_future * v * np.vstack((np.cos(psi[:,0]), np.sin(psi[:,0])))
            d2, r2, i2 = self.get_ego_dr(agent, xy2, v, psi)
            
            # choose closer vehicle (now vs imagined)
            if d2 < d:
                d, r, i = d2, r2, i2   
        
        if d == np.inf:
            d_des = 0
        else:
            d_des = self.d_min + self.tau * s + s * r / (2* (self.a_max*self.b_pref)**0.5 )
        action = self.a_max*(1 - (s/self.s_max)**4 - (d_des/d)**2)

        assert(action.shape==(1,))
        return action, i
    
    def get_ego_dr(self, agent:int, xy: np.ndarray, 
        v: np.ndarray, psi: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Return distance and relative speed of closest car within half angle from heading
        
        Args:
            agent (int): agent index
            xy (np.ndarray): (nv, 2) x and y positions
            v (np.ndarray): (nv, 1) velocity
            psi (np.ndarray): (nv, 1) heading angle

        Returns:
            d (float): distance to closest vehicle in cone
            r (float): relative speed between the two vehicles
            i (Union[None,np.ndarray]): (1,) array of closest vehicle index, or None 
        """
        nv, nxy = xy.shape
        nv2, nvel = v.shape
        nv3, npsi = psi.shape
        assert(nv==nv2==nv3)
        assert(nxy==2)
        assert(nvel==npsi==1)

        dxys = xy - xy[agent] # (nv, 2)
        ds = np.linalg.norm(dxys,axis=1) # (nv,)
        df = (dxys*np.hstack((np.cos(psi),np.sin(psi)))).sum(-1) # (nv, )
        dl = (dxys*np.hstack((-np.sin(psi), np.cos(psi)))).sum(-1) # (nv, )
        alpha = to_circle(np.arctan2(dl, df))

        val_idx = np.arange(nv)[(np.abs(alpha) < self.half_angle*np.pi/180 & np.arange(nv) != agent)]
        
        if len(val_idx)==0:
            i = None
            d = float('inf')
            r = float('inf')
        else:
            idx = np.argmin(ds[val_idx]) # closest car which meets requirements
            i = val_idx[idx]
            d = ds[i]
            r = v[i,0]-v[agent,0] 
            
        return d, r, i
    
def to_circle(x: np.ndarray) -> np.ndarray:
    """
    Casts x (in rad) to [-pi, pi)

    Args:
        x (np.ndarray): (*) input angle (radians)

    Returns:
        y (np.ndarray): (*) x cast to [-pi, pi)
    """
    y = np.remainder(x + np.pi, 2*np.pi) - np.pi
    return y