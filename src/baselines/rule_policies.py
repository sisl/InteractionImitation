from stable_baselines3.common.base_class import BaseAlgorithm
from intersim.envs.intersimple import Intersimple, NormalizedActionSpace
from typing import Tuple, Optional, List
import numpy as np

class PControllerPolicy(BaseAlgorithm):


    def __init__(self, env):
        """
        Initialize policy with pointer to environment it will run on
        """
        assert isinstance(env, Intersimple), 'Environment is not an intersimple environment'
        self._env = env
        self.target_v = 8.94 # m/s
        self.attn_weight = 20

    # BaseAlgorithm abstract methods
    def _setup_model(self):
        return None
    def learn(self, *args, **kwargs):
        return self

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

    def __init__(self, env: Intersimple,
        target_speed:float= 8.94,
        t_future:List[float]=[0., 1., 2., 3.],
        half_angle:float=60.):
        """
        Initialize policy with pointer to environment it will run on and target speed

        Args:
            env (Intersimple): intersimple environment which IDM runs on
            target_speed (float): target speed in roundabout (default: 8.94=20 mph)
            t_future (List[float]): list of future time at which to compare closest vehicle
            half_angle (float): half angle to look inside for closest vehicle
        """

        self._env = env
        self.t_future = t_future
        self.half_angle = half_angle

        # Default IDM parameters
        assert target_speed>0, 'negative target speed'
        self.v_max = target_speed
        self.a_max = np.array([3.]) # nominal acceleration
        self.tau = 0.5 # desired time headway
        self.b_pref = 2.5 # preferred deceleration
        self.d_min = 1 #minimum spacing

        # for np.remainder nan warnings
        np.seterr(invalid='ignore')

    # BaseAlgorithm abstract methods
    def _setup_model(self):
        return None
    def learn(self, *args, **kwargs):
        return self

    def predict(self, observation:np.ndarray,
        *args, **kwargs) -> Tuple[np.ndarray, None]:
        """
        Predict action, state from observation

        (But actually generate next action from underlying environment state)

        Args:
            observation (np.ndarray): instantaneous observation from environment

        Returns
            action (np.ndarray): action for controlled agent to take
            state (None): None (hidden state for a recurrent policy)
        """
        return self.forward(observation, *args, **kwargs), None

    def forward(self, *args, **kwargs) -> np.ndarray:
        """
        Generate action from underlying environment

        Returns
            action (np.ndarray): action for controlled agent to take
        """
        agent = self._env._agent
        state = self._env._env.state.numpy()
        full_state = self._env._env.projected_state.numpy() #(nv, 5)
        ego_state = full_state[agent] # (5,)
        v_ego = ego_state[2]
        # xy = full_state[:,0:2] # (nv, 2)
        v = full_state[:,2:3] # (nv, 1)
        # psi = full_state[:,3:4] # (nv, 1)

        length = 20
        step = 0.5
        x, y = self._env._env._generate_paths(delta=step, n=length/step, is_distance=True)
        heading = to_circle(np.arctan2(np.diff(y), np.diff(x)))
        velocities = state[:,1]

        # something like this could be done to also take future proximity of vehicles to ego path into account
        # time_horizon = np.array(range(3))
        # predictions = state[:,0:1] + np.outer(state[:,1], time_horizon)

        paths = np.stack([x[:,:-1],y[:,:-1], heading], axis=1) # (nv x 3 x (path_length-1))
        ego_path = paths[agent:agent+1] # (1 x 3 x path_length-1)

        # (x,y,phi) of all vehicles
        poses = np.expand_dims(full_state[:, [0,1,3]], 2) # (nv x 3 x 1)

        diff = ego_path - poses
        diff[:, 2, :] = to_circle(diff[:, 2, :])

        # Test if position and heading angle are close for some point on the future vehicle track
        max_pos_error = 1
        pos_close = np.sum(diff[:, 0:2, :]**2, 1) <= max_pos_error**2 # (nv x path_length-1)
        max_deg_error = 20
        heading_close = np.abs(diff[:, 2, :]) <= 20 * np.pi / 180 # (nv x path_length-1)
        # For all vehicles get the path points where they are close to the ego path
        close = np.logical_and(pos_close, heading_close) # (nv x path_length-1)
        close[agent, :] = False # exclude ego agent

        leader = agent
        min_idx = np.Inf
        # Determine vehicle that is closest to ego in terms of path coordinate
        for veh_id in range(len(close)):
            path_idx = np.nonzero(close[veh_id])[0]
            # veh_id is never close to agent
            if len(path_idx) == 0:
                continue
            # first path index where veh_id is close to agent
            elif path_idx[0] < min_idx:
                leader = veh_id
                min_idx = path_idx[0]

        # alternative vectorized code
        # def findfirst(a):
        #     idx = np.argwhere(a)
        #     if len(idx) == 0:
        #         return np.NaN
        #     else:
        #         return float(idx[0]) # float conversion, to get a numpy array of dtype=float64
        # d = np.apply_along_axis(findfirst, 1, close) # (nv)
        # if np.all(np.isnan(d)):
        #     leader = agent
        # else:
        #     leader = np.nanargmin(d)
        #     path_idx = d[leader]
        #     min_idx = np.sqrt(np.sum(diff[leader, 0:2, path_idx]**2))


        if leader != agent:
            # distance along ego path to point with closest distance
            d = step * min_idx
            # add distance from ego path point with closest distance to actual vehicle position
            d += np.sqrt(np.sum(diff[leader, 0:2, min_idx]**2))

            # Update environment interaction graph with leader
            self._env._env._graph._neighbor_dict={agent:[leader]}

            delta_v = v_ego - v[leader, 0]
            d_des = self.d_min + self.tau * v_ego + v_ego * delta_v / (2* (self.a_max*self.b_pref)**0.5 )
            d_des = max(d_des, self.d_min)
        else:
            d = np.Inf
            d_des = self.d_min
            self._env._env._graph._neighbor_dict={}

        assert (d_des>= self.d_min)
        action = self.a_max*(1 - (v_ego/self.v_max)**4 - (d_des/d)**2)

        # normalize action to range if env is a NormalizedActionSpace
        if isinstance(self._env, NormalizedActionSpace):
            action = self._env._normalize(action)

        assert action.shape==(1,)
        return action


    def get_ego_dr(self, agent:int, xy: np.ndarray,
        v: np.ndarray, psi: np.ndarray) -> Tuple[float, float, Optional[int]]:
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
            i (Optional[int]): index of closest vehicle, or None
        """
        nv, nxy = xy.shape
        nv2, nvel = v.shape
        nv3, npsi = psi.shape
        assert nv==nv2==nv3
        assert nxy==2
        assert nvel==npsi==1

        dxys = xy - xy[agent] # (nv, 2)
        ds = np.linalg.norm(dxys,axis=1) # (nv,)
        df = (dxys*np.hstack((np.cos(psi),np.sin(psi)))).sum(-1) # (nv, )
        dl = (dxys*np.hstack((-np.sin(psi), np.cos(psi)))).sum(-1) # (nv, )
        alpha = to_circle(np.arctan2(dl, df))

        val_idx = np.arange(nv)[(np.abs(alpha) < self.half_angle*np.pi/180) & (np.arange(nv) != agent)]

        if len(val_idx)==0:
            i = None
            d = float('inf')
            r = float('inf')
        else:
            idx = np.argmin(ds[val_idx]) # closest car which meets requirements
            i = int(val_idx[idx])
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