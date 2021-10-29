import torch
import numpy as np
from intersim.collisions import state_to_polygon

def feasible(env, plan, method='exact'):
    """Check if input profile is feasible given current `env` state."""
    # zero pad plan - Take (B, T) or (T,) np plan and convert it to (B, T, nv, 1) torch.Tensor
    plan = torch.tensor(plan)
    plan = plan.reshape(-1, plan.shape[-1])
    full_plan = torch.zeros(*plan.shape, env._env._nv, 1)
    full_plan[:, :, env._agent, 0] = plan

    # check_future_collisions_fast takes in B-list and outputs (B,) bool tensor
    if method=='circle':
        valid = check_future_collisions_fast(env, full_plan) 
    elif method=='ncircles':
        valid = check_future_collisions_ncircles(env, full_plan)
    elif method=='exact':
        valid = check_future_collisions_exact(env, full_plan)
    else:
        raise NotImplementedError('Invalid collision-checking method')

    return valid

def check_future_collisions_ncircles(env, actions, n_circles:int=2):
    """Checks whether `env._agent` would collide with other agents assuming `actions` as input.
    
    Vehicles are (over-)approximated by multiple circles.

    Args:
        env (gym.Env): current environment state
        actions (list of torch.Tensor): list of B (T, nv, adims) T-length action profiles
    Returns:
        feasible (torch.Tensor): tensor of shape (B,) indicating whether the respective action profiles are collision-free
    """
    assert n_circles >= 2
    B, (T, nv, _) = len(actions), actions[0].shape

    states = env._env.propagate_action_profile_vectorized(actions)
    assert states.shape == (B, T, nv, 5)
    centers = states[:, :, :, :2]
    psi = states[:, :, :, 3]
    lon = torch.stack([psi.cos(), psi.sin()],dim=-1) # (B, T, nv, 2)

    # offset between [-env._env.lengths+env._env.widths/2, env._env.lengths/2-env._env.widths/2]
    back = (-env._env._lengths/2+env._env._widths/2).unsqueeze(-1) # (nv, 1)
    length = (env._env._lengths-env._env._widths).unsqueeze(-1) # (nv, 1)
    diff_d =  back + length*(torch.arange(n_circles)/(n_circles-1)).unsqueeze(0)  # (nv, n_circles) 
    assert diff_d.shape == (nv, n_circles)
    
    offsets = diff_d[None, None, :, :, None] *  lon[:, :, :, None, :]
    assert offsets.shape == (B, T, nv, n_circles, 2)

    expanded_centers=centers.unsqueeze(-2) + offsets #(B, T, nv, n_circles, 2)
    assert expanded_centers.shape == (B, T, nv, n_circles, 2)
    agent_centers = expanded_centers[:,:,env._agent:env._agent+1,:,:] #(B, T, 1, n_circles, 2)
    ds = expanded_centers.reshape((B, T, nv*n_circles, 1, 2)) - agent_centers #(B, T, nv*nc,1, 2) - (B, T, 1, nc, 2) = (B, T, nv*nc, nc, 2) 

    distance = (ds**2).sum(-1).sqrt().reshape((B, T, nv, n_circles, n_circles)) # (B, T, nv, nc, nc)
    distance = torch.where(distance.isnan(), np.inf*torch.ones_like(distance), distance) # only collide with spawned agents
    distance[:, :, env._agent] = np.inf # cannot collide with itself
    assert distance.shape == (B, T, nv, n_circles, n_circles)

    radius = env._env._widths*np.sqrt(2) / 2
    min_distance = radius[env._agent] + radius
    min_distance = min_distance[None, None, :, None, None]
    assert min_distance.shape == (1, 1, nv, 1, 1)

    return (distance > min_distance).all(-1).all(-1).all(-1).all(-1)

def check_future_collisions_circle(env, actions):
    """Compute collision information for circular vehicle approximations

    Args:
        env (gym.Env): current environment state
        actions (list of torch.Tensor): list of B (T, nv, adims) T-length action profiles
    Returns:
        states (torch.Tensor): tensor of shape (B, T, nv, 5) of future states based on the action profiles
        collision_tensor (torch.Tensor): tensor of shape (B, T, nv) of bools indicating which plan collides with which vehicles in which time frame
            false: colliding, true: not colliding
    """
    B, (T, nv, _) = len(actions), actions[0].shape

    states = env._env.propagate_action_profile_vectorized(actions)
    assert states.shape == (B, T, nv, 5)

    distance = ((states[:, :, :, :2] - states[:, :, env._agent:env._agent+1, :2])**2).sum(-1).sqrt()
    distance = torch.where(distance.isnan(), np.inf*torch.ones_like(distance), distance) # only collide with spawned agents
    distance[:, :, env._agent] = np.inf # cannot collide with itself
    assert distance.shape == (B, T, nv)

    radius = (env._env._lengths**2 + env._env._widths**2).sqrt() / 2
    min_distance = radius[env._agent] + radius
    min_distance = min_distance.unsqueeze(0).unsqueeze(0)
    assert min_distance.shape == (1, 1, nv)

    collision_tensor = distance > min_distance
    assert collision_tensor.shape == (B, T, nv)
    return states, collision_tensor

def check_future_collisions_fast(env, actions):
    """Checks whether `env._agent` would collide with other agents assuming `actions` as input.
    
    Vehicles are (over-)approximated by single circles.

    Args:
        env (gym.Env): current environment state
        actions (list of torch.Tensor): list of B (T, nv, adims) T-length action profiles
    Returns:
        feasible (torch.Tensor): tensor of shape (B,) indicating whether the respective action profiles are collision-free
    """
    _, collision_tensor = check_future_collisions_circle(env, actions)
    return collision_tensor.all(-1).all(-1)

def check_future_collisions_exact(env, actions):
    """
        Checks whether `env._agent` would collide with other agents assuming `actions` as input.

    Args:
        env (gym.Env): current environment state
        actions (list of torch.Tensor): list of B (T, nv, adims) T-length action profiles
    Returns:
        feasible (torch.Tensor): tensor of shape (B,) indicating whether the respective action profiles are collision-free
    """
    # First check with simple circle collision check
    states, collision_tensor = check_future_collisions_circle(env, actions)
    (B, T, nv, _) = states.shape
    # For those that have colliding circles, check exactly
    colliding_mask = ~collision_tensor
    
    ego_states = states[:, :, env._agent:env._agent+1, :].expand(states.shape)
    assert ego_states.shape == states.shape

    # get dimensions
    lengths = env._env._lengths.expand(states.shape[:3])
    widths = env._env._widths.expand(states.shape[:3])
    ego_lengths = lengths[:, :, env._agent:env._agent+1].expand(lengths.shape)
    ego_widths = widths[:, :, env._agent:env._agent+1].expand(widths.shape)
    assert lengths.shape == widths.shape == ego_lengths.shape == ego_widths.shape == (B, T, nv)
    
    # For every collision instance between ego and other vehicle, check whether rectangles intersect
    exact_collisions = torch.zeros_like(collision_tensor[colliding_mask])
    for i, (ego_state, ego_length, ego_width, other_state, other_length, other_width) in enumerate(zip(
        ego_states[colliding_mask], ego_lengths[colliding_mask], ego_widths[colliding_mask],
        states[colliding_mask], lengths[colliding_mask], widths[colliding_mask]
    )):
        assert ego_state.shape == other_state.shape == (5,)
        assert ego_length.shape == ego_width.shape == other_length.shape == other_width.shape == ()
        p_ego = state_to_polygon(ego_state, ego_length, ego_width)
        p_other = state_to_polygon(other_state, other_length, other_width)
        exact_collisions[i] = p_ego.intersects(p_other)

    collision_tensor[colliding_mask] = ~exact_collisions
    return collision_tensor.all(-1).all(-1) 
