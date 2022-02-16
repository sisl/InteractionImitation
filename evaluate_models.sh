# eval_main inputs
# locations: List[Tuple[int,int]]= [(0,0)],
# method: str='expert', 
# policy_file: str='', 
# policy_kwargs: dict={},
# env: str='NRasterizedRouteIncrementingAgent', 
# env_kwargs: dict={},
# seed: int=0

# expert
python -m src.eval_main

# idm
python -m src.eval_main --method=idm

# options GAIL
python -m src.eval_main --method=ogail --policy_file='checkpoints/gail-options-setobs2-Feb15_18-49-05.pt' --env='NormalizedOptionsEvalEnv' --env_kwargs='{stop_on_collision:True}'

# options GAIL-PPO
python -m src.eval_main --method=ogail-ppo --policy_file='checkpoints/gail-ppo-options-setobs2-Feb15_22-05-38.pt' --env='NormalizedOptionsEvalEnv' --env_kwargs='{stop_on_collision:True}'

# behavior cloning
python -m src.eval_main --method=bc --policy_file='checkpoints/bc-intersimple-setobs2.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}'
