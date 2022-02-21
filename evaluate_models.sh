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

# behavior cloning
python -m src.eval_main --method=bc --policy_file='checkpoints/bc-intersimple-setobs2.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=0
python -m src.eval_main --method=bc --policy_file='checkpoints/bc-intersimple-setobs2.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=1
python -m src.eval_main --method=bc --policy_file='checkpoints/bc-intersimple-setobs2.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=2
python -m src.eval_main --method=bc --policy_file='checkpoints/bc-intersimple-setobs2.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=3
python -m src.eval_main --method=bc --policy_file='checkpoints/bc-intersimple-setobs2.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=4
python -m src.evaluation.utils load_and_average out/bc

# GAIL
python -m src.eval_main --method=gail --policy_file='checkpoints/gail-intersimple-setobs2-03-02-22.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=0
python -m src.eval_main --method=gail --policy_file='checkpoints/gail-intersimple-setobs2-03-02-22.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=1
python -m src.eval_main --method=gail --policy_file='checkpoints/gail-intersimple-setobs2-03-02-22.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=2
python -m src.eval_main --method=gail --policy_file='checkpoints/gail-intersimple-setobs2-03-02-22.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=3
python -m src.eval_main --method=gail --policy_file='checkpoints/gail-intersimple-setobs2-03-02-22.pt' --env='NormalizedContinuousEvalEnv' --env_kwargs='{stop_on_collision:True}' --seed=4
python -m src.evaluation.utils load_and_average out/gail

# options GAIL
python -m src.eval_main --method=ogail --policy_file='checkpoints/gail-options-setobs2-Feb15_18-49-05.pt' --env='NormalizedOptionsEvalEnv' --env_kwargs='{stop_on_collision:True}'

# options GAIL-PPO
python -m src.eval_main --method=ogail-ppo --policy_file='checkpoints/gail-ppo-options-setobs2-Feb15_22-05-38.pt' --env='NormalizedOptionsEvalEnv' --env_kwargs='{stop_on_collision:True}'

# SHAIL
python -m src.eval_main --method=sgail --policy_file='checkpoints/sgail-options-setobs2.pt' --env='NormalizedSafeOptionsEvalEnv' --env_kwargs='{stop_on_collision:True,max_episode_steps:1000}'

# SHAIL-PPO
python -m src.eval_main --method=sgail-ppo --policy_file='checkpoints/sgail-ppo-options-setobs2-17-02-2022.pt' --env='NormalizedSafeOptionsEvalEnv' --env_kwargs='{stop_on_collision:True,max_episode_steps:1000}'
