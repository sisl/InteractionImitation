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


