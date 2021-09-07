#python -m intersimple.expert --env=IntersimpleReward --min_timesteps=200 --env_args='{agent:51}' --path='NormalizedIntersimpleExpert_IntersimpleRewardAgent51.pkl'
#python -m intersimple.expert --env=IntersimpleReward --min_timesteps=200 --env_args='{agent:51}' --policy_args='{mu:0.005}' --path='NormalizedIntersimpleExpert_IntersimpleRewardAgent51Mu.005.pkl'
python -m intersimple.expert --env=IntersimpleReward --min_timesteps=200 --env_args='{agent:51}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpert_IntersimpleRewardAgent51Mu.001.pkl' --video
