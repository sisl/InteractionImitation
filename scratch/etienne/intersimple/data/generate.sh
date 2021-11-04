#python -m expert --env=IntersimpleReward --min_timesteps=200 --env_args='{agent:51}' --path='NormalizedIntersimpleExpert_IntersimpleRewardAgent51.pkl'
#python -m expert --env=IntersimpleReward --min_timesteps=200 --env_args='{agent:51}' --policy_args='{mu:0.005}' --path='NormalizedIntersimpleExpert_IntersimpleRewardAgent51Mu.005.pkl'
#python -m expert --env=IntersimpleReward --min_timesteps=200 --env_args='{agent:51}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpert_IntersimpleRewardAgent51Mu.001.pkl'
#python -m expert --env=NRasterized --min_timesteps=200 --env_args='{agent:51,width:36,height:36,m_per_px:2}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001_NRasterizedAgent51w36h36mppx2.pkl'
#python -m expert --env=NRasterized --min_timesteps=200 --env_args='{agent:51,width:36,height:36,m_per_px:2}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001_NRasterizedAgent51w36h36mppx2.pkl'
#python -m expert --env=NRasterized --min_timesteps=3000 --video --env_args='{width:36,height:36,m_per_px:2}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001_NRasterizedRandomAgentw36h36mppx2.pkl'
#python -m expert --env=NRasterizedRandomAgent --min_timesteps=200 --env_args='{width:36,height:36,m_per_px:2}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001_NRasterizedRandomAgentw36h36mppx2.pkl'
#python -m expert --env=NRasterizedRandomAgent --min_timesteps=10000 --env_args='{width:36,height:36,m_per_px:2}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001N10000_NRasterizedRandomAgentw36h36mppx2.pkl'
#python -m expert --env=NRasterizedRouteRandomAgent --min_timesteps=10000 --env_args='{width:70,height:70,m_per_px:1}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001N10000_NRasterizedRouteRandomAgentw70h70mppx1.pkl'
#python -m expert --env=NRasterizedRouteRandomAgentLocation --min_timesteps=100000 --env_args='{width:70,height:70,m_per_px:1}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001N100000_NRasterizedRouteRandomAgentLocationw70h70mppx1.pkl'
#python -m expert --env=NRasterizedRouteRandomAgentLocation --min_timesteps=100000 --env_args='{width:70,height:70,m_per_px:1,map_color:128}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001N100000_NRasterizedRouteRandomAgentLocationw70h70mppx1mapc128.pkl'
#python -m expert --env=NRasterizedRouteSpeedRandomAgentLocation --min_timesteps=10000 --env_args='{width:70,height:70,m_per_px:1,map_color:128,mu:0.001}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001N10000_NRasterizedRouteSpeedRandomAgentLocationw70h70mppx1mapc128mu.001.pkl'
#python -m data.expert --env=NRasterizedRouteSpeedRandomAgentLocation --min_timesteps=10000 --env_args='{width:70,height:70,m_per_px:1,map_color:128,mu:0.001,skip_frames:5}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001N10000_NRasterizedRouteSpeedRandomAgentLocationw70h70mppx1mapc128mu.001skip5.pkl'
python -m data.expert --env=TLNRasterizedRouteRandomAgentLocation --min_timesteps=10000 --env_args='{width:70,height:70,m_per_px:1,mu:0.001,random_skip:True,max_episode_steps:50}' --policy_args='{mu:0.001}' --path='NormalizedIntersimpleExpertMu.001N10000_TLNRasterizedRouteRandomAgentLocationw70h70mppx1mu.001rskips50.pkl'
