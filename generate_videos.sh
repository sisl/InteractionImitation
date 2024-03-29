# can add --skip_running if you've already run the saved policies through the test environments and have appropriate
# metrics in the out folder. Doing so will generate average metrics quickly.

# Experiment A
python -m eval_experiments 
python -m eval_experiments --method expert_agent --save_videos --first_seed_only
python -m eval_experiments --method idm --save_videos --first_seed_only
python -m eval_experiments --method bc --folder='test_policies/bc/expA' --save_videos --first_seed_only
python -m eval_experiments --method gail --folder='test_policies/gail/expA' --save_videos --first_seed_only
python -m eval_experiments --method hail --folder='test_policies/hail/expA' --save_videos --first_seed_only
python -m eval_experiments --method shail --folder='test_policies/shail/expA' --save_videos --first_seed_only

# Experiment B
python -m eval_experiments --locations='[(0,4)]' 
python -m eval_experiments --method expert_agent --locations='[(0,4)]' --save_videos --first_seed_only
python -m eval_experiments --method idm --locations='[(0,4)]' --save_videos --first_seed_only
python -m eval_experiments --method bc --folder='test_policies/bc/expB' --locations='[(0,4)]' --save_videos --first_seed_only
python -m eval_experiments --method gail --folder='test_policies/gail/expB' --locations='[(0,4)]' --save_videos --first_seed_only
python -m eval_experiments --method hail --folder='test_policies/hail/expB' --locations='[(0,4)]' --save_videos --first_seed_only
python -m eval_experiments --method shail --folder='test_policies/shail/expB' --locations='[(0,4)]' --save_videos --first_seed_only
