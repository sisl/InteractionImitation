# can add --skip_running if you've run before

python -m eval_experiments 
python -m eval_experiments --locations='[(0,4)]'
python -m eval_experiments --method idm
python -m eval_experiments --method idm --locations='[(0,4)]'
python -m eval_experiments --method bc --folder='test_policies/bc/expA'
python -m eval_experiments --method bc --folder='test_policies/bc/expB' --locations='[(0,4)]'
python -m eval_experiments --method gail --folder='test_policies/gail/expA'
python -m eval_experiments --method gail --folder='test_policies/gail/expB'--locations='[(0,4)]'
python -m eval_experiments --method hail
python -m eval_experiments --method hail --locations='[(0,4)]'
python -m eval_experiments --method shail
python -m eval_experiments --method shail --locations='[(0,4)]'