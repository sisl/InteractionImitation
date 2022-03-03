# Hyperparameter tuning runs (will run search defined in files)
# Can add --epochs kwarg to change number of training epochs
# shail-experiment.py will generate the hail config aswell

# Experiment A
python bc-experiment.py --train A
python gail-experiment.py --train A
python shail-experiment.py --train A 

# Experiment B
python bc-experiment.py --train B
python gail-experiment.py --train B
python shail-experiment.py --train B 

# Retrain best models with 5 seeds
# Can add --test_seeds and --test_cpus kwargs to change number of seeds, allocate more memory 

# Experiment A
python bc-experiment.py --test best_configs/bc_expA.json
python gail-experiment.py --test best_configs/gail_expA.json
python shail-experiment.py --train best_configs/hail_expA.json
python shail-experiment.py --train best_configs/shail_expA.json

# Experiment B
python bc-experiment.py --test best_configs/bc_expB.json
python gail-experiment.py --test best_configs/gail_expB.json
python shail-experiment.py --train best_configs/hail_expB.json
python shail-experiment.py --train best_configs/shail_expB.json