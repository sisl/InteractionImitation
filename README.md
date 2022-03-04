# InteractionImitation
Imitation Learning with the INTERACTION Dataset

## Getting started
Clone InteractionSimulator and pip install the module.
```
git clone https://github.com/sisl/InteractionSimulator.git
cd InteractionSimulator
pip install -e .
cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH
```
Install additional requirements
```
pip install -r requirements.txt
```
Copy INTERACTION Dataset files:
The INTERACTION dataset contains a two folders which should be copied into a folder called `./InteractionSimulator/datasets`: 
  - the contents of `recorded_trackfiles` should be copied to `./InteractionSimulator/datasets/trackfiles`
  - the contents of `maps` should be copied to `./InteractionSimulator/datasets/maps`

## Processing and saving expert demos
Once the repository has been set up, you need to generate two separate sets of expert demos for tracks 0-4. The first command generates true joint and individual states and actions, necessary for evaluating. The second command generates trajectory rollouts according to individual agent observations, which will be used in behavior cloning, along with gail/hail/shail discriminators.
```
python -m src.expert --locs='[DR_USA_Roundabout_FT]' --tracks='[0,1,2,3,4]'
python -m intersimple-expert-rollout-setobs2 --tracks='[0,1,2,3,4]'
```


## Tuning hyperparameters and training finalized models
To tune models, we use `ray[tune]` grid searches. You can run see the commands we used to train in the top half of `train_models.sh`. After training the models, configurations get saved in `best_configs/`. However, we note some better performance manually at earlier epochs of training, so we adjusted our `best_configs` manually.

After the `best_configs/` are set, we rereun each configuration with multiple seeds, the commands to do so are in the bottom half of `train_models.sh`. This saves different policy files to `test_policies/`.


## Evaluating models
To evaluate the learned policies, we rerun each model in particular setting, evaluate all our metrics, and average over different trained model seeds. The commands to do so are in `evaluate_models.sh`. 


## Package Structure
```
InteractionImitation
|- TODO
```

## Type Definitions
```
Demo: List[Trajectory]
Trajectory: List[Tuple[Observation, Action]] # single expert
Observation: Dict[
  'own_state': [x, y, v, psi, psidot],
  'relative_states': List[[xr, yr, vr, psir, psidotr]],
  'own_path': List[[xr, yr]], # fixed length, constant dt
  'map': Map, # relative
]
Action: Range[0, 1]
Policy: Union[
  Callable[[Observation], Action],
  Callable[[Observation, Action], probability],
]
Discriminator: Callable[[Observation, Action], value]
Map: Dictionary[...]
```
