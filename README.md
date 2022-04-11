# InteractionImitation
Imitation Learning with the [Interaction Dataset](https://interaction-dataset.com/) via the [InteractionSimulator](https://github.com/sisl/InteractionSimulator) gym environments.

Code for "[SHAIL: Safety-Aware Hierarchical Adversarial Imitation Learning for Autonomous Driving in Urban Environments](https://arxiv.org/abs/2204.01922)".
If you find this repository useful, please cite the paper:

```
@article{jamgochian2022shail,
  author = {Arec Jamgochian and Etienne Buehrle and Johannes Fischer and Mykel J. Kochenderfer},
  title = {{SHAIL}: Safety-Aware Hierarchical Adversarial Imitation Learning for Autonomous Driving in Urban Environments},
  journal = {arXiv:2204.01922 [cs]},
  year = {2022}
}
```

## Getting started
Clone the `InteractionSimulator` with the `shail` tag and pip install the module.
```
git clone --branch shail https://github.com/sisl/InteractionSimulator.git
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
Once the repository has been set up, you need to generate two separate sets of expert demos for tracks 0-4. The first command generates true joint and individual states and actions necessary for evaluating, saving them in `expert_data/`. The second command generates trajectory rollouts according to individual agent observations, which is later used as expert data for the learning models.
```
python -m src.expert --locs='[DR_USA_Roundabout_FT]' --tracks='[0,1,2,3,4]'
python -m intersimple-expert-rollout-setobs2 --tracks='[0,1,2,3,4]'
```


## Tuning hyperparameters and training finalized models
To tune models, we use `ray[tune]` grid searches. You can run see the commands we used to train in the top half of `train_models.sh`, as well as the hyperparameters we search over in `bc-experiment.py`, `gail-experiment.py`, and `shail-experiment.py`. After training the models, configurations get saved in `best_configs/` (the best SHAIL confg gets copied to a HAIL config, with the appropriate environment parameters changed for ablation). However, upon manual inspection of the training runs, we note some better performance than the automatically-set configs at earlier epochs, so we adjust the `best_configs` manually.

After the `best_configs/` are set, we rerun each configuration with multiple seeds. The commands to do so are in the bottom half of `train_models.sh`. This saves different learned policy files to `test_policies/`.


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
