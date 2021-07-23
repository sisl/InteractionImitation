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

## Processing, saving, and loading expert demos
Once the repository has been set up, you can process and save expert track demonstrations with:
```
python src/expert_data.py --loc [LOCNUM] --track [TRACKNUM]
```
You can (and should) process all tracks at once at location 0 with:
```
python src/expert_data.py --all-tracks
```

You can then train a default behavior cloning policy with the following. Be sure to check help for main.py for running options.
```
python src/main.py --train
```
You can then test the learned policy with the following, and see the animation file in `output/`:
```
python src/main.py --test
```


You can load the experts actions manually
```
from src import expert_data
observations, actions = expert_data.load_expert_data(loc = [LOCNUM], track = [TRACKNUM])
for (s, a) in zip (observations, actions):
    # do some imitation learning
```


## Package Structure
```
InteractionImitation
|- demos
|- algorithms
   |- BC
   |- AdVIL
|- nets
   |- Encoder
   |- DeepSet
   |- Decoder
|- policies
|- discriminators
|- demo_generators
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
