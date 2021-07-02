# InteractionImitation
Imitation Learning with the INTERACTION Dataset

## Getting started
Clone InteractionSimulator and pip install the module.
```
git clone https://github.com/sisl/InteractionSimulator.git
cd InteractionSimulator
git checkout v0.0.1
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
