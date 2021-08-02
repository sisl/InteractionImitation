#!/bin/sh

python experiments/experiment.py --ray --train -d ./expert_data/base
python experiments/experiment.py --ray --test -d ./expert_data/base --nframes 1000
python experiments/experiment.py --ray --train -d ./expert_data/reg
python experiments/experiment.py --ray --test -d ./expert_data/reg --nframes 1000
python experiments/experiment.py --ray --train -d ./expert_data/reg_graph --graph
python experiments/experiment.py --ray --test -d ./expert_data/reg_graph --graph --nframes 1000

