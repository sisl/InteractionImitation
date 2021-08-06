#!/bin/sh

# python experiments/experiment.py --method vd --train --ray -d expert_data/reg -o output/vd/loc00/reg --nsamples 400
# python experiments/experiment.py --test --ray --method vd -d expert_data/normal -o output/vd/loc00/normal --nframes 1000
python experiments/experiment.py --train --method vd --config config/value_dice.json5