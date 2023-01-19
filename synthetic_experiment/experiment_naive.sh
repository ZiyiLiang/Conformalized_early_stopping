#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate py37
python $HOME/CES/Conformalized_early_stopping/synthetic_experiment/experiment_naive.py $1 $2 $3 $4 $5 $6 $7
