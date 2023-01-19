#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate py37

python3 exp_mc.py $1 $2 $3 $4
