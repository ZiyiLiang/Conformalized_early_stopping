#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate py37

python3 exp_oc_cifar.py $1 $2 $3 $4
