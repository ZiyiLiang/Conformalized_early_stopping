#!/bin/bash

# Parameters
N_DATA_LIST=(200 500 1000 2000 4000)
LR_LIST=(0.001)
EPOCH_LIST=(20 50)
SEED_LIST=$(seq 1 50)

# test job
#N_DATA_LIST=(200)
#LR_LIST=(0.001)
#EPOCH_LIST=(10)
#SEED_LIST=$(seq 1 2) 


# Slurm parameters
MEMO=12G                             # Memory required (12 GB)
TIME=00-00:30:00                    # Time required (60 m)
CORE=1                              # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs/multiclass"
mkdir -p $LOGS

OUT_DIR="results/multiclass"
mkdir -p $OUT_DIR

# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  for N_DATA in "${N_DATA_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do
      for EPOCH in "${EPOCH_LIST[@]}"; do
        JOBN="ndata"$N_DATA"_lr"$LR"_epoch"$EPOCH"_seed"$SEED
        OUT_FILE=$OUT_DIR"/"$JOBN".txt"
        COMPLETE=0
        #ls $OUT_FILE
        if [[ -f $OUT_FILE ]]; then
        COMPLETE=1
        fi

        if [[ $COMPLETE -eq 0 ]]; then
        # Script to be run
        SCRIPT="exp_oc.sh $N_DATA $LR $EPOCH $SEED"
        # Define job name for this chromosome
        OUTF=$LOGS"/"$JOBN".out"
        ERRF=$LOGS"/"$JOBN".err"
        # Assemble slurm order for this job
        ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
        # Print order
        echo $ORD
        # Submit order
        $ORD
        fi
      done
    done
  done
done
