#!/bin/bash

# Parameters
N_CAL_LIST=(10 20 30 50 70 90)
LR_LIST=(0.1, 0.5, 0.8, 1)
EPOCH_LIST=(50, 100, 200)
OPTIM_LIST=("SGD" "ADAM")
STD_LIST = (2, 4)
FEATURE_LIST = (2, 10, 30, 50, 100)
SEED_LIST=$(seq 1 10)

fi

# Slurm parameters
MEMO=3G                             # Memory required (3 GB)
TIME=00-00:30:00                    # Time required (30 m)
CORE=1                              # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME
#ORDP="sbatch --account=sesia_658 --partition=sesia,shared --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs"
mkdir -p $LOGS
mkdir -p $LOGS"/naive"

OUT_DIR="results"
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR"/naive"
"results/naive" + "/" + "_ncal"+str(n_cal) + "_lr" + str(lr) + "_epoch" + str(n_epoch) +\
                 "_optim" + str(optimizer) + "_seed" + str(random_state) + "_std" + str(std) +\
                  "_nfeature" + str(n_features)
# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  for N_CAL in "${N_CAL_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do
      for EPOCH in "${EPOCH_LIST[@]}"; do
        for OPTIM in "${OPTIM_LIST[@]}"; do
          for STD in "${STD_LIST[@]}"; do
            for N_FEATURE in "${FEATURE_LIST[@]}"; do
                JOBN="naive/""_ncal"$N_CAL"_lr"$LR"_epoch"$EPOCH"_optim"$OPTIM"_seed"$SEED"_std"$STD"_nfeature"$N_FEATURE
                OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                COMPLETE=0
                #ls $OUT_FILE
                if [[ -f $OUT_FILE ]]; then
                COMPLETE=1
                fi

                if [[ $COMPLETE -eq 0 ]]; then
                # Script to be run
                SCRIPT="experiment_naive.sh $N_CAL $LR $EPOCH $OPTIM $STD $N_FEATURE $SEED"
                # Define job name for this chromosome
                OUTF=$LOGS"/"$JOBN".out"
                ERRF=$LOGS"/"$JOBN".err"
                # Assemble slurm order for this job
                ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
                # Print order
                echo $ORD
                # Submit order
                #$ORD
                # Run command now
                #./$SCRIPT
                fi
            done
          done
        done
      done
    done
  done
done
