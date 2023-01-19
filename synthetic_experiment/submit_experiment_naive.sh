#!/bin/bash

# Parameters
NCAL_LIST=(10 50 100)
LR_LIST=(0.1 0.5 1)
EPOCH_LIST=(50 100 200)
OPTIM_LIST=("SGD" "ADAM")
STD_LIST=(2)
FEATURE_LIST=(2 50)
SEED_LIST=(1)

# Slurm parameters
MEMO=12G                             # Memory required (3 GB)
TIME=00-01:00:00                    # Time required (30 m)
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

# Loop over configurations and chromosomes
for SEED in $SEED_LIST; do
  echo "seed:"$SEED
  for N_CAL in "${NCAL_LIST[@]}"; do
    echo "cal:"$N_CAL
    for LR in "${LR_LIST[@]}"; do
      echo "LR:"$LR
      for EPOCH in "${EPOCH_LIST[@]}"; do
        echo "epoch:"$EPOCH
        for OPTIM in "${OPTIM_LIST[@]}"; do
          echo "optim:"$OPTIM
          for STD in "${STD_LIST[@]}"; do
            echo "std:"$STD    
            for N_FEATURE in "${FEATURE_LIST[@]}"; do   
                JOBN="naive/""ncal"$N_CAL"_lr"$LR"_epoch"$EPOCH"_optim"$OPTIM"_seed"$SEED"_std"$STD"_nfeature"$N_FEATURE
                OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                COMPLETE=0
                #ls $OUT_FILE
                
		if [[ -f $OUT_FILE ]]; then
                COMPLETE=1
		echo "Completed"
                fi

                if [[ $COMPLETE -eq 0 ]]; then
                # Script to be run
                SCRIPT="$HOME/CES/Conformalized_early_stopping/synthetic_experiment/experiment_naive.sh $N_CAL $LR $EPOCH $OPTIM $SEED $STD $N_FEATURE"
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
    done
  done
done
