#!/bin/bash

# Parameters
CONF=1


if [[ $CONF == 1 ]]; then
  DATA_LIST=("concrete" "bike" "bio")
  METHOD_LIST=("theory" "naive" "benchmark" "ces")
  #METHOD_LIST=("naive" "benchmark")
  N_LIST=(500 1000)
  N_FEAT_LIST=(100)
  NOISE_LIST=(1)
  LR_LIST=(0.001)
  WD_LIST=(0)
  SEED_LIST=$(seq 1 25)

elif [[ $CONF == 10 ]]; then
  DATA_LIST=("friedman1")
  METHOD_LIST=("benchmark")
  N_LIST=(500 1000)
  N_FEAT_LIST=(100)
  NOISE_LIST=(1)
  LR_LIST=(0.001)
  WD_LIST=(0)
  SEED_LIST=$(seq 1 1)

elif [[ $CONF == 2 ]]; then
  DATA_LIST=("friedman1") # "custom")
  METHOD_LIST=("naive" "benchmark") # "ces")
  N_LIST=(500 1000)
  N_FEAT_LIST=(100)
  NOISE_LIST=(10)
  LR_LIST=(0.05)
  WD_LIST=(0.1)
  SEED_LIST=$(seq 11 50)

fi


# Slurm parameters
MEMO=5G                             # Memory required (2 GB)
TIME=00-00:20:00                    # Time required (20 m)
CORE=1                              # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs"
mkdir -p $LOGS
mkdir -p $LOGS"/exp1"

OUT_DIR="results"
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR"/exp1"

PLOT_DIR="plots"
mkdir -p $PLOT_DIR
mkdir -p $PLOT_DIR"/exp1"

# Loop over configurations
for SEED in $SEED_LIST; do
  for DATA in "${DATA_LIST[@]}"; do
    for METHOD in "${METHOD_LIST[@]}"; do
      for N in "${N_LIST[@]}"; do
        for N_FEAT in "${N_FEAT_LIST[@]}"; do
          for NOISE in "${NOISE_LIST[@]}"; do
            for LR in "${LR_LIST[@]}"; do
              for WD in "${WD_LIST[@]}"; do


                JOBN="exp1/exp1_"$DATA"_"$METHOD"_n"$N"_p"$N_FEAT"_noise"$NOISE"_lr"$LR"_wd"$WD"_seed"$SEED
                OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                COMPLETE=0
                #ls $OUT_FILE
                if [[ -f $OUT_FILE ]]; then
                  COMPLETE=1
                fi

                if [[ $COMPLETE -eq 0 ]]; then
                  # Script to be run
                  SCRIPT="exp_reg.sh $DATA $METHOD $N $N_FEAT $NOISE $LR $WD $SEED"
                  # Define job name
                  OUTF=$LOGS"/"$JOBN".out"
                  ERRF=$LOGS"/"$JOBN".err"
                  # Assemble slurm order for this job
                  ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
                  # Print order
                  echo $ORD
                  # Submit order
                  $ORD
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
done
