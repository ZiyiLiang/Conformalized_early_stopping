DATA=$1

mkdir -p results_hpc
mkdir -p plots_hpc

rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/Conformalized_early_stopping/experiments/results/exp1/* results_hpc/exp1/
#rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/Conformalized_early_stopping/experiments/plots/exp1/* plots_hpc/exp1/
