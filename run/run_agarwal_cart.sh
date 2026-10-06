#!/bin/bash
#SBATCH --job-name=agarwal_cart
#SBATCH --time=4-00:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH -c 4
#SBATCH -p LONG3,LONG2,LONG,GPU-UJM

set -euo pipefail

MAX_JOBS=${SLURM_NTASKS:-4}
count=0
mkdir -p logs

run_task () {
    local log_file="logs/j${SLURM_JOB_ID}_d=${d}_sv=${sv}_o=cart_m=${metric}_depth=None_c=${count}.out"
    echo "[$(date '+%F %T')] Launching -> $log_file"
    echo "python3 -u agarwal_training.py $*"

    srun -N1 -n1 -c4 python3 -u agarwal_training.py "$@" \
        > "$log_file" 2>&1 &
    ((count+=1))

    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
        sleep 10
    done
}

trials=5
seed=541944

for d in ACSINC ACSMOB ACSTRAVEL COMPAS GERMAN LAW MEPS
do
  for sv in OTHER GENDER
  do
    for metric in DP EP EO
    do
      run_task \
        num_trials=$trials \
        dataset=$d \
        model.max_depth=None \
        model.criterion=gini \
        model.max_features=sqrt \
        model.min_samples_split=2 \
        model.min_samples_leaf=1 \
        training.sensible_variable=$sv \
        training.seed=$seed \
        training.normalize_data=true \
        training.validation_fraction=0.25 \
        fairness.optimized_metric=$metric \
        fairness.favorable_label=1 \
        fairness.difference_bound=0.0 \
        offline=true \
        results_root=results
    done
  done
done

wait
echo "All CART tasks finished."
