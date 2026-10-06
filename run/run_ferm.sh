#!/bin/bash
#SBATCH --job-name=ferm
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
    local log_file="logs/j${SLURM_JOB_ID}_d=${d}_sv=${sv}_C=${C}_c=${count}.out"
    echo "[$(date '+%F %T')] Launching -> $log_file"
    echo "python3 -u ferm_training.py $*"

    srun -N1 -n1 -c4 python3 -u ferm_training.py "$@" > "$log_file" 2>&1 &
    ((count+=1))

    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
        sleep 10
    done
}

trials=5
seed=541944

C_VALUES=(
  0.0001
  0.0001887391822135098
  0.0003562247890262444
  0.0006723357536499335
  0.001268961003167922
  0.002395026619987486
  0.004520353656360241
  0.008531678524172805
  0.01610262027560939
  0.03039195382313198
  0.05736152510448681
  0.1082636733874054
  0.2043359717856942
  0.3856620421163472
  0.7278953843983146
  1.373823795883264
  2.592943797404667
  4.893900918477489
  9.236708571873866
  17.43328822199987
  32.90344562312671
  62.10169418915616
  117.2102297533479
  221.221629107045
  417.53189365604
  788.0462815669904
  1487.352107293512
  2807.216203941176
  5298.316906283702
  10000
)

for d in ACSINC ACSMOB ACSTRAVEL COMPAS GERMAN LAW MEPS
do
  for sv in OTHER GENDER
  do
    for C in "${C_VALUES[@]}"
    do
      run_task \
        num_trials=$trials \
        dataset=$d \
        model.C=$C \
        training.sensible_variable=$sv \
        training.seed=$seed \
        training.normalize_data=true \
        training.validation_fraction=0.25 \
        fairness.optimized_metric=EP \
        fairness.favorable_label=1 \
        offline=true \
        results_root=results
    done
  done
done

wait
echo "All tasks finished."
