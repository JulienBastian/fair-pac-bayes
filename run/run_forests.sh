#!/bin/bash
#SBATCH --job-name=pbf_forests
#SBATCH --time=7-00:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH -c 4
#SBATCH -p LONG3,LONG2,LONG,GPU-UJM


set -euo pipefail

MAX_JOBS=${SLURM_NTASKS:-4}
count=0
mkdir -p logs

run_task () {
    local log_file="logs/j${SLURM_JOB_ID}_d=${d}_sv=${sv}_r=${r}_fb=${fb}_f=${f}_a=${alpha}_c=${count}.out"
    echo "[$(date '+%F %T')] Launching -> $log_file"
    echo "python3 -u training.py $*"

    srun -N1 -n1 -c4 python3 -u training.py "$@" > "$log_file" 2>&1 &
    ((count+=1))

    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
        sleep 10
    done
}

prd=RandomForests
n=100
trials=5
seed=541944
device=cpu
samples_prop=0.5
max_tree_depth=None
tree_criterion=gini
class_weight=auto
max_features=sqrt
min_samples_split=2
min_samples_leaf=1

for d in ACSINC ACSMOB ACSTRAVEL COMPAS GERMAN LAW MEPS
do
  for sv in OTHER GENDER
  do
    for alpha in 0.1 0.25 0.5 0.75 0.9
    do
      for f in DP EO EP
      do
        r=Triple
        for fb in ours_stochastic_fairness oneto_fairness
        do
          run_task num_trials=$trials dataset=$d model.n=$n training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r fairness_bound.type=$fb training.fairness_metric=$f training.sensible_variable=$sv model.pred=$prd model.samples_prop=$samples_prop model.max_tree_depth=$max_tree_depth model.tree_criterion=$tree_criterion model.tree_class_weight=$class_weight model.tree_max_features=$max_features model.tree_min_samples_split=$min_samples_split model.tree_min_samples_leaf=$min_samples_leaf training.seed=$seed training.alpha=$alpha device=$device
          run_task num_trials=$trials dataset=$d model.n=$n training.lr=0.1 training.distribution=dirichlet  model.prior=1        training.risk=$r fairness_bound.type=$fb training.fairness_metric=$f training.sensible_variable=$sv model.pred=$prd model.samples_prop=$samples_prop model.max_tree_depth=$max_tree_depth model.tree_criterion=$tree_criterion model.tree_class_weight=$class_weight model.tree_max_features=$max_features model.tree_min_samples_split=$min_samples_split model.tree_min_samples_leaf=$min_samples_leaf training.seed=$seed training.alpha=$alpha device=$device
          run_task num_trials=$trials dataset=$d model.n=$n training.lr=0.1 training.distribution=gaussian   model.prior=0        training.risk=$r fairness_bound.type=$fb training.fairness_metric=$f training.sensible_variable=$sv model.pred=$prd model.samples_prop=$samples_prop model.max_tree_depth=$max_tree_depth model.tree_criterion=$tree_criterion model.tree_class_weight=$class_weight model.tree_max_features=$max_features model.tree_min_samples_split=$min_samples_split model.tree_min_samples_leaf=$min_samples_leaf training.seed=$seed training.alpha=$alpha device=$device
        done
      done
    done

    for f in DP EP
    do
      r=Oneto
      fb=oneto_fairness
      for gamma in 0.1 1 10 100
      do
        for alpha in 0.1 1 10 100
        do
          run_task num_trials=$trials dataset=$d model.n=$n training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r fairness_bound.type=$fb training.fairness_metric=$f training.sensible_variable=$sv model.pred=$prd model.samples_prop=$samples_prop model.max_tree_depth=$max_tree_depth model.tree_criterion=$tree_criterion model.tree_class_weight=$class_weight model.tree_max_features=$max_features model.tree_min_samples_split=$min_samples_split model.tree_min_samples_leaf=$min_samples_leaf training.seed=$seed training.alpha=$alpha training.gamma=$gamma device=$device
        done
      done
    done
  done
done

wait
echo "All tasks finished."
