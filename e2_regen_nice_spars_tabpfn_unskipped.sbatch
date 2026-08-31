#!/bin/bash -l

#SBATCH --job-name=e2_nice_regen
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e2_nice_regen.%j.out
#SBATCH --error=e2_nice_regen.%j.err
#SBATCH --time=08:00:00             # 10 pairs x 2400s hard cap each = 6.67h theoretical max; realistically much faster given observed timings. Checkpointed per pair.
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e2_regen_nice_spars_tabpfn_unskipped.py
