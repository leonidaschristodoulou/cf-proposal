#!/bin/bash -l

#SBATCH --job-name=e2_nice_v2r
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e2_nice_v2r.%j.out
#SBATCH --error=e2_nice_v2r.%j.err
#SBATCH --time=23:59:00             # true per-instance 120s cap now enforced; worst-case theoretical max is high but realistic cost much lower. Checkpointed per pair, resumable if needed.
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e2_regen_nice_spars_tabpfn_unskipped_v2.py
