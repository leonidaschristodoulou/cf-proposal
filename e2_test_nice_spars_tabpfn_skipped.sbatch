#!/bin/bash -l

#SBATCH --job-name=e2_nice_tp
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e2_nice_tp.%j.out
#SBATCH --error=e2_nice_tp.%j.err
#SBATCH --time=03:00:00             # diagnostic only (5 datasets x 5 factuals); bounded in case NICE-sparse hangs rather than errors on one of the skipped combos
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e2_test_nice_spars_tabpfn_skipped.py
