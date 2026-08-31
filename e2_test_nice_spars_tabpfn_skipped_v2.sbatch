#!/bin/bash -l

#SBATCH --job-name=e2_nice_v2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e2_nice_v2.%j.out
#SBATCH --error=e2_nice_v2.%j.err
#SBATCH --time=00:30:00             # 3 datasets x 300s hard timeout each, max 15 min possible + normal-case slack
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e2_test_nice_spars_tabpfn_skipped_v2.py
