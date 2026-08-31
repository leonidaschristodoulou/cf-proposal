#!/bin/bash -l

#SBATCH --job-name=e5_test3
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e5_test3.%j.out
#SBATCH --error=e5_test3.%j.err
#SBATCH --time=00:45:00             # 20 factuals x genetic search against LR, unknown per-instance cost
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e5_test3_dice_genetic_realworld_failures.py
