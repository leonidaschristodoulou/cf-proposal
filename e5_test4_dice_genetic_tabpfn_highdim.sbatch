#!/bin/bash -l

#SBATCH --job-name=e5_test4
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e5_test4.%j.out
#SBATCH --error=e5_test4.%j.err
#SBATCH --time=00:30:00
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e5_test4_dice_genetic_tabpfn_highdim.py
