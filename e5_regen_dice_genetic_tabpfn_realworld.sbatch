#!/bin/bash -l

#SBATCH --job-name=e5_tp_gen
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e5_tp_gen.%j.out
#SBATCH --error=e5_tp_gen.%j.err
#SBATCH --time=23:59:00             # genetic+TabPFN cost is uncertain (random already showed ~1000x outliers); 24h partition max, checkpointed per pair
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e5_regen_dice_genetic_tabpfn_realworld.py
