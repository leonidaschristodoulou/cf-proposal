#!/bin/bash -l

#SBATCH --job-name=e5_lr_gen
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e5_lr_gen.%j.out
#SBATCH --error=e5_lr_gen.%j.err
#SBATCH --time=06:00:00             # est. ~2.8h (5000 x ~2s blended success/failure), generous margin; checkpointed
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e5_regen_dice_genetic_lr_realworld.py
