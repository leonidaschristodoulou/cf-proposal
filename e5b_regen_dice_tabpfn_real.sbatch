#!/bin/bash -l

#SBATCH --job-name=e5b_regen_real
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e5b_regen_real.%j.out
#SBATCH --error=e5b_regen_real.%j.err
#SBATCH --time=23:59:00             # raised after job 246716: rare outlier calls (~2000s each, seen on diabetes_binarized) blew the original 6h estimate; 24h is the partition max. Resumes from checkpoint (1100/4870 rows already done).
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e5b_regen_dice_tabpfn_real.py
