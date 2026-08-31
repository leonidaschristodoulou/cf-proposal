#!/bin/bash -l

#SBATCH --job-name=e5b_regen_mock
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=e5b_regen_mock.%j.out
#SBATCH --error=e5b_regen_mock.%j.err
#SBATCH --time=08:00:00             # est. ~3.75h (3000 x ~4.5s blended), generous margin; checkpointed so a timeout just needs resubmission
#SBATCH --account=p318

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e5b_regen_dice_tabpfn_mock.py
