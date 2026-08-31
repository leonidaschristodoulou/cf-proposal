#!/bin/bash -l

#SBATCH --job-name=e5b_test
#SBATCH --partition=a100            # Partition
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --ntasks-per-node=1         # Number of tasks
#SBATCH --cpus-per-task=8           # Number of cpu cores
#SBATCH --mem=20G                   # Total memory per node
#SBATCH --output=e5b_test.%j.out    # Stdout (%j=jobId)
#SBATCH --error=e5b_test.%j.err     # Stderr (%j=jobId)
#SBATCH --time=00:30:00             # Walltime -- bounded, this is a feasibility test only
#SBATCH --account=p318                      # Accounting project

module load Anaconda3
conda activate tabpfn

cd /nvme/h/lchristodoulou/pace/cf-proposal
python e5b_test_dice_tabpfn.py
