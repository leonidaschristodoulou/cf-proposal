#!/bin/bash -l

#SBATCH --job-name=cfprop
#SBATCH --partition=a100            # Partition
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --ntasks-per-node=1         # Number of tasks
#SBATCH --cpus-per-task=8           # Number of cpu cores
#SBATCH --mem=20G                   # Total memory per node
#SBATCH --output=job.%j.out         # Stdout (%j=jobId)
#SBATCH --error=job.%j.err          # Stderr (%j=jobId)
#SBATCH --time=23:59:00             # Walltime
#SBATCH -A p241                     # Accounting project


# Load any necessary modules and activate environment

module load Anaconda3

conda activate tabpfn

# Add our environment as a notebook kernel
python -m ipykernel install --user --name=tabpfn

# Figure out the directory containing this script so outputs land there
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    SCRIPT_PATH=$(realpath "$0")
fi
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# Compute node hostname
HOSTNAME=$(hostname)

# Generate random ports for Jupyter
JUPYTER_PORT=$(shuf -i 10000-60000 -n 1)

# Generate a random password for Jupyter Notebook
PASSWORD=$(openssl rand -base64 12)

# Hash the password using Jupyter's built-in function
HASHED_PASSWORD=$(python -c "from jupyter_server.auth import passwd; print(passwd('$PASSWORD'))")


# Run Jupyter notebook
jupyter notebook --port=$JUPYTER_PORT --NotebookApp.password="$HASHED_PASSWORD" --notebook-dir="$SCRIPT_DIR" --no-browser --ip 0.0.0.0 > "$SCRIPT_DIR/jupyter.log" 2>&1 &

sleep 5


LOGIN_HOST="front01.hpcf.cyi.ac.cy"


# Prepare the message to be displayed and saved to a file
CONNECTION_MESSAGE=$(cat <<EOF
==================================================================
Run this command to connect on your jupyter notebooks remotely
ssh -N -J ${USER}@${LOGIN_HOST} ${USER}@${HOSTNAME} -L ${JUPYTER_PORT}:localhost:${JUPYTER_PORT}


Jupyter Notebook is running at: http://localhost:$JUPYTER_PORT
Password to access the notebook: $PASSWORD
==================================================================
EOF
)

# Print the connection details to both the terminal and a txt file
echo "$CONNECTION_MESSAGE" | tee "$SCRIPT_DIR/connection_info.txt"

wait