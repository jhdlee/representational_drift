#!/bin/bash
#SBATCH --job-name=rt_sim
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=ALL
#SBATCH --requeue

# Sherlock-specific options (comment out for Marlowe)
# #SBATCH --error=/scratch/users/hdlee/representational_drift/logs/rt_sim_%j_%a.err
# #SBATCH --output=/scratch/users/hdlee/representational_drift/logs/rt_sim_%j_%a.out
# #SBATCH --constraint='(GPU_SKU:H100_SXM5&GPU_MEM:80GB)|(GPU_SKU:A100_SXM4&GPU_MEM:80GB)'

# Marlowe-specific options (uncomment for Marlowe, comment out Sherlock options above)
#SBATCH --error=/scratch/m000215-pm05/hdlee/representational_drift/logs/rt_sim_%j_%a.err
#SBATCH --output=/scratch/m000215-pm05/hdlee/representational_drift/logs/rt_sim_%j_%a.out
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -A marlowe-m000215-pm05
#SBATCH --qos=medium

# Run a wandb sweep for simulated data from LDS and SMDS models

# Detect cluster and activate conda environment
if [ -d "/projects/m000215/hdlee" ]; then
    # Marlowe cluster
    export CLUSTER_NAME=marlowe
    module load cudatoolkit/12.5
    export WANDB_DIR=/scratch/m000215-pm05/hdlee/representational_drift
    mkdir -p $WANDB_DIR
    source /projects/m000215/hdlee/miniconda3/bin/activate smds
elif [ -d "/oak/stanford/groups/swl1/hdlee" ]; then
    # Sherlock cluster
    export CLUSTER_NAME=sherlock
    export WANDB_DIR=/scratch/users/hdlee/representational_drift
    mkdir -p $WANDB_DIR
    source /oak/stanford/groups/swl1/hdlee/miniconda3/bin/activate /scratch/users/hdlee/miniconda3/envs/smds
else
    echo "Warning: Unknown cluster, attempting default conda activation"
    conda activate smds
fi

# Get Sweep ID from the command line argument
SWEEP_ID=$1

# Check if a Sweep ID was provided
if [ -z "$SWEEP_ID" ]; then
  echo "Error: No Sweep ID provided. Usage: sbatch wandb_agent.sbatch <SWEEP_ID>"
  exit 1
fi

# Construct the full Sweep Identifier
FULL_SWEEP_ID="jhdlee/smds/$SWEEP_ID"

# Run the W&B agent
wandb agent $FULL_SWEEP_ID 