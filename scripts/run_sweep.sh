#!/bin/bash
#SBATCH --job-name=rt
#SBATCH --error=.logs/rt_%j_%a.err         
#SBATCH --output=.logs/rt_%j_%a.out     
#SBATCH --time=47:59:59                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint='(GPU_SKU:H100_SXM5&GPU_MEM:80GB)|(GPU_SKU:A100_SXM4&GPU_MEM:80GB)'
#SBATCH --mail-type=ALL
#SBATCH --requeue

# Run a wandb sweep for SMDS model

# Detect cluster and activate conda environment
if [ -d "/projects/m000215/hdlee" ]; then
    # Marlowe cluster
    export CLUSTER_NAME=marlowe
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
