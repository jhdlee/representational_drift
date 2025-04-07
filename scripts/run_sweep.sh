#!/bin/bash
#SBATCH --job-name=rt
#SBATCH --error=.logs/rt_%j_%a.err         
#SBATCH --output=.logs/rt_%j_%a.out     
#SBATCH --time=5:59:59                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint='(GPU_SKU:H100_SXM5&GPU_MEM:80GB)|(GPU_SKU:A100_SXM4&GPU_MEM:80GB)'
#SBATCH --mail-type=ALL
#SBATCH --requeue

# Run a wandb sweep for SMDS model

# Activate conda environment
source /home/groups/swl1/hdlee/miniconda3/bin/activate smds

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
