#!/bin/bash
#SBATCH --job-name=s11_train
#SBATCH --output=outputs/s11_%j.out
#SBATCH --error=outputs/s11_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================================="

# Setup environment (adjust modules as needed for CEDIA)
# module load python/3.10
# module load cuda/11.8

# Optionally activate virtualenv
# source venv/bin/activate

# Move to the project root
cd "/home/kevin.landazuri__yachaytech.edu.ec/Mateo Gavilanes/Mohamed"

echo "Current directory: $(pwd)"
echo "Running training script..."

# This will run the main training entry point once implemented
# python src/main.py --config configs/experiment.yaml

echo "=========================================================="
echo "End time: $(date)"
echo "=========================================================="
