#!/bin/bash
#SBATCH --job-name=s11_train
#SBATCH --output=outputs/s11_%j.out
#SBATCH --error=outputs/s11_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================================="

# Setup environment (adjust modules as needed for CEDIA)
# module load python/3.10
# module load cuda/11.8

# Activate virtualenv if available
source venv/Scripts/activate || source venv/bin/activate || echo "No venv found, using system python"

# Move to the project root
cd "/home/kevin.landazuri__yachaytech.edu.ec/Mateo Gavilanes/Mohamed"

echo "Current directory: $(pwd)"
echo "Running Route A Level 0 and Level 1 baselines..."

python src/route_a_1d_spectral/models/kmeans_probe.py
python src/route_a_1d_spectral/models/linear_baselines.py

echo "=========================================================="
echo "End time: $(date)"
echo "=========================================================="
