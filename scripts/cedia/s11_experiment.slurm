#!/bin/bash
#SBATCH --job-name=s11_master_experiment
#SBATCH --output=s11_master_%j.out
#SBATCH --error=s11_master_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate environment (adjust module/path as needed for your CEDIA env)
# module load miniconda
# conda activate s11_env

echo "Starting S11 Master Experiment on CEDIA..."
# Volver a la raíz del proyecto para ejecutar
cd ../..
# Asegurar dependencias (xgboost no está en el nodo de CEDIA por defecto)
python -m pip install --user xgboost
python scripts/run_full_experiment.py
echo "Experiment Completed."
