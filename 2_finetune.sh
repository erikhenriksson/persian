#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --account=project_2002026
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err

# Activate virtual environment
source venv/bin/activate

# Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# Run the Python script
python3 2_finetune.py