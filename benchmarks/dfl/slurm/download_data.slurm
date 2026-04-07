#!/bin/bash
#SBATCH --job-name=cifar10-dl
#SBATCH --partition=b40x4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/cifar10_download.log

module load slurm
cd /lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning

export SINGULARITYENV_PYTHONNOUSERSITE=1

mkdir -p /lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning/data /lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning/logs

singularity exec dfl.sif python -c "
from torchvision import datasets
datasets.CIFAR10('/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning/data', train=True, download=True)
datasets.CIFAR10('/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning/data', train=False, download=True)
print('CIFAR-10 downloaded successfully')
"
