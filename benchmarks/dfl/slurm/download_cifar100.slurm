#!/bin/bash
#SBATCH --job-name=cifar100-dl
#SBATCH --partition=b40x4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/cifar100_download.log

module load slurm
cd /lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning

export SINGULARITYENV_PYTHONNOUSERSITE=1

mkdir -p data logs

singularity exec dfl.sif python -c "
from torchvision import datasets
datasets.CIFAR100('data', train=True, download=True)
datasets.CIFAR100('data', train=False, download=True)
print('CIFAR-100 downloaded successfully')
"
