#!/bin/bash
#SBATCH --job-name=DeepGenomicsGRU
#SBATCH --gpus=1
#SBATCH -с 4
python main.py

echo "Working on node `hostname`"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
