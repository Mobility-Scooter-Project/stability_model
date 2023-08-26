#!/bin/bash
#SBATCH --job-name=Job3-1
#SBATCH --output=Job3-1.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=compute
source /data03/home/ruoqihuang/anaconda3/etc/profile.d/conda.sh
conda activate tf
chmod +x src/test.py
srun src/test.py