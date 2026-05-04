#!/bin/bash
#SBATCH -J csv_to_parquet
#SBATCH -p interactive-cpu
#SBATCH -A gts-jhamilton80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /storage/scratch1/8/sverma87/csv_to_parquet_%j.out
#SBATCH -e /storage/scratch1/8/sverma87/csv_to_parquet_%j.err

source /storage/home/hcoda1/8/sverma87/miniconda3/etc/profile.d/conda.sh
conda activate /storage/scratch1/8/sverma87/conda-envs/GGD_gpu_env

python /storage/home/hcoda1/8/sverma87/scratch/GGD_CSE-7850/src/WGAN/validation/csv_to_parquet.py
