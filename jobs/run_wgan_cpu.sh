#!/bin/bash
#SBATCH --job-name=wgan_cpu_${EXP}
#SBATCH --partition=cpu-large
#SBATCH --account=gts-jhamilton80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/storage/scratch1/8/sverma87/GGD_CSE-7850/results/%x_%j.log

DATA=/storage/scratch1/8/sverma87/GGD_CSE-7850/data
SRC=/storage/scratch1/8/sverma87/GGD_CSE-7850/src
OUT=/storage/scratch1/8/sverma87/GGD_CSE-7850/results

echo "Starting WGAN (CPU) for ${EXP} on $(hostname) at $(date)"

micromamba run -n GGD_gpu_env python "$SRC/WGAN/core/pca_synthetic_pipeline.py" \
    --input          "$DATA/${EXP}_unphased_tensor.csv" \
    --vcf            "$DATA/${EXP}.vcf" \
    --model          wgan \
    --wgan_epochs    6000 \
    --wgan_latent_dim 64 \
    --n_synthetic    200 \
    ${PCA_ALL_FLAG} \
    --output         "$OUT/${EXP}_wgan_synthetic.csv" \
    --plot           "$OUT/${EXP}_wgan_pca.png"

echo "Done at $(date)"
