#!/bin/bash
#SBATCH --job-name=pca_val_${POP}
#SBATCH --partition=cpu-large
#SBATCH --account=gts-jhamilton80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/scratch1/8/sverma87/GGD_CSE-7850/results/%x_%j.log

DATA=/storage/scratch1/8/sverma87/GGD_CSE-7850/data
SRC=/storage/scratch1/8/sverma87/GGD_CSE-7850/src
OUT=/storage/scratch1/8/sverma87/GGD_CSE-7850/results
PLINK2=/storage/scratch1/8/sverma87/conda-envs/GGD_gpu_env/bin/plink2
REAL_ONLY=/tmp/${POP}_real_only_$$.csv

echo "Starting PLINK PCA validation for ${POP} on $(hostname) at $(date)"

echo "  Extracting real ${POP} samples from tensor CSV..."
micromamba run -n GGD_gpu_env python - <<EOF
import pandas as pd

vcf_path  = "$DATA/${EXP}.vcf"
csv_path  = "$DATA/${EXP}_unphased_tensor.csv"
pop       = "$POP"
out_path  = "$REAL_ONLY"

sample_names = []
with open(vcf_path) as f:
    for line in f:
        if line.startswith('#CHROM'):
            sample_names = line.strip().split('\t')[9:]
            break

target_mask = [s.split('_')[0] == pop for s in sample_names]
print(f"  VCF samples: {len(sample_names)}, target ({pop}): {sum(target_mask)}")

df = pd.read_csv(csv_path, header=0)
df[target_mask].reset_index(drop=True).to_csv(out_path, index=False)
print(f"  Saved {sum(target_mask)} real {pop} rows to {out_path}")
EOF

echo "  Running PLINK2 PCA validation..."
micromamba run -n GGD_gpu_env python "$SRC/WGAN/validation/plink_pca_validation.py" \
    --synthetic  "$OUT/${SYNTH_CSV}" \
    --real       "$REAL_ONLY" \
    --output_dir "$OUT/plink_pca_validation" \
    --label      "${POP}" \
    --n_components 20 \
    --plink2     "$PLINK2" \
    --memory     100000

rm -f "$REAL_ONLY"
echo "Done at $(date)"
