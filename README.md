# Generative Genomic Data — CSE 7850 Group 17 Final Project

Synthetic genotype generation using VAE and Wasserstein GAN (WGAN-GP) on real population genomic data from the 1000 Genomes Project. 

## WGAN
The pipeline compresses genotype matrices into PCA space, trains a generative model, and produces synthetic samples that preserve population-level allele frequency distributions.

## VAE

---

## Project Structure

```
GGD_CSE-7850/
├── data/                          # Input VCF files and processed tensors (gitignored)
├── results/                       # Model outputs, PCA plots, validation results (gitignored)
├── jobs/                          # SLURM batch submission scripts
│   ├── run_wgan.sh                # GPU WGAN training job
│   ├── run_wgan_cpu.sh            # CPU WGAN training job
│   ├── run_plink_pca_validation.sh# PLINK2 PCA validation job
│   └── csv_to_parquet.sh          # CSV → Parquet conversion job
└── src/
    └── WGAN/
        ├── core/
        │   └── pca_synthetic_pipeline.py     # Main generation pipeline
        ├── preprocessing/
        │   └── vcf_to_unphased_tensor.py     # VCF → unphased dosage CSV
        └── validation/
            ├── plink_pca_validation.py       # PLINK2-based PCA validation
            └── csv_to_parquet.py             # CSV → Parquet conversion
```

---

## Pipeline Overview

### Step 1 — Preprocess VCF to Unphased Tensor

Convert a phased VCF file to an unphased dosage matrix (samples × SNPs), encoded as 0/1/2 with -1 for missing.

```bash
python src/WGAN/preprocessing/vcf_to_unphased_tensor.py \
    --input  data/exp1_CLM.vcf \
    --output data/exp1_CLM_unphased_tensor.csv
```

### Step 2 — Train WGAN and Generate Synthetic Samples

Runs PCA compression, trains a WGAN-GP in PC space, generates synthetic genotypes, and outputs a PCA overlay plot.

```bash
# Submit GPU job
EXP=exp1_CLM sbatch jobs/run_wgan.sh

# Submit CPU job
EXP=exp1_CLM sbatch jobs/run_wgan_cpu.sh
```

**Key arguments for `pca_synthetic_pipeline.py`:**

| Argument | Description |
|---|---|
| `--input` | Unphased tensor CSV (samples × SNPs) |
| `--vcf` | Original VCF (used to split target vs. reference samples) |
| `--model` | `wgan`, `gaussian_copula`, or `admixture_interpolation` |
| `--wgan_epochs` | Number of training epochs (default: 6000) |
| `--wgan_latent_dim` | Latent dimension size (default: 64) |
| `--n_synthetic` | Number of synthetic samples to generate |
| `--output` | Output synthetic genotype CSV |
| `--plot` | Output PCA overlay PNG |

### Step 3 — Validate with PLINK2 PCA

Runs independent and joint (real + synthetic) PCA using PLINK2 to assess how well synthetic samples mirror real population structure.

```bash
POP=CLM EXP=exp1_CLM SYNTH_CSV=exp1_CLM_wgan_synthetic.csv \
    sbatch jobs/run_plink_pca_validation.sh
```

---

## Populations

| Experiment | Population | Description |
|---|---|---|
| exp1 | CLM | Colombians from Medellín, Colombia |
| exp2 | PUR | Puerto Ricans from Puerto Rico |
| exp3 | ACB | African Caribbeans in Barbados |
| exp4 | JPT | Japanese in Tokyo, Japan |
| exp5 | GIH | Gujarati Indians in Houston, Texas |

---

## Environment

Jobs use a micromamba/conda environment located at:
```
/storage/scratch1/8/sverma87/conda-envs/GGD_gpu_env
```

**Key dependencies:**
- PyTorch (GPU support via CUDA)
- scikit-allel
- scikit-learn
- pandas, numpy
- matplotlib
- pyarrow
- PLINK2

---
