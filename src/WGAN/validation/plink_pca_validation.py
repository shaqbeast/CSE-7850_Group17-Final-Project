#!/usr/bin/env python3
"""
plink_pca_validation.py  —  Independent PLINK2 PCA on synthetic genotypes
==========================================================================
Implements the professor's recommendation: "do the PCA from the simulated
variant data."

Why this is stricter than the pipeline's built-in PCA overlay
--------------------------------------------------------------
The pipeline projects synthetic genotypes into the SAME reference PCA space
that was used during training — the axes are already defined.  That means
synthetic samples will look structured even if the underlying genotypes are
poor, because the projection scaffold does the heavy lifting.

This script runs `plink2 --pca` fresh on the synthetic genotype matrix with
NO reference projection.  PC axes emerge entirely from variation within the
synthetic data itself.  If the model has learned real ancestry structure, the
PCs should recapitulate that structure.  If it hasn't, the PCA will reveal
artifacts (clustering, wrong axes, excessive heterozygosity effects).

Two outputs per synthetic CSV
------------------------------
1. Independent PCA — plink2 PCA on synthetic samples only.
   Pass/fail: does PC1 separate ancestries?  Does the spread look like
   real population variation?

2. Joint PCA (if --real given) — plink2 PCA on real BAQ + synthetic merged.
   Pass/fail: do synthetic samples fall within the real BAQ cloud, or do they
   form a separate cluster?

Usage
-----
    python plink_pca_validation.py \\
        --synthetic results/cWGAN_vs1p5_gen-data/synthetic_genotypes_seed42.csv \\
        --real      data/BAQ/BAQ_qc2_grch38_final_qc_ldpruned_common_snps_unphased_tensor.csv \\
        --output_dir results/plink_pca_validation

    # Evaluate all CSVs in a directory:
    for f in results/cWGAN_vs1p5_gen-data/*.csv; do
        python plink_pca_validation.py --synthetic "$f" --real data/BAQ/... --output_dir results/plink_pca_validation
    done
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# VCF writer  (adapted from csv_dosage_to_plink.py, accepts a DataFrame)
# ---------------------------------------------------------------------------

_VARIANT_RE = re.compile(
    r'^(?:chr)?([^:]+):(\d+):([ACGTN]+):([ACGTN]+)$', re.IGNORECASE
)
_GT_MAP = {0: '0/0', 1: '0/1', 2: '1/1'}


def _parse_col(col: str) -> tuple[str, str, str, str, str]:
    """Parse 'chr1:12345:A:T' → (chrom, pos, vid, ref, alt)."""
    m = _VARIANT_RE.match(col.strip())
    if not m:
        raise ValueError(f"Column '{col}' not in chr:pos:ref:alt format")
    chrom, pos, ref, alt = m.groups()
    chrom = f'chr{chrom}' if not chrom.lower().startswith('chr') else chrom
    return chrom, pos, f'{chrom}:{pos}:{ref}:{alt}', ref.upper(), alt.upper()


def df_to_vcf(df: pd.DataFrame, vcf_path: Path, sample_ids: list[str]) -> None:
    """Write a diploid GT-only VCF from a dosage DataFrame.

    df:         (n_samples × n_snps) integer DataFrame, values in {0, 1, 2}.
    sample_ids: list of length df.shape[0] — written as VCF sample columns.
    """
    vcf_path.parent.mkdir(parents=True, exist_ok=True)
    # Transpose once so each row access is contiguous in memory (SNP-major)
    geno_T = df.values.T  # (n_snps, n_samples)

    with vcf_path.open('w') as fh:
        fh.write('##fileformat=VCFv4.2\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        header_cols = ('#CHROM', 'POS', 'ID', 'REF', 'ALT',
                       'QUAL', 'FILTER', 'INFO', 'FORMAT', *sample_ids)
        fh.write('\t'.join(header_cols) + '\n')

        for j, col in enumerate(df.columns):
            chrom, pos, vid, ref, alt = _parse_col(col)
            gts = [_GT_MAP.get(int(v), './.') for v in geno_T[j]]
            fh.write('\t'.join(
                [chrom, pos, vid, ref, alt, '.', 'PASS', '.', 'GT', *gts]
            ) + '\n')


def vcf_to_plink(vcf_path: Path, out_prefix: Path, plink2_bin: str) -> None:
    """Convert VCF to binary PLINK with plink2 --make-bed."""
    cmd = [plink2_bin, '--vcf', str(vcf_path), '--allow-extra-chr',
           '--double-id', '--make-bed', '--out', str(out_prefix)]
    print(f"  plink2: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("PLINK2 stderr:\n", r.stderr)
        raise RuntimeError(f"plink2 --make-bed failed (exit {r.returncode})")


# ---------------------------------------------------------------------------
# PLINK2 PCA runner
# ---------------------------------------------------------------------------

def run_plink2_pca(bfile: Path, n_pcs: int, out_prefix: Path,
                   plink2_bin: str, memory_mb: int) -> tuple[Path, Path]:
    """Run plink2 --pca.  Returns (eigenval_path, eigenvec_path)."""
    cmd = [plink2_bin, '--bfile', str(bfile),
           '--pca', str(n_pcs),
           '--out', str(out_prefix),
           '--memory', str(memory_mb),
           '--allow-extra-chr']
    print(f"  plink2: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("PLINK2 stderr:\n", r.stderr)
        raise RuntimeError(f"plink2 --pca failed (exit {r.returncode})")
    return out_prefix.with_suffix('.eigenval'), out_prefix.with_suffix('.eigenvec')


def load_eigenvec(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load .eigenvec → (sample_ids array, PC matrix).

    PLINK2 eigenvec header: '#FID IID PC1 PC2 ...'
    PLINK1 eigenvec header: none (raw space-separated: FID IID PC1 ...)
    """
    with path.open() as fh:
        first = fh.readline().strip()

    if first.startswith('#FID') or first.startswith('FID'):
        df = pd.read_csv(path, sep=r'\s+')
        df.columns = [c.lstrip('#') for c in df.columns]
        ids = df['IID'].values.astype(str)
        pcs = df[[c for c in df.columns if c.startswith('PC')]].values.astype(float)
    else:
        # No header line
        df = pd.read_csv(path, sep=r'\s+', header=None)
        ids = df.iloc[:, 1].values.astype(str)
        pcs = df.iloc[:, 2:].values.astype(float)

    return ids, pcs


def load_eigenval(path: Path) -> np.ndarray:
    """Load .eigenval → fraction of variance explained per PC."""
    vals = np.loadtxt(path)
    total = vals.sum()
    return (vals / total) if total > 0 else vals


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_independent_pca(pcs: np.ndarray, var_exp: np.ndarray,
                          out_path: Path, label: str) -> None:
    """Two-panel PCA plot for synthetic-only run.

    PC1 vs PC2 (colored by PC3 score to reveal any hidden third axis),
    and PC1 vs PC3.  If there are fewer than 3 PCs, fall back gracefully.
    """
    n_pcs = pcs.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ve = [f'{var_exp[i]*100:.1f}%' if i < len(var_exp) else '?' for i in range(3)]

    # Panel 1: PC1 vs PC2, colored by PC3
    ax = axes[0]
    color_vals = pcs[:, 2] if n_pcs > 2 else np.zeros(len(pcs))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=color_vals,
                    cmap='RdYlBu', alpha=0.85, s=45,
                    edgecolors='white', linewidths=0.4)
    ax.set_xlabel(f'PC1 ({ve[0]})', fontsize=11)
    ax.set_ylabel(f'PC2 ({ve[1]})', fontsize=11)
    ax.set_title('PC1 vs PC2  (color = PC3)')
    plt.colorbar(sc, ax=ax, label='PC3' if n_pcs > 2 else 'n/a', shrink=0.8)

    # Panel 2: PC1 vs PC3, colored by PC2
    ax = axes[1]
    pc_y = pcs[:, 2] if n_pcs > 2 else pcs[:, 1]
    color_vals2 = pcs[:, 1]
    sc2 = ax.scatter(pcs[:, 0], pc_y, c=color_vals2,
                     cmap='RdYlBu', alpha=0.85, s=45,
                     edgecolors='white', linewidths=0.4)
    ax.set_xlabel(f'PC1 ({ve[0]})', fontsize=11)
    ax.set_ylabel(f'PC3 ({ve[2]})' if n_pcs > 2 else f'PC2 ({ve[1]})', fontsize=11)
    ax.set_title('PC1 vs PC3  (color = PC2)' if n_pcs > 2 else 'PC1 vs PC2')
    plt.colorbar(sc2, ax=ax, label='PC2', shrink=0.8)

    plt.suptitle(
        f'Independent PLINK2 PCA — {label}  (synthetic genotypes only)\n'
        f'PC axes derived from synthetic variation — no reference projection',
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}', flush=True)


def plot_joint_pca(real_pcs: np.ndarray, synth_pcs: np.ndarray,
                   var_exp: np.ndarray, out_path: Path, label: str,
                   n_real: int, n_synth: int) -> None:
    """Two-panel joint PCA of real BAQ + synthetic samples."""
    n_pcs = min(real_pcs.shape[1], synth_pcs.shape[1])
    ve    = [f'{var_exp[i]*100:.1f}%' if i < len(var_exp) else '?' for i in range(3)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [(0, 1), (0, 2)] if n_pcs > 2 else [(0, 1), (0, 1)]

    for ax, (xi, yi) in zip(axes, pairs):
        ax.scatter(real_pcs[:, xi], real_pcs[:, yi],
                   c='#E63946', alpha=0.85, s=50, label=f'Real BAQ (n={n_real})',
                   edgecolors='white', linewidths=0.4, zorder=2)
        ax.scatter(synth_pcs[:, xi], synth_pcs[:, yi],
                   c='#7B2FBE', alpha=0.75, s=40, label=f'Synthetic (n={n_synth})',
                   edgecolors='white', linewidths=0.4, zorder=3)
        pc_labels = [f'PC1 ({ve[0]})', f'PC2 ({ve[1]})', f'PC3 ({ve[2]})']
        ax.set_xlabel(pc_labels[xi], fontsize=11)
        ax.set_ylabel(pc_labels[yi], fontsize=11)
        ax.set_title(f'PC{xi+1} vs PC{yi+1}')

    axes[0].legend(fontsize=9, markerscale=0.9)

    plt.suptitle(
        f'Joint PLINK2 PCA — Real BAQ + {label}\n'
        f'PC axes from merged real+synthetic data — no reference projection',
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}', flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--synthetic', required=True,
                        help='Synthetic genotype CSV (samples × SNPs, dosage 0/1/2)')
    parser.add_argument('--real', default=None,
                        help='Real BAQ genotype CSV for joint PCA (optional but recommended)')
    parser.add_argument('--output_dir', default='results/plink_pca_validation')
    parser.add_argument('--label', default=None,
                        help='Label for plots (defaults to stem of --synthetic)')
    parser.add_argument('--n_components', type=int, default=20,
                        help='Number of PCs for PLINK2 PCA (default 20)')
    parser.add_argument('--plink2', default='plink2',
                        help='Path to plink2 binary (default: plink2 on PATH)')
    parser.add_argument('--memory', type=int, default=28000,
                        help='Memory for PLINK2 in MB (default 28000)')
    parser.add_argument('--keep_vcf', action='store_true',
                        help='Keep intermediate VCF files (default: delete after conversion)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.synthetic).stem

    t_start = time.time()
    print(f'\n{"="*60}', flush=True)
    print(f'PLINK2 PCA Validation: {label}', flush=True)
    print(f'{"="*60}', flush=True)

    # ------------------------------------------------------------------
    # Load synthetic CSV
    # ------------------------------------------------------------------
    print(f'\n[1] Loading synthetic CSV...', flush=True)
    synth_df  = pd.read_csv(args.synthetic, header=0)
    n_synth   = synth_df.shape[0]
    n_snps    = synth_df.shape[1]
    print(f'  {n_synth} samples × {n_snps} SNPs', flush=True)

    # ------------------------------------------------------------------
    # Independent PCA — synthetic only
    # ------------------------------------------------------------------
    print(f'\n[2] Writing synthetic-only VCF...', flush=True)
    synth_ids    = [f'SYN{i+1:04d}' for i in range(n_synth)]
    synth_vcf    = out_dir / f'{label}_synth.vcf'
    synth_prefix = out_dir / f'{label}_synth'
    t0 = time.time()
    df_to_vcf(synth_df, synth_vcf, synth_ids)
    print(f'  VCF written in {time.time()-t0:.1f}s', flush=True)

    print(f'\n[3] Converting VCF → PLINK binary...', flush=True)
    vcf_to_plink(synth_vcf, synth_prefix, args.plink2)
    if not args.keep_vcf:
        synth_vcf.unlink(missing_ok=True)

    print(f'\n[4] Running PLINK2 PCA (synthetic only, {args.n_components} PCs)...', flush=True)
    pca_prefix     = out_dir / f'{label}_synth_pca'
    eval_p, evec_p = run_plink2_pca(synth_prefix, args.n_components,
                                     pca_prefix, args.plink2, args.memory)
    _, synth_pcs   = load_eigenvec(evec_p)
    var_exp_s      = load_eigenval(eval_p)
    print(f'  Variance explained — PC1:{var_exp_s[0]*100:.1f}%  '
          f'PC2:{var_exp_s[1]*100:.1f}%  PC3:{var_exp_s[2]*100:.1f}%', flush=True)

    indep_plot = out_dir / f'{label}_independent_pca.png'
    plot_independent_pca(synth_pcs, var_exp_s, indep_plot, label)

    # ------------------------------------------------------------------
    # Joint PCA — real BAQ + synthetic
    # ------------------------------------------------------------------
    if args.real:
        print(f'\n[5] Loading real BAQ CSV...', flush=True)
        real_df = pd.read_csv(args.real, header=0)
        n_real  = real_df.shape[0]
        print(f'  {n_real} samples × {real_df.shape[1]} SNPs', flush=True)

        # Align to shared SNPs — QC may have dropped a few between datasets
        shared = list(synth_df.columns.intersection(real_df.columns))
        if len(shared) < n_snps:
            print(f'  Aligned to {len(shared)} shared SNPs '
                  f'(dropped {n_snps - len(shared)} from synthetic, '
                  f'{real_df.shape[1] - len(shared)} from real)', flush=True)
        synth_sub = synth_df[shared]
        real_sub  = real_df[shared]
        del real_df

        # Stack: real samples first, then synthetic
        # Keep track of boundary index for splitting eigenvec later
        real_ids   = [f'BAQ{i+1:04d}' for i in range(n_real)]
        synth_ids2 = [f'SYN{i+1:04d}' for i in range(n_synth)]
        merged_df  = pd.concat([real_sub, synth_sub], ignore_index=True)

        print(f'\n[6] Writing merged VCF ({n_real} real + {n_synth} synthetic)...', flush=True)
        merged_vcf    = out_dir / f'{label}_merged.vcf'
        merged_prefix = out_dir / f'{label}_merged'
        t0 = time.time()
        df_to_vcf(merged_df, merged_vcf, real_ids + synth_ids2)
        print(f'  VCF written in {time.time()-t0:.1f}s', flush=True)
        del merged_df

        print(f'\n[7] Converting merged VCF → PLINK binary...', flush=True)
        vcf_to_plink(merged_vcf, merged_prefix, args.plink2)
        if not args.keep_vcf:
            merged_vcf.unlink(missing_ok=True)

        print(f'\n[8] Running PLINK2 PCA (merged, {args.n_components} PCs)...', flush=True)
        mpca_prefix     = out_dir / f'{label}_merged_pca'
        meval_p, mevec_p = run_plink2_pca(merged_prefix, args.n_components,
                                           mpca_prefix, args.plink2, args.memory)
        ids_m, pcs_m  = load_eigenvec(mevec_p)
        var_exp_m     = load_eigenval(meval_p)
        print(f'  Variance explained — PC1:{var_exp_m[0]*100:.1f}%  '
              f'PC2:{var_exp_m[1]*100:.1f}%  PC3:{var_exp_m[2]*100:.1f}%', flush=True)

        # Split eigenvec back into real vs synthetic by position
        real_pcs_m  = pcs_m[:n_real]
        synth_pcs_m = pcs_m[n_real:]

        joint_plot = out_dir / f'{label}_joint_pca.png'
        plot_joint_pca(real_pcs_m, synth_pcs_m, var_exp_m,
                       joint_plot, label, n_real, n_synth)

    print(f'\n{"="*60}', flush=True)
    print(f'Done.  Total time: {time.time()-t_start:.1f}s', flush=True)
    print(f'Output directory: {out_dir}', flush=True)
    print(f'{"="*60}\n', flush=True)


if __name__ == '__main__':
    main()
