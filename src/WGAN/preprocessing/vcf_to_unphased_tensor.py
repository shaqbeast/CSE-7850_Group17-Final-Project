import allel
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="Convert genomic data (VCF or CSV) to an unphased genotype tensor (CSV)"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input file (VCF or CSV)"
    )
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["vcf", "csv"],
        default="vcf",
        help="Type of input file: 'vcf' or 'csv' (default: vcf)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output CSV file. If not specified, auto-generated from input name."
    )
    parser.add_argument(
        "--samples-as-rows",
        action="store_true",
        help="For CSV input: if set, samples are rows and SNPs are columns. Otherwise, SNPs are rows and samples are columns."
    )
    args = parser.parse_args()
    
    input_path = args.input
    
    # Auto-generate output path if not provided
    if args.output is None:
        output_path = Path(input_path).stem + "_unphased_tensor.csv"
    else:
        output_path = args.output
    
    # Process based on input type
    if args.input_type == "vcf":
        print(f"Loading VCF: {input_path}", flush=True)

        # Parse sample names from VCF header without loading any genotype data
        samples = []
        with open(input_path) as fh:
            for line in fh:
                if line.startswith('#CHROM'):
                    samples = line.strip().split('\t')[9:]
                    break
        n_samples = len(samples)
        print(f"[INFO] Found {n_samples} samples", flush=True)

        # Stream VCF in variant chunks to keep peak RAM ~700 MB regardless of file size.
        # Each sample's genotype values are appended to a small binary temp file
        # (n_variants × 1 byte), then assembled into the final CSV at the end.
        tmp_dir = Path(output_path + '.tmpdir')
        tmp_dir.mkdir(exist_ok=True)

        snp_ids = []
        n_variants = 0
        chunk_length = 100_000
        fields = ['calldata/GT', 'variants/CHROM', 'variants/POS', 'variants/REF', 'variants/ALT']

        print(f"[INFO] Streaming in chunks of {chunk_length} variants...", flush=True)
        # iter_vcf_chunks returns (field_names, samples, headers, VCFChunkIterator)
        # Each iteration of VCFChunkIterator yields (chunk_dict, n_variants_in_chunk)
        _, _, _, vcf_iter = allel.iter_vcf_chunks(input_path, fields=fields, chunk_length=chunk_length)

        sample_fds = [open(tmp_dir / f's{i}.bin', 'wb') for i in range(n_samples)]
        try:
            for chunk, chunk_n, *_ in vcf_iter:
                gt = chunk['calldata/GT']          # (chunk_size, n_samples, 2) int8
                chunk_size = gt.shape[0]

                a0 = gt[:, :, 0]
                a1 = gt[:, :, 1]
                missing = (a0 < 0) | (a1 < 0)
                ug = (a0.astype(np.int16) + a1.astype(np.int16)).astype(np.int8)
                ug[missing] = np.int8(-1)
                del a0, a1, missing, gt

                # Transpose to (n_samples, chunk_size) and flush each sample's slice
                ug_T = np.ascontiguousarray(ug.T)
                del ug
                for i in range(n_samples):
                    sample_fds[i].write(ug_T[i].tobytes())
                del ug_T
                gc.collect()

                chroms    = chunk['variants/CHROM']
                positions = chunk['variants/POS'].astype(str)
                refs      = chunk['variants/REF']
                alts      = chunk['variants/ALT'][:, 0]
                snp_ids.extend(f"chr{c}:{p}:{r}:{a}" for c, p, r, a in zip(chroms, positions, refs, alts))
                n_variants += chunk_size
                print(f"[INFO] Processed {n_variants} variants...", flush=True)
        finally:
            for fd in sample_fds:
                fd.close()

        print(f"\nUnphased tensor shape: ({n_samples}, {n_variants}) (samples × variants)", flush=True)

        # Assemble final CSV: one row per sample, read from its tiny binary temp file
        print(f"[INFO] Writing CSV to {output_path}...", flush=True)
        with open(output_path, 'w') as f:
            f.write(','.join(snp_ids) + '\n')
            for i in range(n_samples):
                data = np.fromfile(tmp_dir / f's{i}.bin', dtype=np.int8)
                f.write(','.join(map(str, data)) + '\n')
                if (i + 1) % 100 == 0:
                    print(f"[INFO] Written {i+1}/{n_samples} samples", flush=True)
        print(f"[INFO] Written {n_samples}/{n_samples} samples", flush=True)

        shutil.rmtree(tmp_dir)
        gc.collect()

    elif args.input_type == "csv":
        # Load CSV
        print(f"Loading CSV: {input_path}")
        df = pd.read_csv(input_path)
        
        # If samples are rows and SNPs are columns, transpose so SNPs are rows and samples are columns
        if args.samples_as_rows:
            print("Transposing data: samples are rows, SNPs are columns")
            df = df.T
            df.reset_index(drop=True, inplace=True)
        
        print(f"Tensor shape: {df.shape}")
        print(f"First 5 rows (all columns):")
        print(df.iloc[:5, :10])
        
        # Save DataFrame to CSV
        df.to_csv(output_path, index=False)
    
    print(f"CSV file saved: {output_path}")

if __name__ == "__main__":
    main()