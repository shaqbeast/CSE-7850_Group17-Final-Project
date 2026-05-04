import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

src = "/storage/home/hcoda1/8/sverma87/scratch/GGD_CSE-7850/data/exp1_CLM_unphased_tensor.csv"
dst = "/storage/home/hcoda1/8/sverma87/scratch/GGD_CSE-7850/data/exp1_CLM_unphased_tensor.parquet"

print("Reading CSV...")
read_opts = pa_csv.ReadOptions(block_size=512 * 1024 * 1024)  # 512MB blocks for wide file
table = pa_csv.read_csv(src, read_options=read_opts)
print(f"Shape: {table.num_rows} rows x {table.num_columns} cols")
print(f"Columns (first 10): {table.column_names[:10]}")
print("Writing Parquet...")
pq.write_table(table, dst, compression="snappy")
print(f"Done. Saved to: {dst}")
