# copy_chunk.py
# Usage: python copy_chunk.py 02
# Copies chunk 02 images to C:\geosolve_images\ and creates local CSV

import sys
import pandas as pd
import shutil
from pathlib import Path

chunk_num = sys.argv[1] if len(sys.argv) > 1 else "02"
input_csv  = f"Data/chunks/train_chunk_{chunk_num}.csv"
output_csv = f"Data/chunks/train_chunk_{chunk_num}_local.csv"

print(f"Processing chunk {chunk_num}...")
df = pd.read_csv(input_csv)
print(f"Rows: {len(df):,}")

dest_root = Path("C:/geosolve_images")
success   = 0
failed    = 0

for i, filepath in enumerate(df["Filepath"]):
    try:
        src = Path(filepath)
        if src.exists():
            rel_parts = src.parts[1:]
            dest = dest_root.joinpath(*rel_parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            success += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

    if (i+1) % 5000 == 0:
        print(f"  {i+1:,}/{len(df):,} | copied: {success:,} | failed: {failed:,}")

print(f"Done! Copied: {success:,} Failed: {failed:,}")

# create local CSV
def update_path(filepath):
    src = Path(filepath)
    rel_parts = src.parts[1:]
    return str(Path("C:/geosolve_images").joinpath(*rel_parts))

df["Filepath"] = df["Filepath"].apply(update_path)
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")