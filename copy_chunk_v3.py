# copy_chunk_v3.py
# Usage: python copy_chunk_v3.py 01
# Copies chunks_v3 images to C:\geosolve_images\ and creates local CSV

import sys
import pandas as pd
import shutil
from pathlib import Path

chunk_num  = sys.argv[1] if len(sys.argv) > 1 else "01"
input_csv  = f"Data/chunks_v3/train_chunk_v3_{chunk_num}.csv"
output_csv = f"Data/chunks_v3/train_chunk_v3_{chunk_num}_local.csv"

print(f"Processing chunk {chunk_num}...")
df = pd.read_csv(input_csv)
print(f"Rows: {len(df):,}")

dest_root = Path("C:/geosolve_images")
if dest_root.exists():
    print("Deleting old images...")
    shutil.rmtree(dest_root)
    print("Old images deleted")

dest_root.mkdir(exist_ok=True)

success = 0
failed  = 0

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
    except:
        failed += 1

    if (i+1) % 5000 == 0:
        print(f"  {i+1:,}/{len(df):,} | copied: {success:,} | failed: {failed:,}")

print(f"\nDone! Copied: {success:,} Failed: {failed:,}")

def update_path(filepath):
    src = Path(filepath)
    rel_parts = src.parts[1:]
    return str(Path("C:/geosolve_images").joinpath(*rel_parts))

df["Filepath"] = df["Filepath"].apply(update_path)
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")