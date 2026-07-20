import pandas as pd
from pathlib import Path

# ================================================================
# SETTINGS
# ================================================================

ROOT_FOLDER = r"J:\Testing\260234.01 AT_Data_Collection_LMD_26-27\Photos\QFM954-2026-07-15-NZST"

OUTPUT_CSV = r"J:\geosolve_training\Data\new_survey_to_predict.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ================================================================
# SCAN ALL SUBFOLDERS FOR IMAGES
# ================================================================

root = Path(ROOT_FOLDER)
print(f"Scanning: {root}")

all_filepaths = []
for ext in IMAGE_EXTENSIONS:
    found = list(root.rglob(f"*{ext}"))
    print(f"  Found {len(found):,} {ext} files")
    all_filepaths.extend(found)

print(f"\nTotal images found: {len(all_filepaths):,}")

if len(all_filepaths) == 0:
    print("WARNING: No images found! Check the path and extensions.")
    exit()

# ================================================================
# EXTRACT BEARING FROM FILENAME
# ================================================================
# GeoSolve filename pattern:
# 260078-2026-05-09-04-14-29-813-3715.1818S-17502.0811E-100.2-1-0-QFM954---F-.jpg
#                                                          ^^^^^ bearing

def extract_bearing_from_filename(filepath):
    try:
        name = Path(filepath).stem
        parts = name.split("-")
        for i, part in enumerate(parts):
            if part.endswith("E") and i + 1 < len(parts):
                return float(parts[i + 1])
    except:
        pass
    return 0.0

# ================================================================
# BUILD CSV
# ================================================================

rows = []
for fp in all_filepaths:
    rows.append({
        "Filepath": str(fp),
        "Filename": fp.name,
        "Bearing": extract_bearing_from_filename(fp),
        "Ignore": 0,
    })

df = pd.DataFrame(rows)
print(f"\nCSV preview:")
print(df.head())
print(f"\nExample bearing values: {df['Bearing'].head(10).tolist()}")

Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"Total rows: {len(df):,}")