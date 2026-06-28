import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# test saving to exact same path you're using in train.py
save_path = r"C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\checkpoints\training_progress.png"

print(f"Saving to: {save_path}")
print(f"Folder exists: {Path(save_path).parent.exists()}")

# create a simple test plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot([1, 2, 3], [1, 4, 9], "b-o")
ax.set_title("Test plot")

try:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved successfully!")
    print(f"File exists after save: {Path(save_path).exists()}")
    print(f"File size: {Path(save_path).stat().st_size} bytes")
except Exception as e:
    print(f"ERROR: {e}")