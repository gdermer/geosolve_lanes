from pathlib import Path

# ====== Paths =======
CSV_PATH = (
    r"J:\- Macros\AI-LaneDetector\lane_analysis"
    r"\coordinates_report_with_lane_combined_with_ignore_full1.csv"
)

DATA_DIR = Path("Data")

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "val.csv"
TEST_DIR = DATA_DIR / "test.csv"

# saving training data folder here
CHECKPOINT_DIR = Path("checkpoints")

# if already exists doesnt crash
CHECKPOINT_DIR.mkdir(exist_ok = True)

# resize images for training
IMG_SUZE = 224

# start with common cases later on i'll add the edge cases
LANE_CLASSES = {
    "1": 0, #lane 1
    "2": 1, # lane 2
    "3" : 2,    # lane 3
    "SK1": 3    # side kurb lane

}

IDX_TO_LANE = {v: k for k, v in LANE_CLASSES.items()}   # switch key and value

N_CLASSES = len(LANE_CLASSES)   # 4 at the miment

MODEL_NAME = "efficientnet_b0" # from TIMM library,  b0 = smallest and fastest EfficientNet,  upgrade to b2 or b4 later if accuracy needs improvement

BATCH_SIZE = 32 # images processed together

SEQ_LEN = 10     #consecutive fed to the model at once

LR_PHASE1 = 1e-3 # learning data rate phase 1

LR_PHASE2 = 1e-5  # leaning data rate phase 2

EPOCHS_PHASE1 = 5  # traiing epochs for phase 1

EPOCHS_PHASE2 = 30  # training epochs for phase 2

EARLY_STOP_PATIENCE = 7 # stop if value accuracy doesnt improve for 7 epochs

SEED = 42    # new random (42)










