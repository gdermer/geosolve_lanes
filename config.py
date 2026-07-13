from pathlib import Path

# ====== Paths =======
CSV_PATH = (
    r"J:\- Macros\AI-LaneDetector\lane_analysis"
    r"\coordinates_report_with_lane_combined_with_ignore_full1.csv"
)

DATA_DIR = Path("Data")

TRAIN_CSV = DATA_DIR / "chunks_v2/train_chunk_v2_02_local.csv"
VAL_CSV   = DATA_DIR / "val_real.csv"
TEST_CSV  = DATA_DIR / "test.csv"

# saving training data folder here
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# resize images for training
IMG_SIZE = 224

# lane classes
LANE_CLASSES = {
    "1":   0,
    "2":   1,
    "3":   2,
    "SK1": 3,
}

IDX_TO_LANE = {v: k for k, v in LANE_CLASSES.items()}

N_CLASSES = len(LANE_CLASSES)

MODEL_NAME = "efficientnet_b0"

BATCH_SIZE          = 64
NUM_WORKERS         = 4
LR_PHASE1           = 1e-3
LR_PHASE2           = 1e-5
EPOCHS_PHASE1       = 5
EPOCHS_PHASE2       = 30
EARLY_STOP_PATIENCE = 7
SEQ_LEN             = 10
SEED                = 42

TEST_SEGMENTS = [
    "250357 UHCC_25_LMD",
]

VAL_SEGMENTS = [
    "250821 KaikouraDC_Network25_LMD Demo",
    "250846 DunedinCC LMD Network26",
]

OSM_DATA_PATH          = Path("Data/nz_roads.graphml")
OSM_SEARCH_RADIUS      = 50
OSM_LANES_FORWARD_COL  = "lanes_forward"
OSM_ROAD_TYPE_COL      = "road_type"
OSM_ONEWAY_COL         = "oneway"