
import os
from pathlib import Path
import torch


BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
IMAGES_TRAIN  = DATA_DIR / "images" / "train"
IMAGES_VAL    = DATA_DIR / "images" / "val"
LABELS_TRAIN  = DATA_DIR / "labels" / "train"
LABELS_VAL    = DATA_DIR / "labels" / "val"
RUNS_DIR      = BASE_DIR / "runs"
TAHMINLER_DIR = BASE_DIR / "tahminler"
GORSELLER_DIR = BASE_DIR / "gorseller"


for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL,
          RUNS_DIR, TAHMINLER_DIR, GORSELLER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


NDJSON_DOSYA = BASE_DIR / "dental-prosthetic-crown-detection-x-rays.ndjson"
DATA_YAML    = DATA_DIR / "data.yaml"


ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
ROBOFLOW_WORKSPACE = "dental-workspace"
ROBOFLOW_PROJECT   = "dental-crown-obb"
ROBOFLOW_VERSION   = 1
ROBOFLOW_FORMAT    = "yolov8-obb"


MODEL_ADI    = "yolov8n-obb.pt"
PROJE_ADI    = "dental_crown_obb"
DENEY_ADI    = "yolov8n_run1"


EPOCH        = 50
BATCH        = 8
IMGSZ        = 640
LR0          = 0.01
LRF          = 0.01
MOMENTUM     = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCH = 3
PATIENCE     = 50


MOSAIC   = 1.0
FLIPLR   = 0.5
FLIPUD   = 0.5
DEGREES  = 10
SCALE    = 0.5


CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
MAX_DET        = 300


DOWNLOAD_WORKERS = 8
DOWNLOAD_TIMEOUT = 30


DEVICE = "0" if torch.cuda.is_available() else "cpu"
DEVICE_TR = "GPU" if torch.cuda.is_available() else "CPU"


NUM_CLASSES  = 1
CLASS_NAMES  = ["crown"]
TRAIN_SPLIT  = 251
VAL_SPLIT    = 151
TOTAL_IMAGES = 402
