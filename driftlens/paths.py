"""Central place for every filesystem path the project touches."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASETS = WORKSPACE_ROOT / "03_Datasets"

CLPSYCH_TRAIN = DATASETS / "CLPsych_timelines" / "train_tasks12"
CLPSYCH_TEST = DATASETS / "CLPsych_timelines" / "test_tasks12nolabels"

ERISK_T2_DIR = (DATASETS / "eRisk2026" / "task2-contextualized-depression"
                / "eRisk26-task2-trainingdata" / "final-eriskt2-dataset-with-ground-truth")
ERISK_T2_JSON = ERISK_T2_DIR / "all_combined"
ERISK_T2_LABELS = ERISK_T2_DIR / "shuffled_ground_truth_labels.txt"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"

ERISK_USERS = DATA_PROCESSED / "erisk26_t2_users.jsonl.gz"

for _p in (DATA_PROCESSED, MODELS, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
