import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import ANNOTATIONS_TSV_PATH, ANNOTATIONS_DEDUP_TSV_PATH, ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH, PROTEINS_WITH_SPLIT_TSV_PATH

ann = pd.read_csv(ANNOTATIONS_TSV_PATH, sep="\t")
dup_cols = ["accession", "feature_type", "start", "end"]

before = len(ann)
ann = ann.drop_duplicates(subset=dup_cols)
after = len(ann)

print(f"Removed {before - after:,} duplicates")

# ------------

ann.to_csv(ANNOTATIONS_DEDUP_TSV_PATH, sep="\t", index=False)

annotations = pd.read_csv(ANNOTATIONS_DEDUP_TSV_PATH, sep="\t")
proteins = pd.read_csv(PROTEINS_WITH_SPLIT_TSV_PATH, sep="\t")

annotations = annotations.merge(
    proteins[["accession", "cluster_id", "split"]],
    on="accession",
    how="inner"
)

annotations.to_csv(
    ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH,
    sep="\t",
    index=False
)