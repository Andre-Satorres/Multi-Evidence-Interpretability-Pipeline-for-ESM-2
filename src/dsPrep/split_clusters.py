import pandas as pd
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import CLUSTERS_TSV_PATH, SPLITS_TSV_PATH

SEED = 42
random.seed(SEED)

if len(sys.argv) != 3:
    print("Usage: python split_clusters.py <train_percentage> <val_percentage>")
    sys.exit(1)

train_percentage = float(sys.argv[1])
val_percentage = float(sys.argv[2])

if train_percentage <= 0 or val_percentage <= 0 or train_percentage + val_percentage >= 1:
    raise ValueError("train_percentage and val_percentage must be > 0 and sum to < 1")

clusters = pd.read_csv(
    CLUSTERS_TSV_PATH,
    sep="\t",
    header=None,
    names=["cluster_id", "accession"]
)

cluster_sizes = (
    clusters.groupby("cluster_id", as_index=False)["accession"]
    .count()
    .rename(columns={"accession": "size"})
)

cluster_list = cluster_sizes.to_dict("records")
random.shuffle(cluster_list)

total = cluster_sizes["size"].sum()
train_target = train_percentage * total
val_target = val_percentage * total

split_map = {}

train_count = 0
val_count = 0
test_count = 0

for row in cluster_list:
    cid = row["cluster_id"]
    size = row["size"]

    if train_count < train_target:
        split_map[cid] = "train"
        train_count += size
    elif val_count < val_target:
        split_map[cid] = "val"
        val_count += size
    else:
        split_map[cid] = "test"
        test_count += size

clusters["split"] = clusters["cluster_id"].map(split_map)

if clusters["split"].isna().any():
    raise RuntimeError("Some cluster_ids were not assigned a split.")

clusters.to_csv(SPLITS_TSV_PATH, sep="\t", index=False)

print(clusters["split"].value_counts())
print(clusters["cluster_id"].nunique(), "clusters")