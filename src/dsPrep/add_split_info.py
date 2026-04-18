import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import (
    PROTEINS_TSV_PATH,
    ANNOTATIONS_TSV_PATH,
    SPLITS_TSV_PATH,
    PROTEINS_WITH_SPLIT_TSV_PATH,
    ANNOTATIONS_WITH_SPLIT_TSV_PATH,
)

print("Reading inputs...")
proteins = pd.read_csv(PROTEINS_TSV_PATH, sep="\t")
annotations = pd.read_csv(ANNOTATIONS_TSV_PATH, sep="\t")
splits = pd.read_csv(SPLITS_TSV_PATH, sep="\t")

# Padronização de nomes
if "Entry" in proteins.columns:
    proteins = proteins.rename(columns={"Entry": "accession"})
if "Sequence" in proteins.columns:
    proteins = proteins.rename(columns={"Sequence": "sequence"})
if "Length" in proteins.columns:
    proteins = proteins.rename(columns={"Length": "length"})

if "Entry" in annotations.columns:
    annotations = annotations.rename(columns={"Entry": "accession"})

# Limpeza de ids
for df in [proteins, annotations, splits]:
    if "accession" in df.columns:
        df["accession"] = df["accession"].astype(str).str.strip()

if "cluster_id" in splits.columns:
    splits["cluster_id"] = splits["cluster_id"].astype(str).str.strip()
if "split" in splits.columns:
    splits["split"] = splits["split"].astype(str).str.strip()

print("Merging proteins...")
proteins_with_split = proteins.merge(
    splits[["accession", "cluster_id", "split"]],
    on="accession",
    how="inner",
    validate="one_to_one"
)

print("Merging annotations...")
annotations_with_split = annotations.merge(
    proteins_with_split[["accession", "cluster_id", "split"]],
    on="accession",
    how="inner",
    validate="many_to_one"
)

print("Saving...")
proteins_with_split.to_csv(PROTEINS_WITH_SPLIT_TSV_PATH, sep="\t", index=False)
annotations_with_split.to_csv(ANNOTATIONS_WITH_SPLIT_TSV_PATH, sep="\t", index=False)

print("Done.")
print(f"proteins_with_split:    {len(proteins_with_split):,}")
print(f"annotations_with_split: {len(annotations_with_split):,}")