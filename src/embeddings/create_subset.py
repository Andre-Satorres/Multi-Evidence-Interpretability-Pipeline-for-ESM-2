import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from constants import (
    PROTEINS_WITH_SPLIT_TSV_PATH,
    ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH,
    SUBSET_OUT_DIR,
)

# ── fixed random seed ─────────────────────────────────────────────────────────
RANDOM_SEED  = 42
SPLIT_ORDER  = ["train", "val", "test"]


# =============================================================================
#  LOAD
# =============================================================================

def load_data(
    proteins_path: Path = PROTEINS_WITH_SPLIT_TSV_PATH,
    annotations_path: Path = ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read proteins_with_split.tsv and annotations_dedup_with_split.tsv.
    Normalises column names and strip-cleans the split column.
    """
    print("── Loading data ─────────────────────────────────────────")

    def _read(path: Path, label: str) -> pd.DataFrame:
        print(f"  {label}: {path.name} ...", flush=True)
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        # normalise split values
        if "split" in df.columns:
            df["split"] = df["split"].str.strip().str.lower()
        print(f"    {len(df):>10,} rows  ×  {df.shape[1]} cols")
        return df

    proteins    = _read(proteins_path,    "proteins   ")
    annotations = _read(annotations_path, "annotations")
    return proteins, annotations


# =============================================================================
#  CLUSTER-BASED SAMPLING
# =============================================================================

def sample_clusters_by_split(
    proteins: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> dict[str, list]:
    """
    For each split independently, shuffle its clusters and greedily include
    whole clusters until cumulative protein count >= target.

    Returns
    -------
    selected : dict[split_name -> list[cluster_id]]
        The cluster ids selected for each split.
    """
    print(f"\n── Sampling clusters  (fraction={fraction:.4f}, "
          f"seed={RANDOM_SEED}) ────")

    required_cols = {"split", "cluster_id"}
    missing = required_cols - set(proteins.columns)
    if missing:
        raise ValueError(f"proteins table missing columns: {missing}")

    selected: dict[str, list] = {}

    for split in SPLIT_ORDER:
        split_df = proteins[proteins["split"] == split]
        if split_df.empty:
            print(f"  {split}: no proteins found — skipping")
            selected[split] = []
            continue

        n_target = int(len(split_df) * fraction)  # target protein count

        # cluster → protein count for this split
        cluster_sizes = (
            split_df.groupby("cluster_id")
                    .size()
                    .rename("size")
                    .reset_index()
        )

        # shuffle clusters with the RNG (reproducible)
        shuffled = cluster_sizes.sample(
            frac=1, random_state=rng.integers(0, 2**31)
        ).reset_index(drop=True)

        # greedy selection: accumulate whole clusters
        cumulative = 0
        chosen_clusters: list = []
        for _, row in shuffled.iterrows():
            chosen_clusters.append(row["cluster_id"])
            cumulative += row["size"]
            if cumulative >= n_target:
                break
            # if last cluster would overshoot, we include it anyway (per spec)

        actual_pct = cumulative / len(split_df) * 100
        print(f"  {split:5s}: target={n_target:>7,}  "
              f"selected={cumulative:>7,} proteins  "
              f"({actual_pct:.2f}%)  "
              f"from {len(chosen_clusters):,} / "
              f"{len(cluster_sizes):,} clusters")

        selected[split] = chosen_clusters

    return selected


# =============================================================================
#  BUILD PROTEIN SUBSET
# =============================================================================

def build_protein_subset(
    proteins: pd.DataFrame,
    selected_clusters: dict[str, list],
) -> pd.DataFrame:
    """
    Collect all proteins whose cluster_id is in the selected set for their
    respective split.  Returns a new DataFrame with the same columns as the
    input plus no duplicated accessions.
    """
    # build a flat set of (cluster_id, split) → keep protein if it matches
    # We do this per-split to be explicit and safe (each cluster belongs to
    # exactly one split after a clean homology split).
    frames = []
    for split in SPLIT_ORDER:
        clusters = set(selected_clusters.get(split, []))
        if not clusters:
            continue
        mask = (proteins["split"] == split) & (proteins["cluster_id"].isin(clusters))
        frames.append(proteins[mask])

    if not frames:
        return pd.DataFrame(columns=proteins.columns)

    subset = pd.concat(frames, ignore_index=True)
    return subset


# =============================================================================
#  BUILD ANNOTATION SUBSET
# =============================================================================

def build_annotation_subset(
    annotations: pd.DataFrame,
    protein_subset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only annotations whose accession is present in protein_subset.
    Uses a set-based membership test — O(n) rather than a merge for large files.
    """
    acc_col = _acc_col(annotations)
    prot_acc_col = _acc_col(protein_subset)

    selected_accessions = set(protein_subset[prot_acc_col])
    mask = annotations[acc_col].isin(selected_accessions)
    return annotations[mask].reset_index(drop=True)


# =============================================================================
#  SANITY CHECKS
# =============================================================================

def run_sanity_checks(
    protein_subset: pd.DataFrame,
    annotation_subset: pd.DataFrame,
    selected_clusters: dict[str, list],
) -> bool:
    """
    Validate the subset for correctness.  Prints a report and returns True
    if all checks pass, False otherwise.
    """
    print("\n── Sanity checks ───────────────────────────────────────")
    passed = True

    acc_col  = _acc_col(protein_subset)
    ann_col  = _acc_col(annotation_subset)

    # 1. no duplicate accessions in protein subset
    dup_acc = protein_subset[acc_col].duplicated().sum()
    _check(dup_acc == 0,
           "No duplicate accessions in protein subset",
           f"{dup_acc:,} duplicate accessions found!")
    if dup_acc > 0: passed = False

    # 2. no selected cluster spans multiple splits
    if "cluster_id" in protein_subset.columns and "split" in protein_subset.columns:
        splits_per_cluster = (
            protein_subset.groupby("cluster_id")["split"].nunique()
        )
        multi = (splits_per_cluster > 1).sum()
        _check(multi == 0,
               "All selected clusters are confined to one split",
               f"{multi:,} clusters span multiple splits — possible leakage!")
        if multi > 0: passed = False

    # 3. all annotation accessions belong to subset proteins
    subset_accs = set(protein_subset[acc_col])
    orphan_ann  = (~annotation_subset[ann_col].isin(subset_accs)).sum()
    _check(orphan_ann == 0,
           "All annotation accessions match a subset protein",
           f"{orphan_ann:,} annotations reference proteins not in subset!")
    if orphan_ann > 0: passed = False

    # 4. cross-split cluster overlap in selected_clusters dicts
    all_selected: list = []
    for clusters in selected_clusters.values():
        all_selected.extend(clusters)
    dup_clusters = len(all_selected) - len(set(all_selected))
    _check(dup_clusters == 0,
           "No cluster selected in more than one split",
           f"{dup_clusters:,} clusters appear in multiple split selections!")
    if dup_clusters > 0: passed = False

    return passed


def _check(condition: bool, ok_msg: str, fail_msg: str):
    if condition:
        print(f"  ✅ {ok_msg}")
    else:
        print(f"  ❌ {fail_msg}")


# =============================================================================
#  SUMMARISE
# =============================================================================

def summarize_subset(
    proteins_orig: pd.DataFrame,
    proteins_sub:  pd.DataFrame,
    annotations_orig: pd.DataFrame,
    annotations_sub:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Build and print a per-split summary table comparing original vs subset.
    Returns the summary DataFrame.
    """
    print("\n── Summary ─────────────────────────────────────────────")

    rows = []
    for split in SPLIT_ORDER:
        po = proteins_orig[proteins_orig["split"] == split]
        ps = proteins_sub[proteins_sub["split"] == split]
        ao = annotations_orig[annotations_orig["split"] == split]
        as_ = annotations_sub[annotations_sub["split"] == split]

        n_clust_orig = po["cluster_id"].nunique() if "cluster_id" in po.columns else None
        n_clust_sub  = ps["cluster_id"].nunique() if "cluster_id" in ps.columns else None

        rows.append({
            "split":              split,
            "proteins_orig":      len(po),
            "proteins_sub":       len(ps),
            "proteins_pct":       len(ps) / len(po) * 100 if len(po) else 0,
            "clusters_orig":      n_clust_orig,
            "clusters_sub":       n_clust_sub,
            "clusters_pct":       (n_clust_sub / n_clust_orig * 100
                                   if n_clust_orig else None),
            "annotations_orig":   len(ao),
            "annotations_sub":    len(as_),
            "annotations_pct":    len(as_) / len(ao) * 100 if len(ao) else 0,
        })

    summary = pd.DataFrame(rows).set_index("split")

    # pretty print
    fmt_cols = {
        "proteins_orig":    "{:>10,}",
        "proteins_sub":     "{:>10,}",
        "proteins_pct":     "{:>9.2f}%",
        "clusters_orig":    "{:>10,}",
        "clusters_sub":     "{:>10,}",
        "clusters_pct":     "{:>9.2f}%",
        "annotations_orig": "{:>13,}",
        "annotations_sub":  "{:>13,}",
        "annotations_pct":  "{:>12.2f}%",
    }
    header = (f"  {'split':<6}  {'prot_orig':>10}  {'prot_sub':>10}  "
              f"{'prot%':>9}  {'clust_orig':>10}  {'clust_sub':>10}  "
              f"{'clust%':>9}  {'ann_orig':>13}  {'ann_sub':>13}  {'ann%':>12}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for split, row in summary.iterrows():
        def v(col, default="      N/A"):
            val = row[col]
            if pd.isna(val): return default
            return fmt_cols[col].format(val)
        print(f"  {split:<6}  {v('proteins_orig')}  {v('proteins_sub')}  "
              f"{v('proteins_pct')}  {v('clusters_orig')}  {v('clusters_sub')}  "
              f"{v('clusters_pct')}  {v('annotations_orig')}  "
              f"{v('annotations_sub')}  {v('annotations_pct')}")

    return summary


# =============================================================================
#  SAVE OUTPUTS
# =============================================================================

def save_outputs(
    proteins_sub:     pd.DataFrame,
    annotations_sub:  pd.DataFrame,
    summary:          pd.DataFrame,
    fraction:         float,
    tag:              str,
    out_dir:          Path = SUBSET_OUT_DIR,
):
    """Write the three output files and print their paths."""
    pct_label = tag if tag else f"{int(fraction * 100)}pct"

    paths = {
        "proteins":    out_dir / f"proteins_subset_{pct_label}.tsv",
        "annotations": out_dir / f"annotations_subset_{pct_label}.tsv",
        "summary":     out_dir / f"subset_summary_{pct_label}.tsv",
    }

    print("\n── Saving outputs ──────────────────────────────────────")
    proteins_sub.to_csv(paths["proteins"],    sep="\t", index=False)
    annotations_sub.to_csv(paths["annotations"], sep="\t", index=False)
    summary.to_csv(paths["summary"],          sep="\t")

    for label, path in paths.items():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {label:12s}: {path.name}  ({size_mb:.1f} MB)")

    return paths


# =============================================================================
#  KEY FINDINGS
# =============================================================================

def print_key_findings(
    summary: pd.DataFrame,
    sanity_ok: bool,
    fraction: float,
):
    bar = "═" * 60
    print(f"\n{bar}")
    print("  KEY FINDINGS — Subset Report")
    print(bar)
    print(f"  Target fraction : {fraction:.1%}")
    print(f"  Random seed     : {RANDOM_SEED}")
    print()

    for split in SPLIT_ORDER:
        if split not in summary.index:
            continue
        row = summary.loc[split]
        print(f"  [{split}]")
        print(f"    Proteins   : {int(row['proteins_sub']):>8,} / "
              f"{int(row['proteins_orig']):>8,}  "
              f"({row['proteins_pct']:.2f}%)")
        n_co = row['clusters_orig']
        n_cs = row['clusters_sub']
        n_cp = row['clusters_pct']
        if pd.notna(n_co):
            print(f"    Clusters   : {int(n_cs):>8,} / {int(n_co):>8,}  "
                  f"({n_cp:.2f}%)")
        print(f"    Annotations: {int(row['annotations_sub']):>8,} / "
              f"{int(row['annotations_orig']):>8,}  "
              f"({row['annotations_pct']:.2f}%)")

    print()
    if sanity_ok:
        print("  ✅ All sanity checks passed.")
        print("  ✅ No homology leakage detected in subset.")
        print("  ✅ Cluster boundaries fully preserved.")
    else:
        print("  ❌ One or more sanity checks FAILED — inspect output above.")

    print()
    print("  The subset preserves split labels and cluster structure.")
    print("  Suitable for reproducible initial experiments.")
    print(bar)


# =============================================================================
#  HELPERS
# =============================================================================

def _acc_col(df: pd.DataFrame) -> str:
    """Return the accession column name, trying common variants."""
    return next(
        (c for c in df.columns if c in ("accession", "entry", "protein_id")),
        df.columns[0],
    )


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create a cluster-preserving subset of the protein dataset."
    )
    parser.add_argument(
        "--fraction", type=float, default=0.10,
        help="Fraction of each split to retain (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Optional label for output file names (default: auto from fraction)"
    )
    parser.add_argument(
        "--proteins", type=Path, default=PROTEINS_WITH_SPLIT_TSV_PATH,
        help="Path to proteins_with_split.tsv"
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH,
        help="Path to annotations_dedup_with_split.tsv"
    )
    args = parser.parse_args()

    if not (0 < args.fraction <= 1.0):
        parser.error("--fraction must be between 0 (exclusive) and 1.0")

    # ── fixed RNG ─────────────────────────────────────────────────────────────
    # We use numpy's modern Generator API for reproducibility.
    rng = np.random.default_rng(RANDOM_SEED)

    # ── pipeline ──────────────────────────────────────────────────────────────
    proteins, annotations = load_data(args.proteins, args.annotations)

    selected_clusters = sample_clusters_by_split(proteins, args.fraction, rng)

    proteins_sub     = build_protein_subset(proteins, selected_clusters)
    annotations_sub  = build_annotation_subset(annotations, proteins_sub)

    summary    = summarize_subset(proteins, proteins_sub,
                                  annotations, annotations_sub)
    sanity_ok  = run_sanity_checks(proteins_sub, annotations_sub,
                                   selected_clusters)
    save_outputs(proteins_sub, annotations_sub, summary,
                 args.fraction, args.tag)
    print_key_findings(summary, sanity_ok, args.fraction)


if __name__ == "__main__":
    main()