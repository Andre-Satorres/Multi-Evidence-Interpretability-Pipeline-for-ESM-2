"""
Addresses the selection bias in per_annot_summary.tsv: in the original
pipeline, the best feature per annotation subtype is chosen using the
same test set used for evaluation, inflating reported AUPRC.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Unbiased feature selection: val for selection, test for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--val-scores", type=Path, required=True,
                   help="alignment_scores.parquet from the validation run.")
    p.add_argument("--test-scores", type=Path, required=True,
                   help="alignment_scores.parquet from the test run.")
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ── Load val and test scores ──────────────────────────────────────────────
    log.info("Loading validation scores ...")
    val_df = pd.read_parquet(args.val_scores)
    log.info(f"  {len(val_df):,} (feature, annot) pairs, "
             f"{val_df['annot_type'].nunique()} subtypes, "
             f"{val_df['feature_id'].nunique()} features")

    log.info("Loading test scores ...")
    test_df = pd.read_parquet(args.test_scores)
    log.info(f"  {len(test_df):,} (feature, annot) pairs, "
             f"{test_df['annot_type'].nunique()} subtypes, "
             f"{test_df['feature_id'].nunique()} features")

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage A: select best feature per annotation type on VALIDATION set
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\nStage A: feature selection on validation set ...")

    # For each annotation subtype, pick the feature with highest val AUPRC
    idx_best_val = val_df.groupby("annot_type")["auprc_mean"].idxmax()
    val_best = val_df.loc[idx_best_val].copy()

    selection = val_best[["annot_type", "feature_id", "auprc_mean",
                          "n_proteins", "n_clusters"]].copy()
    selection = selection.rename(columns={
        "feature_id": "selected_feature_id",
        "auprc_mean": "val_auprc",
        "n_proteins": "val_n_proteins",
        "n_clusters": "val_n_clusters",
    })

    log.info(f"  Selected {len(selection)} (feature, annot) pairs from validation")

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage B: look up test AUPRC for the FROZEN selected pairs
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\nStage B: test evaluation for selected pairs (frozen selection) ...")

    # Prepare test columns for merge
    test_cols = test_df[["annot_type", "feature_id",
                         "auprc_mean", "auprc_median",
                         "odds_ratio", "fisher_p",
                         "n_proteins", "n_clusters",
                         "tp", "fp", "fn", "tn"]].copy()
    test_cols = test_cols.rename(columns={
        "feature_id":   "selected_feature_id",
        "auprc_mean":   "test_auprc",
        "auprc_median": "test_auprc_median",
        "odds_ratio":   "test_odds_ratio",
        "fisher_p":     "test_fisher_p",
        "n_proteins":   "test_n_proteins",
        "n_clusters":   "test_n_clusters",
        "tp": "test_tp", "fp": "test_fp",
        "fn": "test_fn", "tn": "test_tn",
    })

    # Left join: keep all val-selected pairs, attach test scores where available
    result_df = selection.merge(
        test_cols,
        on=["annot_type", "selected_feature_id"],
        how="left",
    )

    n_found   = result_df["test_auprc"].notna().sum()
    n_missing = result_df["test_auprc"].isna().sum()

    result_df = result_df.sort_values("test_auprc", ascending=False, na_position="last")
    result_df.to_csv(args.outdir / "per_annot_val_test.tsv", sep="\t", index=False)

    log.info(f"  Found test scores for {n_found}/{len(selection)} subtypes")
    if n_missing:
        log.warning(f"  Missing test scores for {n_missing} subtypes "
                     "(annotation may not meet min-protein threshold in test)")

    # ══════════════════════════════════════════════════════════════════════════
    #  Feature-level triage (analogous to feature_triage_summary.tsv)
    # ══════════════════════════════════════════════════════════════════════════
    valid = result_df.dropna(subset=["test_auprc"])
    feat_best = {}
    for _, row in valid.iterrows():
        fid = int(row["selected_feature_id"])
        if fid not in feat_best or row["test_auprc"] > feat_best[fid]["best_test_auprc"]:
            feat_best[fid] = {
                "feature_id":       fid,
                "best_test_auprc":  row["test_auprc"],
                "best_val_auprc":   row["val_auprc"],
                "best_annot":       row["annot_type"],
            }

    if feat_best:
        feat_df = pd.DataFrame(feat_best.values()).sort_values(
            "best_test_auprc", ascending=True)
        feat_df.to_csv(args.outdir / "feature_triage_val_test.tsv",
                       sep="\t", index=False)
    else:
        feat_df = pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    #  Summary statistics
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'═'*60}")

    if len(valid) > 0:
        val_mean  = valid["val_auprc"].mean()
        test_mean = valid["test_auprc"].mean()
        gap       = (valid["val_auprc"] - valid["test_auprc"]).mean()
        n_test_08 = int((valid["test_auprc"] >= 0.8).sum())
        n_val_08  = int((valid["val_auprc"] >= 0.8).sum())

        log.info(f"  Val  AUPRC — mean: {val_mean:.3f}, "
                 f"median: {valid['val_auprc'].median():.3f}")
        log.info(f"  Test AUPRC — mean: {test_mean:.3f}, "
                 f"median: {valid['test_auprc'].median():.3f}")
        log.info(f"  Selection gap (val − test): {gap:+.3f}")
        log.info(f"  Subtypes with val  AUPRC ≥ 0.8: {n_val_08}/{len(valid)}")
        log.info(f"  Subtypes with test AUPRC ≥ 0.8: {n_test_08}/{len(valid)}")

        # Compare against naïve (test-only) selection
        if "auprc_mean" in test_df.columns:
            naive_best = test_df.loc[
                test_df.groupby("annot_type")["auprc_mean"].idxmax()
            ]
            naive_annots = set(valid["annot_type"])
            naive_sub = naive_best[naive_best["annot_type"].isin(naive_annots)]
            if len(naive_sub) > 0:
                naive_mean = naive_sub["auprc_mean"].mean()
                log.info(f"  Naïve (test-selected) AUPRC mean: {naive_mean:.3f}")
                log.info(f"  Optimistic bias (naïve − unbiased): "
                         f"{naive_mean - test_mean:+.3f}")
    else:
        val_mean = test_mean = gap = float("nan")
        n_test_08 = n_val_08 = 0

    log.info(f"{'═'*60}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "timestamp":            datetime.now().isoformat(),
        "val_scores_path":      str(args.val_scores),
        "test_scores_path":     str(args.test_scores),
        "n_subtypes_val":       int(val_df["annot_type"].nunique()),
        "n_subtypes_test":      int(test_df["annot_type"].nunique()),
        "n_subtypes_selected":  len(selection),
        "n_test_found":         int(n_found),
        "n_test_missing":       int(n_missing),
        "val_auprc_mean":       float(val_mean),
        "test_auprc_mean":      float(test_mean),
        "selection_gap_mean":   float(gap),
        "n_subtypes_test_ge08": n_test_08,
        "n_features_selected":  len(feat_best),
    }
    with open(args.outdir / "selection_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"\n✅ Done → {args.outdir}")
    log.info(f"   per_annot_val_test.tsv       — {len(result_df)} subtypes")
    log.info(f"   feature_triage_val_test.tsv   — {len(feat_df)} features")
    log.info(f"   selection_summary.json        — metadata")


if __name__ == "__main__":
    main()
