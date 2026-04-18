from pathlib import Path
import numpy as np
import pandas as pd

outdir = Path("outputs/feature_alignment_streaming_all")

# carrega resultados
results_df = pd.read_parquet(outdir / "alignment_scores.parquet")
print(f"Loaded {len(results_df):,} (feature, annot) pairs")

# per-feature: melhor auprc entre todos os annotation types
feat_summary = (
    results_df
    .sort_values("auprc_mean", ascending=False)
    .groupby("feature_id")
    .first()
    .reset_index()
    .rename(columns={"annot_type": "best_annot", "auprc_mean": "best_auprc"})
)
feat_summary["mystery_score"] = 1.0 - feat_summary["best_auprc"]

# contagens de annots acima de threshold por feature
counts = (
    results_df[results_df["auprc_mean"] > 0.30]
    .groupby("feature_id")
    .size()
    .rename("n_annots_above_03")
)
feat_summary = feat_summary.merge(counts, on="feature_id", how="left")
feat_summary["n_annots_above_03"] = feat_summary["n_annots_above_03"].fillna(0).astype(int)

feat_summary = feat_summary.sort_values("best_auprc")
feat_summary.to_csv(outdir / "feature_triage_summary.tsv", sep="\t", index=False)
print(f"Saved feature_triage_summary.tsv ({len(feat_summary):,} features)")

print(f"\nDistribution of best_auprc:")
for thr in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90]:
    n = (feat_summary["best_auprc"] < thr).sum()
    pct = 100 * n / len(feat_summary)
    print(f"  < {thr:.2f}: {n:>5,} features ({pct:.1f}%)")

novel = feat_summary[feat_summary["best_auprc"] < 0.25].sort_values("mystery_score", ascending=False)
print(f"\nTotal novel candidates: {len(novel):,}")
if not novel.empty:
    print(novel.head(20)[["feature_id","best_auprc","best_annot",
                           "n_annots_above_03"]].to_string(index=False))