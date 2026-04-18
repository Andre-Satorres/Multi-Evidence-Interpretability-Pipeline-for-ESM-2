"""
triage_novel_features.py — PIPLV2 / sae
=========================================
Triage script: identify SAE features with HIGH causal necessity
but LOW alignment with any known UniProt annotation.

These are candidates for "novel biological concepts" — features that
the model uses functionally but that current databases have not catalogued.

Step 1 (this script, no GPU needed):
  - Load alignment_scores.parquet (already computed)
  - For each feature, compute best_auprc across ALL annotation types
  - Rank by best_auprc ascending → low-annotation features
  - Output: triage_candidates.tsv with features ranked by "mystery score"

Step 2 (separate, needs GPU):
  - Run ablation for the top-N triage candidates
  - Features with high ablation_contrast AND low best_auprc = novel concept candidates

Step 3 (analysis):
  - Visualise: scatter plot of best_auprc vs ablation_contrast
  - Quadrant of interest: low AUPRC, high ablation = novel concepts
  - Inspect top candidates: what proteins activate them? Any pattern?

Usage
-----
  # Step 1 only (fast, no GPU):
  python src/sae/triage_novel_features.py \\
      --alignment-dir outputs/feature_alignment \\
      --outdir        outputs/novel_features

  # Step 1 + 2 + 3 (full pipeline, needs GPU):
  python src/sae/triage_novel_features.py \\
      --alignment-dir  outputs/feature_alignment \\
      --checkpoint     outputs/sae_runs/latent8192_l1_3e-05_lr_3e-04/checkpoints/best.pt \\
      --esm2-model     esm2_t33_650M_UR50D \\
      --esm2-layer     33 \\
      --annotations    data/annotations_dedup_with_split.tsv \\
      --split          test \\
      --run-ablation \\
      --top-candidates 30 \\
      --outdir         outputs/novel_features
"""

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import OUT_DIR, ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH

torch = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--alignment-dir", type=Path,
                   default=OUT_DIR / "feature_alignment")
    p.add_argument("--checkpoint",    type=Path, default=None,
                   help="SAE checkpoint. Required if --run-ablation.")
    p.add_argument("--esm2-model", default="esm2_t33_650M_UR50D",
                   help="ESM-2 model name for on-the-fly embedding.")
    p.add_argument("--esm2-layer", type=int, default=33,
                   help="ESM-2 transformer layer to extract embeddings from.")
    p.add_argument("--max-seq-len", type=int, default=1022,
                   help="Truncate sequences longer than this.")
    p.add_argument("--annotations",   type=Path,
                   default=ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH)
    p.add_argument("--proteins",      type=Path,
                   default=OUT_DIR.parent / "data" / "proteins_with_split.tsv")
    p.add_argument("--split",         default="test")
    p.add_argument("--run-ablation",  action="store_true",
                   help="Run ablation for top triage candidates (needs GPU).")
    p.add_argument("--top-candidates", type=int, default=30,
                   help="How many low-AUPRC features to run ablation on.")
    p.add_argument("--max-proteins",  type=int, default=30,
                   help="Max proteins per feature for ablation (speed).")
    p.add_argument("--auprc-threshold", type=float, default=0.25,
                   help="Features with best_auprc < this are 'low-annotation'.")
    p.add_argument("--activation-threshold", type=float, default=0.1)
    p.add_argument("--outdir",        type=Path,
                   default=OUT_DIR / "novel_features")
    p.add_argument("--device",        default="auto")
    return p.parse_args()


# =============================================================================
#  STEP 1 — TRIAGE FROM ALIGNMENT SCORES (no GPU)
# =============================================================================

def triage_from_alignment(alignment_dir: Path, auprc_threshold: float) -> pd.DataFrame:
    """
    Load alignment_scores.parquet and compute per-feature summary:
      - best_auprc:       max AUPRC across all annotation types
      - best_annot:       annotation type with highest AUPRC
      - best_or:          max odds ratio across all annotation types
      - n_annots_above05: number of annotation types with AUPRC > 0.5
      - mystery_score:    1 - best_auprc  (higher = more mysterious)
    """
    parquet_path = alignment_dir / "alignment_scores.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"alignment_scores.parquet not found at {parquet_path}")

    log.info(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    log.info(f"  {len(df):,} (feature, annotation) pairs, "
             f"{df['feature_id'].nunique():,} features, "
             f"{df['annot_type'].nunique()} annotation types")

    # per-feature aggregation
    feat_summary = []
    for fid, grp in df.groupby("feature_id"):
        best_row  = grp.loc[grp["auprc_mean"].idxmax()]
        feat_summary.append({
            "feature_id":        int(fid),
            "best_auprc":        float(best_row["auprc_mean"]),
            "best_annot":        best_row["annot_type"],
            "best_or":           float(grp["odds_ratio"].replace([np.inf, -np.inf],
                                                                  np.nan).max()),
            "mean_auprc_all":    float(grp["auprc_mean"].mean()),
            "n_annots_above_03": int((grp["auprc_mean"] > 0.30).sum()),
            "n_annots_above_05": int((grp["auprc_mean"] > 0.50).sum()),
            "mystery_score":     float(1.0 - best_row["auprc_mean"]),
        })

    summary = pd.DataFrame(feat_summary).sort_values("best_auprc")
    log.info(f"\n  Distribution of best_auprc across {len(summary):,} features:")
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        n = (summary["best_auprc"] < threshold).sum()
        pct = 100 * n / len(summary)
        log.info(f"    best_auprc < {threshold:.2f}: {n:>5,} features ({pct:.1f}%)")

    low_annot = summary[summary["best_auprc"] < auprc_threshold]
    log.info(f"\n  → {len(low_annot):,} features with best_auprc < {auprc_threshold} "
             f"(low-annotation candidates)")

    return summary


# =============================================================================
#  STEP 2 — ABLATION FOR TRIAGE CANDIDATES (needs GPU)
# =============================================================================

def load_sae(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model_state_dict"]
    W_enc = sd["encoder.weight"].float().to(device)
    b_enc = sd["encoder.bias"].float().to(device)
    W_dec = sd["decoder.weight"].float().to(device)
    b_dec = sd["decoder.bias"].float().to(device)
    return W_enc, b_enc, W_dec, b_dec


from esm_utils import load_esm2
from esm_utils import get_embedding as _get_embedding_live


def ablation_contrast_for_feature(
    feature_id: int,
    accs: list[str],
    get_emb,
    W_enc, b_enc, W_dec, b_dec,
    activation_threshold: float = 0.1,
) -> dict:
    """
    Annotation-free ablation: measure ablation delta distribution
    at HIGH-activation vs LOW-activation residues within each protein.

    This is annotation-agnostic — we don't need a ground-truth mask.
    We use the feature's own activation pattern as the "positive" set:
      high_act residues: acts > activation_threshold
      low_act residues:  acts <= activation_threshold

    If the feature is causally important for what it activates on,
    delta should be higher at high_act residues than low_act residues.

    Returns summary dict with contrast and frac_positive.
    """
    contrasts = []
    n_proteins_processed = 0

    for acc in accs:
        emb = get_emb(acc)
        L   = emb.shape[0]

        with torch.no_grad():
            x     = torch.tensor(emb).float().to(W_enc.device)
            z     = torch.relu(x @ W_enc.T + b_enc)
            z_abl = z.clone()
            z_abl[:, feature_id] = 0.0
            x_hat     = (z     @ W_dec.T + b_dec).cpu().numpy()
            x_hat_abl = (z_abl @ W_dec.T + b_dec).cpu().numpy()

        delta_norm = np.linalg.norm(x_hat - x_hat_abl, axis=1)  # [L]
        acts = z[:, feature_id].cpu().numpy()                     # [L]

        hi_idx = acts > activation_threshold
        lo_idx = ~hi_idx

        if hi_idx.sum() == 0 or lo_idx.sum() == 0:
            continue

        mean_hi = float(delta_norm[hi_idx].mean())
        mean_lo = float(delta_norm[lo_idx].mean())
        contrasts.append(mean_hi - mean_lo)
        n_proteins_processed += 1

    if not contrasts:
        return {"abl_contrast_self": float("nan"),
                "abl_frac_pos_self": float("nan"),
                "n_proteins_abl":    0}

    return {
        "abl_contrast_self": float(np.mean(contrasts)),
        "abl_frac_pos_self": float(np.mean([c > 0 for c in contrasts])),
        "n_proteins_abl":    n_proteins_processed,
    }


def run_ablation_triage(
    candidates: pd.DataFrame,
    seq_by_acc: dict,
    get_emb,
    W_enc, b_enc, W_dec, b_dec,
    max_proteins: int,
    activation_threshold: float,
) -> pd.DataFrame:
    """Run annotation-free ablation for each candidate feature."""
    all_accs = list(seq_by_acc.keys())[:max_proteins * 5]

    results = []
    total = len(candidates)
    for i, (_, row) in enumerate(candidates.iterrows()):
        fid = int(row["feature_id"])
        log.info(f"  [{i+1}/{total}] f{fid} (best_auprc={row['best_auprc']:.3f}) ...")

        abl = ablation_contrast_for_feature(
            feature_id=fid,
            accs=all_accs[:max_proteins],
            get_emb=get_emb,
            W_enc=W_enc, b_enc=b_enc,
            W_dec=W_dec, b_dec=b_dec,
            activation_threshold=activation_threshold,
        )
        results.append({**row.to_dict(), **abl})

    return pd.DataFrame(results)


# =============================================================================
#  STEP 3 — VISUALISATION
# =============================================================================

def plot_triage_scatter(summary: pd.DataFrame, outdir: Path,
                        auprc_threshold: float, abl_threshold: float = 0.05):
    """
    Scatter: x = best_auprc, y = abl_contrast_self (if available)
    Quadrant of interest: x < auprc_threshold AND y > abl_threshold
    """
    has_ablation = "abl_contrast_self" in summary.columns and \
                   summary["abl_contrast_self"].notna().any()

    if not has_ablation:
        # just plot AUPRC distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(summary["best_auprc"], bins=40, color="#1D9E75",
                alpha=0.8, edgecolor="none")
        ax.axvline(auprc_threshold, color="#E8593C", lw=1.5, linestyle="--",
                   label=f"threshold = {auprc_threshold}")
        n_low = (summary["best_auprc"] < auprc_threshold).sum()
        ax.text(auprc_threshold + 0.01, ax.get_ylim()[1] * 0.9,
                f"{n_low} features\nbelow threshold",
                fontsize=9, color="#E8593C", va="top")
        ax.set_xlabel("Best AUPRC (max over all annotation types)", fontsize=10)
        ax.set_ylabel("Number of features", fontsize=10)
        ax.set_title("Feature AUPRC distribution — triage for novel concept candidates",
                     fontsize=10)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "triage_auprc_distribution.png", dpi=150)
        plt.close(fig)
        log.info(f"  → triage_auprc_distribution.png")
        return

    # scatter with ablation
    fig, ax = plt.subplots(figsize=(8, 7))

    valid = summary.dropna(subset=["abl_contrast_self"])
    x = valid["best_auprc"].values
    y = valid["abl_contrast_self"].values

    # colour by quadrant
    colors = []
    for xi, yi in zip(x, y):
        if xi < auprc_threshold and yi > abl_threshold:
            colors.append("#E8593C")   # novel concept candidate
        elif xi >= auprc_threshold and yi > abl_threshold:
            colors.append("#1D9E75")   # known + causally important
        elif xi < auprc_threshold and yi <= abl_threshold:
            colors.append("#D3D1C7")   # unknown, not causally important
        else:
            colors.append("#3266AD")   # known but low causal evidence

    ax.scatter(x, y, c=colors, alpha=0.7, s=25, edgecolors="none")

    # quadrant lines
    ax.axvline(auprc_threshold, color="#888780", lw=0.8, linestyle="--")
    ax.axhline(abl_threshold,   color="#888780", lw=0.8, linestyle="--")

    # quadrant labels
    ymax = max(y.max() * 1.05, abl_threshold * 3)
    ax.text(auprc_threshold / 2, ymax * 0.95,
            "★ Novel concept\ncandidates",
            ha="center", va="top", fontsize=9, color="#E8593C", fontweight="bold")
    ax.text((auprc_threshold + 1.0) / 2, ymax * 0.95,
            "Known annotation\n+ causally important",
            ha="center", va="top", fontsize=9, color="#1D9E75")
    ax.text(auprc_threshold / 2, abl_threshold * 0.5,
            "Low annotation\nlow causal signal",
            ha="center", va="top", fontsize=8, color="#888780")

    # annotate top novel candidates
    novel = valid[(valid["best_auprc"] < auprc_threshold) &
                  (valid["abl_contrast_self"] > abl_threshold)]
    novel_top = novel.nlargest(8, "abl_contrast_self")
    for _, row in novel_top.iterrows():
        ax.annotate(f"f{int(row['feature_id'])}",
                    (row["best_auprc"], row["abl_contrast_self"]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=7.5, color="#E8593C")

    ax.set_xlabel("Best AUPRC across all annotation types", fontsize=10)
    ax.set_ylabel("Ablation contrast (Δhi-act − Δlo-act)", fontsize=10)
    ax.set_title("Novel concept candidates:\nlow annotation alignment but high causal necessity",
                 fontsize=10)
    ax.set_xlim(-0.02, 1.02)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="#E8593C", label=f"Novel candidate (AUPRC<{auprc_threshold}, abl>0.05)"),
        Patch(color="#1D9E75", label="Known annotation + causally important"),
        Patch(color="#3266AD", label="Known annotation, low ablation signal"),
        Patch(color="#D3D1C7", label="Low annotation + low ablation"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False,
              loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "triage_novel_candidates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → triage_novel_candidates.png")

    # print top novel candidates
    if len(novel) > 0:
        log.info(f"\n  ★ TOP NOVEL CONCEPT CANDIDATES "
                 f"(AUPRC < {auprc_threshold}, ablation > {abl_threshold}):")
        print(novel.nlargest(15, "abl_contrast_self")[
            ["feature_id","best_auprc","best_annot","best_or",
             "abl_contrast_self","abl_frac_pos_self","n_proteins_abl"]
        ].to_string(index=False))
    else:
        log.info("  No novel concept candidates found with current thresholds.")
        log.info(f"  Consider lowering --auprc-threshold (current: {auprc_threshold})")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    global torch
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: triage from alignment scores (no GPU) ─────────────────────────
    log.info("=" * 54)
    log.info("  STEP 1: Triage from alignment scores")
    log.info("=" * 54)

    summary = triage_from_alignment(args.alignment_dir, args.auprc_threshold)

    # save full summary
    out_path = args.outdir / "feature_triage_summary.tsv"
    summary.to_csv(out_path, sep="\t", index=False)
    log.info(f"\n  Full triage summary: {out_path}")

    # identify candidates
    candidates = (summary[summary["best_auprc"] < args.auprc_threshold]
                  .sort_values("best_auprc")
                  .head(args.top_candidates))
    cand_path = args.outdir / "triage_candidates.tsv"
    candidates.to_csv(cand_path, sep="\t", index=False)
    log.info(f"  Top-{len(candidates)} candidates: {cand_path}")

    # quick sanity print
    log.info(f"\n  Top-10 most 'mysterious' features (lowest best_auprc):")
    print(summary.head(10)[["feature_id","best_auprc","best_annot",
                             "mean_auprc_all","n_annots_above_03"]].to_string(index=False))

    # ── Step 2: ablation (optional, needs GPU) ────────────────────────────────
    if args.run_ablation:
        try:
            import torch as _t
            torch = _t
        except ImportError:
            log.error("torch not available. Cannot run ablation.")
            args.run_ablation = False

    if args.run_ablation:
        if args.checkpoint is None:
            log.error("--checkpoint required for --run-ablation")
            sys.exit(1)

        log.info("\n" + "=" * 54)
        log.info("  STEP 2: Ablation for triage candidates")
        log.info("=" * 54)

        device = torch.device(
            "cuda" if (args.device == "auto" and torch.cuda.is_available())
            else (args.device if args.device != "auto" else "cpu")
        )
        log.info(f"Device: {device}")

        log.info("Loading SAE ...")
        W_enc, b_enc, W_dec, b_dec = load_sae(args.checkpoint, device)

        log.info("Loading sequences ...")
        prot_df = pd.read_csv(args.proteins, sep="\t", low_memory=False)
        prot_df = prot_df[prot_df["split"] == args.split]
        seq_by_acc = dict(zip(prot_df["accession"], prot_df["sequence"]))
        log.info(f"  {len(seq_by_acc):,} sequences loaded")

        log.info("Loading ESM-2 ...")
        esm_model, converter, backend = load_esm2(args.esm2_model, device)
        get_emb = lambda acc: _get_embedding_live(
            acc, seq_by_acc[acc], esm_model, converter, backend,
            device, args.esm2_layer, args.max_seq_len,
        )

        log.info(f"\nRunning ablation for {len(candidates)} candidates ...")
        results = run_ablation_triage(
            candidates=candidates,
            seq_by_acc=seq_by_acc,
            get_emb=get_emb,
            W_enc=W_enc, b_enc=b_enc,
            W_dec=W_dec, b_dec=b_dec,
            max_proteins=args.max_proteins,
            activation_threshold=args.activation_threshold,
        )

        results_path = args.outdir / "triage_with_ablation.tsv"
        results.to_csv(results_path, sep="\t", index=False)
        log.info(f"\n  Results with ablation: {results_path}")

        # merge back into full summary for plotting
        abl_cols = results[["feature_id","abl_contrast_self",
                             "abl_frac_pos_self","n_proteins_abl"]]
        summary = summary.merge(abl_cols, on="feature_id", how="left")

    # ── Step 3: visualise ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 54)
    log.info("  STEP 3: Visualisation")
    log.info("=" * 54)

    plot_triage_scatter(summary, args.outdir, args.auprc_threshold)

    log.info(f"\n✅ Done. Outputs in {args.outdir}")
    if not args.run_ablation:
        log.info("\n  💡 Next step: run with --run-ablation to measure causal")
        log.info("     necessity for the low-annotation candidates.")
        log.info(f"     Found {(summary['best_auprc'] < args.auprc_threshold).sum()} "
                 f"candidates with best_auprc < {args.auprc_threshold}")


if __name__ == "__main__":
    main()