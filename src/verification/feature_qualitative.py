"""
feature_qualitative.py — PIPLV2 / sae
======================================
Qualitative analysis of top SAE features from alignment results.

For each candidate (feature_id, annot_type) pair, produces:

1. Positional profile  — mean activation at normalised relative position
   (0=N-term, 1=C-term), split by annotated vs unannotated residues.
   Answers: "does f3087 always fire at the N-terminus?"

2. Activation distribution — histogram of activation magnitudes for
   annotated vs unannotated residues across all proteins.
   Answers: "how much stronger is the signal inside vs outside the annotation?"

3. Top activating proteins — ranked list with annotation overlap,
   protein length, and description.
   Answers: "what kind of proteins does this feature specialise in?"

4. Per-protein activation heatmap strip — a compact grid of activation
   profiles for the top-20 proteins, with annotation regions highlighted.
   Answers: "is the positional pattern consistent across proteins?"

5. Co-annotation analysis — for features that lead multiple annotation
   types, shows overlap between those annotations to characterise polysemy.
   Answers: "is f3087 polysemous or does it capture a general concept?"

Outputs (per feature):
  outputs/feature_qualitative/
    f{id}_{annot_type}/
      positional_profile.png
      activation_distribution.png
      top_proteins.tsv
      protein_heatmap.png
      summary.json

Usage
-----
  python src/sae/feature_qualitative.py \\
      --alignment    outputs/feature_alignment/per_annot_summary.tsv \\
      --esm2-model   esm2_t33_650M_UR50D \\
      --esm2-layer   33 \\
      --checkpoint   outputs/sae_runs/latent8192_l1_3e-05_lr_3e-04/checkpoints/best.pt \\
      --annotations  data/annotations_dedup_with_split.tsv \\
      --split        test \\
      --top-pairs    10        # analyse top-N (feature, annot_type) pairs by AUPRC
      --activation-threshold 0.1
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from constants import OUT_DIR, ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH

torch = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Qualitative analysis of top SAE features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--alignment",   type=Path,
                   default=OUT_DIR / "feature_alignment" / "per_annot_summary.tsv",
                   help="per_annot_summary.tsv from feature_alignment.py")
    p.add_argument("--all-scores",  type=Path,
                   default=OUT_DIR / "feature_alignment" / "alignment_scores.parquet",
                   help="Full alignment_scores.parquet (for co-annotation analysis).")
    p.add_argument("--proteins",    type=Path,
                   default=OUT_DIR.parent / "data" / "proteins_with_split.tsv")
    p.add_argument("--esm2-model", default="esm2_t33_650M_UR50D",
                   help="ESM-2 model name for on-the-fly embedding.")
    p.add_argument("--esm2-layer", type=int, default=33,
                   help="ESM-2 transformer layer to extract embeddings from.")
    p.add_argument("--max-seq-len", type=int, default=1022,
                   help="Truncate sequences longer than this.")
    p.add_argument("--checkpoint",  type=Path, required=True)
    p.add_argument("--annotations", type=Path,
                   default=ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH)
    p.add_argument("--split",       default="test")
    p.add_argument("--top-pairs",   type=int, default=10,
                   help="Analyse top-N (feature, annot_type) pairs by AUPRC.")
    p.add_argument("--min-auprc",   type=float, default=0.3,
                   help="Skip pairs with AUPRC below this.")
    p.add_argument("--activation-threshold", type=float, default=0.1)
    p.add_argument("--top-proteins", type=int, default=20,
                   help="Proteins shown in per-protein heatmap.")
    p.add_argument("--outdir",      type=Path,
                   default=OUT_DIR / "feature_qualitative")
    p.add_argument("--device",      default="auto")
    return p.parse_args()


# =============================================================================
#  SAE ENCODER (same as feature_alignment.py — standalone, no class needed)
# =============================================================================

def load_encoder(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model_state_dict"]
    W = sd["encoder.weight"].float().to(device)
    b = sd["encoder.bias"].float().to(device)
    return W, b


def encode(emb, W, b):
    """emb: [L, D] tensor → z: [L, K] numpy float32. W and b already on device."""
    with torch.no_grad():
        z = torch.relu(emb.float().to(W.device) @ W.T + b)
    return z.cpu().numpy().astype(np.float32)


# =============================================================================
#  DATA LOADING HELPERS
# =============================================================================

from esm_utils import load_esm2
from esm_utils import get_embedding as _get_embedding_live


def build_mask(ann_df_acc, L):
    """Build binary annotation mask [L] from a per-accession annotation df."""
    mask = np.zeros(L, dtype=np.float32)
    for _, row in ann_df_acc.iterrows():
        s = max(0, int(row["start"]) - 1)
        e = min(L, int(row["end"]))
        mask[s:e] = 1.0
    return mask


# =============================================================================
#  PER-PAIR ANALYSIS
# =============================================================================

def analyse_pair(
    feature_id:   int,
    annot_type:   str,
    proteins_df:  pd.DataFrame,   # proteins with this annotation in split
    ann_by_acc:   dict,
    get_emb,
    W, b,
    threshold:    float,
    top_n:        int,
    outdir:       Path,
) -> dict:
    """
    Full qualitative analysis for one (feature_id, annot_type) pair.
    Returns summary dict.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # ── collect per-protein data ──────────────────────────────────────────────
    records = []   # list of dicts per protein
    pos_act_in  = []   # activations at annotated residues (relative position)
    pos_act_out = []   # activations outside annotation
    act_in_all  = []   # raw activation values inside annotation
    act_out_all = []   # raw activation values outside annotation

    accs = [a for a in proteins_df["accession"].unique()
            if a in ann_by_acc]

    log.info(f"  Analysing f{feature_id} × {annot_type}: {len(accs)} proteins ...")

    for acc in accs:
        emb  = get_emb(acc)
        L    = emb.shape[0]
        z    = encode(emb, W, b)         # [L, K]
        acts = z[:, feature_id]          # [L] activation for this feature

        ann_sub = ann_by_acc[acc][
            ann_by_acc[acc]["feature_type"] == annot_type
        ]
        if ann_sub.empty:
            continue

        mask = build_mask(ann_sub, L)    # [L] binary

        # per-protein summary
        acts_in  = acts[mask == 1]
        acts_out = acts[mask == 0]
        mean_in  = float(acts_in.mean())  if len(acts_in)  > 0 else 0.0
        mean_out = float(acts_out.mean()) if len(acts_out) > 0 else 0.0
        tp = float(((acts > threshold) * mask).sum())
        precision = tp / max((acts > threshold).sum(), 1)
        recall    = tp / max(mask.sum(), 1)

        records.append({
            "accession":    acc,
            "length":       L,
            "mean_act_in":  mean_in,
            "mean_act_out": mean_out,
            "contrast":     mean_in - mean_out,
            "precision":    float(precision),
            "recall":       float(recall),
            "annot_frac":   float(mask.mean()),
            "n_annot_res":  int(mask.sum()),
        })

        # collect positional data (normalised position 0→1)
        rel_pos = np.arange(L) / max(L - 1, 1)
        for i in range(L):
            if mask[i] == 1:
                pos_act_in.append((rel_pos[i], acts[i]))
            else:
                pos_act_out.append((rel_pos[i], acts[i]))

        # collect raw activations
        act_in_all.extend(acts_in.tolist())
        act_out_all.extend(acts_out.tolist())

    if not records:
        log.warning(f"  No data for f{feature_id} × {annot_type}")
        return {}

    rec_df = pd.DataFrame(records).sort_values("contrast", ascending=False)
    rec_df.to_csv(outdir / "top_proteins.tsv", sep="\t", index=False)

    # ── figure 1: positional profile ─────────────────────────────────────────
    _plot_positional_profile(pos_act_in, pos_act_out, feature_id, annot_type,
                             outdir / "positional_profile.png")

    # ── figure 2: activation distribution ────────────────────────────────────
    _plot_activation_distribution(act_in_all, act_out_all, threshold,
                                  feature_id, annot_type,
                                  outdir / "activation_distribution.png")

    # ── figure 3: per-protein heatmap strip ──────────────────────────────────
    top_accs = rec_df.head(top_n)["accession"].tolist()
    _plot_protein_heatmap(top_accs, feature_id, annot_type,
                          ann_by_acc, get_emb, W, b, threshold,
                          outdir / "protein_heatmap.png")

    # ── summary ───────────────────────────────────────────────────────────────
    summary = {
        "feature_id":       feature_id,
        "annot_type":       annot_type,
        "n_proteins":       len(records),
        "mean_contrast":    float(rec_df["contrast"].mean()),
        "median_contrast":  float(rec_df["contrast"].median()),
        "mean_precision":   float(rec_df["precision"].mean()),
        "mean_recall":      float(rec_df["recall"].mean()),
        "top5_accessions":  rec_df.head(5)["accession"].tolist(),
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =============================================================================
#  PLOT HELPERS
# =============================================================================

STYLE = {
    "in_color":  "#1D9E75",
    "out_color": "#888780",
    "thresh_color": "#E8593C",
    "fig_facecolor": "white",
    "fontsize": 10,
}


def _binned_mean(xs, ys, n_bins=50):
    """Bin (x, y) pairs into n_bins and return (bin_centres, means, stds)."""
    xs, ys = np.array(xs), np.array(ys)
    edges  = np.linspace(0, 1, n_bins + 1)
    centres, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (xs >= lo) & (xs < hi)
        if mask.sum() > 0:
            centres.append((lo + hi) / 2)
            means.append(ys[mask].mean())
            stds.append(ys[mask].std())
    return np.array(centres), np.array(means), np.array(stds)


def _plot_positional_profile(pos_in, pos_out, feat_id, annot_type, path):
    fig, ax = plt.subplots(figsize=(8, 3.5))

    if pos_out:
        cx, my, sy = _binned_mean(*zip(*pos_out))
        ax.plot(cx, my, color=STYLE["out_color"], lw=1.5, label="outside annotation")
        ax.fill_between(cx, my - sy, my + sy, alpha=0.15, color=STYLE["out_color"])

    if pos_in:
        cx, my, sy = _binned_mean(*zip(*pos_in))
        ax.plot(cx, my, color=STYLE["in_color"], lw=2, label="inside annotation")
        ax.fill_between(cx, my - sy, my + sy, alpha=0.2, color=STYLE["in_color"])

    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.set_xlabel("Relative position (0=N-term, 1=C-term)", fontsize=STYLE["fontsize"])
    ax.set_ylabel("Mean activation", fontsize=STYLE["fontsize"])
    ax.set_title(f"f{feat_id} × {annot_type} — positional profile",
                 fontsize=STYLE["fontsize"] + 1)
    ax.legend(fontsize=STYLE["fontsize"] - 1)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_activation_distribution(act_in, act_out, threshold, feat_id, annot_type, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    bins = np.linspace(0, max(
        max(act_in) if act_in else 1,
        max(act_out) if act_out else 1,
    ) * 1.05, 60)

    if act_out:
        ax.hist(act_out, bins=bins, density=True, alpha=0.5,
                color=STYLE["out_color"], label="outside")
    if act_in:
        ax.hist(act_in,  bins=bins, density=True, alpha=0.6,
                color=STYLE["in_color"],  label="inside")

    ax.axvline(threshold, color=STYLE["thresh_color"], lw=1.5,
               linestyle="--", label=f"threshold={threshold}")
    ax.set_xlabel("Activation value", fontsize=STYLE["fontsize"])
    ax.set_ylabel("Density", fontsize=STYLE["fontsize"])
    ax.set_title(f"f{feat_id} × {annot_type} — activation distribution",
                 fontsize=STYLE["fontsize"] + 1)
    ax.legend(fontsize=STYLE["fontsize"] - 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_protein_heatmap(accs, feat_id, annot_type, ann_by_acc,
                          get_emb, W, b, threshold, path):
    """
    Compact strip: each row = one protein, columns = normalised position bins.
    Cells coloured by activation; annotation regions outlined in green.
    """
    N_BINS = 100
    n      = len(accs)
    if n == 0:
        return

    heatmap = np.zeros((n, N_BINS), dtype=np.float32)
    ann_map = np.zeros((n, N_BINS), dtype=np.float32)
    lengths = []

    for row_i, acc in enumerate(accs):
        emb  = get_emb(acc)
        L    = emb.shape[0]
        z    = encode(emb, W, b)[:, feat_id]   # [L]
        lengths.append(L)

        # bin activations into N_BINS
        for bin_j in range(N_BINS):
            lo = int(L * bin_j / N_BINS)
            hi = int(L * (bin_j + 1) / N_BINS)
            hi = max(hi, lo + 1)
            heatmap[row_i, bin_j] = z[lo:hi].mean()

        # annotation mask binned
        if acc in ann_by_acc:
            ann_sub = ann_by_acc[acc][
                ann_by_acc[acc]["feature_type"] == annot_type
            ]
            if not ann_sub.empty:
                mask = build_mask(ann_sub, L)
                for bin_j in range(N_BINS):
                    lo = int(L * bin_j / N_BINS)
                    hi = int(L * (bin_j + 1) / N_BINS)
                    hi = max(hi, lo + 1)
                    ann_map[row_i, bin_j] = mask[lo:hi].mean()

    height = max(3.5, n * 0.3)
    fig, ax = plt.subplots(figsize=(12, height))

    vmax = np.percentile(heatmap, 95)
    im   = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd",
                     vmin=0, vmax=max(vmax, threshold),
                     interpolation="nearest")

    # overlay annotation regions as green contour
    for row_i in range(n):
        in_ann = False
        for bin_j in range(N_BINS):
            if ann_map[row_i, bin_j] > 0.3 and not in_ann:
                x_start = bin_j
                in_ann  = True
            elif ann_map[row_i, bin_j] <= 0.3 and in_ann:
                ax.add_patch(plt.Rectangle(
                    (x_start - 0.5, row_i - 0.5),
                    bin_j - x_start, 1,
                    fill=False, edgecolor="#1D9E75", lw=1.5,
                ))
                in_ann = False
        if in_ann:
            ax.add_patch(plt.Rectangle(
                (x_start - 0.5, row_i - 0.5),
                N_BINS - x_start, 1,
                fill=False, edgecolor="#1D9E75", lw=1.5,
            ))

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{a} (L={l})" for a, l in zip(accs, lengths)],
        fontsize=7,
    )
    ax.set_xticks([0, 24, 49, 74, 99])
    ax.set_xticklabels(["N-term", "25%", "50%", "75%", "C-term"],
                       fontsize=STYLE["fontsize"] - 1)
    ax.set_title(
        f"f{feat_id} × {annot_type} — per-protein activation\n"
        f"(green outline = annotated region)",
        fontsize=STYLE["fontsize"] + 1,
    )
    plt.colorbar(im, ax=ax, label="Activation", fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  CO-ANNOTATION ANALYSIS
# =============================================================================

def analyse_coannotation(all_scores_df, feature_id, outdir):
    """
    For a feature that appears in multiple annotation types, show all its
    annotation scores side by side to characterise polysemy.
    """
    feat_df = (all_scores_df[all_scores_df["feature_id"] == feature_id]
               .sort_values("auprc_mean", ascending=False))
    if feat_df.empty:
        return

    # bar chart of AUPRC across all annotation types for this feature
    fig, ax = plt.subplots(figsize=(max(8, len(feat_df) * 0.5), 4))
    colors = ["#1D9E75" if a >= 0.4 else "#BA7517" if a >= 0.2 else "#D3D1C7"
              for a in feat_df["auprc_mean"]]
    ax.barh(feat_df["annot_type"], feat_df["auprc_mean"],
            color=colors, edgecolor="none")
    ax.axvline(0.3, color="#E8593C", lw=1, linestyle="--", label="AUPRC=0.3")
    ax.set_xlabel("AUPRC (mean over proteins)", fontsize=STYLE["fontsize"])
    ax.set_title(f"f{feature_id} — alignment with all annotation types",
                 fontsize=STYLE["fontsize"] + 1)
    ax.legend(fontsize=STYLE["fontsize"] - 1)
    fig.tight_layout()
    fig.savefig(outdir / f"f{feature_id}_coannotation.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    global torch

    args = parse_args()

    try:
        import torch as _t
        torch = _t
    except ImportError:
        sys.exit("\n[ERROR] pip install torch\n")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    log.info(f"Device: {device}")

    # ── load encoder ──────────────────────────────────────────────────────────
    log.info("Loading SAE encoder ...")
    W, b = load_encoder(args.checkpoint, device)
    K = W.shape[0]
    log.info(f"  K={K} latents")

    # ── load alignment summary ─────────────────────────────────────────────────
    log.info("Loading alignment results ...")
    summary_df = pd.read_csv(args.alignment, sep="\t")
    summary_df = summary_df[summary_df["auprc_mean"] >= args.min_auprc]
    summary_df = summary_df.sort_values("auprc_mean", ascending=False)

    pairs = list(zip(summary_df["feature_id"], summary_df["annot_type"]))
    pairs = pairs[: args.top_pairs]
    log.info(f"  Analysing {len(pairs)} pairs")

    # ── load full scores for co-annotation ────────────────────────────────────
    all_scores_df = None
    if args.all_scores.exists():
        all_scores_df = pd.read_parquet(args.all_scores)

    # ── load annotations ──────────────────────────────────────────────────────
    log.info("Loading annotations ...")
    ann_df = pd.read_csv(args.annotations, sep="\t", low_memory=False)
    ann_df.columns = ann_df.columns.str.strip().str.lower()
    ann_df = ann_df[ann_df["split"] == args.split].copy()
    ann_df["start"] = ann_df["start"].astype(int)
    ann_df["end"]   = ann_df["end"].astype(int)
    ann_by_acc = {acc: grp for acc, grp in ann_df.groupby("accession")}

    # ── load sequences ────────────────────────────────────────────────────────
    log.info("Loading sequences ...")
    prot_df = pd.read_csv(args.proteins, sep="\t", low_memory=False)
    prot_df = prot_df[prot_df["split"] == args.split]
    seq_by_acc = dict(zip(prot_df["accession"], prot_df["sequence"]))
    log.info(f"  {len(seq_by_acc):,} sequences loaded")

    # ── load ESM-2 for on-the-fly embeddings ─────────────────────────────────
    log.info("Loading ESM-2 ...")
    esm_model, converter, backend = load_esm2(args.esm2_model, device)
    get_emb = lambda acc: _get_embedding_live(
        acc, seq_by_acc[acc], esm_model, converter, backend,
        device, args.esm2_layer, args.max_seq_len,
    )

    # ── analyse each pair ─────────────────────────────────────────────────────
    all_summaries = []
    seen_features = set()

    for feature_id, annot_type in pairs:
        feature_id = int(feature_id)
        safe_name  = annot_type.lower().replace(" ", "_")
        pair_dir   = args.outdir / f"f{feature_id}_{safe_name}"

        log.info(f"\n{'─'*50}")
        log.info(f"  f{feature_id} × {annot_type}")

        # proteins that have this annotation in the split
        ann_sub = ann_df[ann_df["feature_type"] == annot_type]
        proteins_df = ann_sub[["accession"]].drop_duplicates()

        s = analyse_pair(
            feature_id=feature_id,
            annot_type=annot_type,
            proteins_df=proteins_df,
            ann_by_acc=ann_by_acc,
            get_emb=get_emb,
            W=W, b=b,
            threshold=args.activation_threshold,
            top_n=args.top_proteins,
            outdir=pair_dir,
        )
        if s:
            all_summaries.append(s)

        # co-annotation plot for each unique feature (once per feature)
        if feature_id not in seen_features and all_scores_df is not None:
            co_dir = args.outdir / f"f{feature_id}_coannotation"
            co_dir.mkdir(exist_ok=True)
            analyse_coannotation(all_scores_df, feature_id, co_dir)
            seen_features.add(feature_id)

    # ── global summary table ──────────────────────────────────────────────────
    if all_summaries:
        gs = pd.DataFrame(all_summaries).sort_values("mean_contrast", ascending=False)
        gs_path = args.outdir / "global_summary.tsv"
        gs.to_csv(gs_path, sep="\t", index=False)
        log.info(f"\nGlobal summary: {gs_path}")
        print("\n  Top pairs by activation contrast (inside − outside annotation):")
        print(gs[["feature_id","annot_type","mean_contrast",
                   "mean_precision","mean_recall","n_proteins"]].to_string(index=False))

    log.info(f"\n✅ Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()