"""
Generate publication-quality figures for the 53 robust
feature-annotation pairs from the enriched alignment.
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import OUT_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        False,
})

BG      = "#F8F7F4"
CARD_BG = "#FFFFFF"
GRID_C  = "#E8E6E0"
TEXT_S  = "#6B6964"

# annotation category → colour
CATEGORY_COLORS = {
    "Functional":    "#E8593C",   # red-orange
    "Structural":    "#3266AD",   # blue
    "Topological":   "#1D9E75",   # green
    "Compositional": "#BA7517",   # amber
    "PTM":           "#9B59B6",   # purple
    "Other":         "#7F8C8D",   # grey
}

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "ht", ["#F8F7F4", "#E8593C", "#7B1010"])


def categorise(annot_type: str) -> str:
    """Map annot_type to broad biological category."""
    t = annot_type.lower()
    if any(x in t for x in ["active site", "binding site", "dna binding",
                              "site:", "transit peptide", "propeptide"]):
        return "Functional"
    if any(x in t for x in ["helix", "beta strand", "turn", "coiled",
                              "transmembrane", "intramembrane"]):
        return "Structural"
    if any(x in t for x in ["topological domain", "region:"]):
        return "Topological"
    if any(x in t for x in ["compositional bias", "modified residue",
                              "glycosylation", "lipidation"]):
        return "Compositional / PTM"
    if any(x in t for x in ["domain:"]):
        return "Domain"
    return "Other"


def load_robust(alignment_dir: Path,
                min_proteins: int = 100,
                max_fisher_p: float = 0.01) -> pd.DataFrame:
    """Load and filter to robust pairs."""
    robust_path = alignment_dir / "per_annot_summary_robust.tsv"
    if robust_path.exists():
        df = pd.read_csv(robust_path, sep="\t")
        log.info(f"  Loaded pre-filtered robust TSV: {len(df)} pairs")
        return df

    summary_path = alignment_dir / "per_annot_summary_minprot30.tsv"
    if not summary_path.exists():
        summary_path = alignment_dir / "per_annot_summary.tsv"

    df = pd.read_csv(summary_path, sep="\t")
    df = df[(df["n_proteins"] >= min_proteins) &
            (df["fisher_p"] < max_fisher_p)].copy()
    log.info(f"  Filtered to {len(df)} robust pairs "
             f"(n≥{min_proteins}, p<{max_fisher_p})")
    return df


def shorten(s: str, maxlen: int = 48) -> str:
    return s[:maxlen] + "…" if len(s) > maxlen else s


# =============================================================================
#  FIG A — Bar chart top 20
# =============================================================================
def fig_bar_top20(df: pd.DataFrame, outdir: Path):
    top = df.sort_values("auprc_mean", ascending=False).head(20).copy()
    top["label"]    = top["annot_type"].apply(lambda s: shorten(s, 50))
    top["category"] = top["annot_type"].apply(categorise)
    top["or_label"] = top["odds_ratio"].apply(
        lambda x: "∞" if not np.isfinite(x) else f"{x:.1f}")

    cat_colors = {
        "Functional": "#E8593C", "Structural": "#3266AD",
        "Topological": "#1D9E75", "Compositional / PTM": "#9B59B6",
        "Domain": "#BA7517", "Other": "#7F8C8D",
    }

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(CARD_BG)

    colors = [cat_colors.get(c, "#7F8C8D") for c in top["category"]]
    bars = ax.barh(range(len(top)), top["auprc_mean"],
                   color=colors, alpha=0.85, linewidth=0, zorder=3)

    # annotate: n_proteins and OR
    for i, (_, row) in enumerate(top.iterrows()):
        x = row["auprc_mean"]
        ax.text(x + 0.008, i,
                f"n={int(row['n_proteins']):,}  OR={row['or_label']}",
                va="center", fontsize=7, color=TEXT_S)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("AUPRC (mean over proteins, residue-level)", fontsize=9)
    ax.set_title("Top 20 Feature–Annotation Pairs by Residue-Level AUPRC\n"
                 "(robust filter: n ≥ 100 proteins, Fisher p < 0.01)",
                 fontsize=10, fontweight="bold")

    # baseline annotation
    ax.axvline(0.5, color=GRID_C, linewidth=0.8, linestyle="--", zorder=1)
    ax.text(0.502, len(top)-0.5, "AUPRC = 0.5", fontsize=7,
            color=TEXT_S, va="top")

    # legend
    handles = [mpatches.Patch(color=c, label=l, alpha=0.85)
               for l, c in cat_colors.items()
               if l in top["category"].values]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=False)

    # grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.5)

    fig.tight_layout()
    path = outdir / "figA_bar_top20.png"
    fig.savefig(path, dpi=300, facecolor=BG)
    plt.close(fig)
    log.info(f"  → {path.name}")


# =============================================================================
#  FIG B — Scatter AUPRC × OR
# =============================================================================
def fig_scatter_auprc_or(df: pd.DataFrame, outdir: Path):
    # Use best feature per annotation type to avoid overplotting
    # (multiple features may achieve similar AUPRC for same annotation)
    best = (df.sort_values("auprc_mean", ascending=False)
              .groupby("annot_type").first().reset_index())
    # count how many features per annotation (polysemy indicator)
    n_features = df.groupby("annot_type").size().rename("n_features")
    best = best.join(n_features, on="annot_type")

    best["category"] = best["annot_type"].apply(categorise)
    best["or_plot"]  = best["odds_ratio"].apply(
        lambda x: min(x, 200) if np.isfinite(x) else 200)
    best["is_inf"]   = ~np.isfinite(best["odds_ratio"])

    cat_colors = {
        "Functional": "#E8593C", "Structural": "#3266AD",
        "Topological": "#1D9E75", "Compositional / PTM": "#9B59B6",
        "Domain": "#BA7517", "Other": "#7F8C8D",
    }

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(CARD_BG)

    for cat, grp in best.groupby("category"):
        # ensure sizes array aligns with finite/infinite mask
        grp = grp.copy()
        color = cat_colors.get(cat, "#7F8C8D")
        # size encodes number of features capturing this annotation
        fin  = grp[~grp["is_inf"]]
        inf_ = grp[ grp["is_inf"]]
        s_fin  = (fin["n_features"].clip(upper=8)  * 12).values
        s_inf  = (inf_["n_features"].clip(upper=8) * 12).values

        if not fin.empty:
            ax.scatter(fin["auprc_mean"], fin["or_plot"],
                       color=color, alpha=0.80, s=s_fin,
                       edgecolors="white", linewidths=0.5,
                       label=cat, zorder=4)
        if not inf_.empty:
            ax.scatter(inf_["auprc_mean"],
                       [205] * len(inf_),
                       color=color, marker="^", alpha=0.80,
                       s=s_inf,
                       edgecolors="white", linewidths=0.5,
                       zorder=4)

    # annotate interesting points
    annotate = [
        ("Binding site: GTP", "GTP binding\n(n=1,376)"),
        ("Region: G1", "GTPase G1\n(n=1,340)"),
        ("Domain: Protein kinase", "Protein kinase\ndomain"),
        ("Binding site: Mg(2+)", "Mg²⁺ binding\n(n=1,004)"),
        ("Domain: Carrier", "Carrier domain"),
        ("Transit peptide: Mitochondrion", "Mito. transit\npeptide"),
    ]
    for annot_type, label in annotate:
        row = df[df["annot_type"] == annot_type]
        if row.empty:
            continue
        row = row.iloc[0]
        xp = row["auprc_mean"]
        yp = min(row["odds_ratio"], 200) if np.isfinite(row["odds_ratio"]) else 205
        ax.annotate(label,
                    xy=(xp, yp), xytext=(xp + 0.04, yp + 10),
                    fontsize=7, color=TEXT_S,
                    arrowprops=dict(arrowstyle="-", color=GRID_C,
                                    lw=0.8),
                    va="bottom")

    # OR=inf line
    ax.axhline(200, color=GRID_C, linewidth=0.8, linestyle="--")
    ax.text(0.01, 205, "OR = ∞", fontsize=7.5, color=TEXT_S, va="bottom")

    ax.set_xlabel("AUPRC (residue-level, mean over proteins)", fontsize=9)
    ax.set_ylabel("Odds Ratio (capped at 200)", fontsize=9)
    ax.set_title("Residue-Level Specificity of SAE Features\n"
                 "Across Enriched UniProt Annotation Subtypes",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-5, 215)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.5)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.5)

    handles = [mpatches.Patch(color=c, label=l, alpha=0.85)
               for l, c in cat_colors.items()
               if l in best["category"].values]
    handles.append(mpatches.Patch(color="grey", label="▲ OR = ∞",
                                  alpha=0.5))
    ax.legend(handles=handles, fontsize=7.5, frameon=False,
              loc="upper left")

    fig.tight_layout()
    path = outdir / "figB_scatter_auprc_or.png"
    fig.savefig(path, dpi=300, facecolor=BG)
    plt.close(fig)
    log.info(f"  → {path.name}")


# =============================================================================
#  FIG C — Heatmap top pairs
# =============================================================================
def fig_heatmap(df: pd.DataFrame, outdir: Path, top_n: int = 30):
    """Heatmap: annotation subtypes (rows) × feature IDs (cols), AUPRC colour."""
    # take top_n pairs by AUPRC, deduplicate to unique (annot, feature)
    top = df.sort_values("auprc_mean", ascending=False).head(top_n).copy()

    annot_types = top["annot_type"].unique().tolist()
    feature_ids = top["feature_id"].unique().tolist()

    # build matrix
    mat = pd.DataFrame(np.nan, index=annot_types, columns=feature_ids)
    for _, row in top.iterrows():
        mat.loc[row["annot_type"], row["feature_id"]] = row["auprc_mean"]

    # shorten labels
    row_labels = [shorten(a, 52) for a in mat.index]
    col_labels  = [f"f{int(f)}" for f in mat.columns]

    fig_h = max(6, len(annot_types) * 0.42)
    fig_w = max(8, len(feature_ids) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(CARD_BG)

    mat_vals = mat.values.astype(float)
    im = ax.imshow(mat_vals, cmap=HEAT_CMAP, aspect="auto",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # annotate cells
    for i in range(mat_vals.shape[0]):
        for j in range(mat_vals.shape[1]):
            v = mat_vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if v > 0.6 else "#2C3E50")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("AUPRC (residue-level)", fontsize=8)

    ax.set_title("SAE Feature × Annotation Subtype Alignment (AUPRC)\n"
                 f"Top {top_n} robust pairs (n ≥ 100 proteins, Fisher p < 0.01)",
                 fontsize=10, fontweight="bold")

    # category colour strip on left
    cat_colors = {
        "Functional": "#E8593C", "Structural": "#3266AD",
        "Topological": "#1D9E75", "Compositional / PTM": "#9B59B6",
        "Domain": "#BA7517", "Other": "#7F8C8D",
    }
    for i, annot in enumerate(mat.index):
        cat   = categorise(annot)
        color = cat_colors.get(cat, "#7F8C8D")
        ax.add_patch(plt.Rectangle(
            (-0.55, i - 0.5), 0.45, 1.0,
            color=color, alpha=0.85, clip_on=False, zorder=5))

    handles = [mpatches.Patch(color=c, label=l, alpha=0.85)
               for l, c in cat_colors.items()
               if l in [categorise(a) for a in mat.index]]
    ax.legend(handles=handles, fontsize=7, frameon=False,
              bbox_to_anchor=(1.18, 1), loc="upper right")

    fig.tight_layout()
    path = outdir / "figC_heatmap_robust.png"
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → {path.name}")


# =============================================================================
#  FIG D — GTPase spotlight
# =============================================================================
def fig_gtpase_spotlight(df: pd.DataFrame, outdir: Path):
    """Highlight GTPase motifs: G1-G4 regions + GTP binding site."""
    gtpase_annots = [
        "Region: G1", "Region: G2", "Region: G3", "Region: G4",
        "Binding site: GTP",
    ]
    sub = df[df["annot_type"].isin(gtpase_annots)].copy()

    if sub.empty:
        log.warning("  No GTPase motif pairs found — skipping Fig D")
        return

    # for each annotation, take best feature by AUPRC
    best = (sub.sort_values("auprc_mean", ascending=False)
               .groupby("annot_type").first().reset_index())

    # colour by annotation
    colors = {
        "Region: G1":        "#E8593C",
        "Region: G2":        "#C0392B",
        "Region: G3":        "#E67E22",
        "Region: G4":        "#F39C12",
        "Binding site: GTP": "#1D9E75",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG,
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # ── Left: bar chart AUPRC per GTPase motif ────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(CARD_BG)

    order = [a for a in gtpase_annots if a in best["annot_type"].values]
    vals  = [best.loc[best["annot_type"]==a, "auprc_mean"].iloc[0] for a in order]
    cols  = [colors[a] for a in order]
    fids  = [f"f{int(best.loc[best['annot_type']==a,'feature_id'].iloc[0])}"
             for a in order]
    nprots= [int(best.loc[best["annot_type"]==a,"n_proteins"].iloc[0])
              for a in order]

    bars = ax.barh(range(len(order)), vals, color=cols,
                   alpha=0.85, linewidth=0, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("AUPRC (residue-level)", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.5)

    for i, (v, fid, n) in enumerate(zip(vals, fids, nprots)):
        ax.text(v + 0.015, i, f"{fid}  n={n:,}",
                va="center", fontsize=7.5, color=TEXT_S)

    ax.set_title("GTPase Motifs — Best Feature per Region\n"
                 "(residue-level AUPRC)",
                 fontsize=9.5, fontweight="bold")

    # ── Right: strip/dot plot of AUPRC distribution per GTPase motif ──────────
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)

    for i, annot in enumerate(order):
        grp = df[df["annot_type"] == annot]
        if grp.empty:
            continue
        # jitter vertically
        ys = np.random.uniform(i - 0.25, i + 0.25, len(grp))
        ax2.scatter(grp["auprc_mean"], ys,
                    color=colors[annot], alpha=0.75, s=50,
                    edgecolors="white", linewidths=0.5,
                    zorder=4)
        # mean marker
        ax2.scatter([grp["auprc_mean"].mean()], [i],
                    color=colors[annot], s=120, marker="D",
                    edgecolors="white", linewidths=1.0,
                    zorder=5)
        # annotate n features
        ax2.text(1.01, i, f"n={len(grp)} features",
                 va="center", fontsize=7, color=TEXT_S)

    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order, fontsize=8.5)
    ax2.set_xlabel("AUPRC (residue-level)", fontsize=9)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(-0.7, len(order) - 0.3)
    ax2.axvline(0.5, color=GRID_C, linewidth=0.7, linestyle="--")
    ax2.text(0.51, -0.55, "0.5", fontsize=7, color=TEXT_S)
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(True, color=GRID_C, linewidth=0.5)
    ax2.set_title("AUPRC Distribution Across SAE Features\n"
                  "(◆ = mean, dots = individual features)",
                  fontsize=9.5, fontweight="bold")

    # suptitle
    fig.suptitle(
        "SAE Features Decompose GTPase Domain Functional Subregions\n"
        "G1 (P-loop), G2, G3, G4, and GTP binding site captured by distinct features",
        fontsize=10, fontweight="bold", y=1.02)

    fig.tight_layout()
    path = outdir / "figD_gtpase_spotlight.png"
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → {path.name}")


# =============================================================================
#  CLI + Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--alignment-dir", type=Path,
                   default=OUT_DIR / "feature_alignment_enriched2")
    p.add_argument("--outdir", type=Path,
                   default=OUT_DIR / "paper_figures_enriched")
    p.add_argument("--min-proteins", type=int, default=100)
    p.add_argument("--max-fisher-p", type=float, default=0.01)
    p.add_argument("--top-heatmap",  type=int, default=30,
                   help="Number of pairs to show in heatmap.")
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    log.info("Loading robust alignment pairs ...")
    df = load_robust(args.alignment_dir, args.min_proteins, args.max_fisher_p)

    if df.empty:
        log.error("No robust pairs found. Check --alignment-dir and filters.")
        return

    log.info(f"  {len(df)} pairs, {df['annot_type'].nunique()} annotation subtypes")
    log.info(f"  AUPRC range: {df['auprc_mean'].min():.3f} – "
             f"{df['auprc_mean'].max():.3f}")

    log.info("\nGenerating figures ...")

    log.info("  Fig A — Bar chart top 20 ...")
    fig_bar_top20(df, args.outdir)

    log.info("  Fig B — Scatter AUPRC × OR ...")
    fig_scatter_auprc_or(df, args.outdir)

    log.info("  Fig C — Heatmap ...")
    fig_heatmap(df, args.outdir, top_n=args.top_heatmap)

    log.info("  Fig D — GTPase spotlight ...")
    fig_gtpase_spotlight(df, args.outdir)

    log.info(f"\n✅ Done. Figures in {args.outdir}")
    for f in sorted(args.outdir.glob("fig*.png")):
        log.info(f"    {f.name}")


if __name__ == "__main__":
    main()