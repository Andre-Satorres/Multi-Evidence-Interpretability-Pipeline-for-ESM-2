import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from constants import SPLIT_ANALYSIS_DIR, PROTEINS_WITH_SPLIT_TSV_PATH, ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH

# ── Paths ─────────────────────────────────────────────────────────────────────

# ── Design ────────────────────────────────────────────────────────────────────
SPLIT_ORDER  = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#3266AD", "val": "#1D9E75", "test": "#E8593C"}
SPLIT_PAL    = [SPLIT_COLORS[s] for s in SPLIT_ORDER]

BG      = "#F8F7F4"
CARD    = "#FFFFFF"
GRID_C  = "#E8E6E0"
TEXT_P  = "#1A1917"
TEXT_S  = "#6B6964"

TOP_N_TYPES = 15     # max feature types shown in crowded plots
SAMPLE_N    = 200_000  # rows sampled for heavy scatter/KDE

sns.set_theme(style="whitegrid", font_scale=1.0)

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     CARD,
    "axes.edgecolor":     GRID_C,
    "axes.labelcolor":    TEXT_S,
    "axes.titlecolor":    TEXT_P,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.color":         GRID_C,
    "grid.linewidth":     0.6,
    "xtick.color":        TEXT_S,
    "ytick.color":        TEXT_S,
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig, name: str, dpi: int = 150):
    path = SPLIT_ANALYSIS_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    → {path.name}")


def _style(ax):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(GRID_C); sp.set_linewidth(0.5)
    ax.set_axisbelow(True)


def _kfmt(x, _):
    if x >= 1e6: return f"{x/1e6:.1f}M"
    if x >= 1e3: return f"{x/1e3:.0f}k"
    return str(int(x))


def _section(title: str):
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD
# ═════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load proteins_with_split.tsv and annotations_dedup_with_split.tsv."""
    _section("Loading data")

    def _read(path, label):
        print(f"  reading {label} ...", flush=True)
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        print(f"    {len(df):,} rows × {df.shape[1]} cols")
        return df

    prot = _read(PROTEINS_WITH_SPLIT_TSV_PATH,    "proteins_with_split.tsv")
    ann  = _read(ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH, "annotations_dedup_with_split.tsv")

    # normalise split column
    for df in (prot, ann):
        if "split" in df.columns:
            df["split"] = df["split"].str.strip().str.lower()

    # annotation length
    ann["ann_len"] = (ann["end"] - ann["start"]).abs() + 1

    # annotation count per protein
    acc_col = _acc_col(prot)
    ann_counts = ann.groupby(acc_col).size().rename("n_annotations")
    prot = prot.merge(ann_counts, left_on=acc_col, right_index=True, how="left")
    prot["n_annotations"] = prot["n_annotations"].fillna(0).astype(int)

    # annotation density per 100 aa
    if "length" in prot.columns:
        prot["ann_density"] = prot["n_annotations"] / prot["length"].clip(lower=1) * 100

    return prot, ann


def _acc_col(df):
    return next((c for c in df.columns
                 if c in ("accession", "entry", "protein_id")), df.columns[0])


# ═════════════════════════════════════════════════════════════════════════════
#  A — SPLIT-LEVEL SUMMARIES
# ═════════════════════════════════════════════════════════════════════════════

def summarize_splits(prot: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    _section("A — Split-level summaries")
    acc = _acc_col(prot)

    rows = []
    total_p = len(prot); total_a = len(ann)
    total_c = prot["cluster_id"].nunique() if "cluster_id" in prot.columns else None

    for sp in SPLIT_ORDER:
        p = prot[prot["split"] == sp]
        a = ann[ann["split"] == sp]
        n_clust = p["cluster_id"].nunique() if "cluster_id" in p.columns else None
        rows.append({
            "split":           sp,
            "n_proteins":      len(p),
            "pct_proteins":    len(p) / total_p * 100,
            "n_clusters":      n_clust,
            "pct_clusters":    n_clust / total_c * 100 if total_c else None,
            "n_annotations":   len(a),
            "pct_annotations": len(a) / total_a * 100,
        })

    summary = pd.DataFrame(rows).set_index("split")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))
    summary.to_csv(SPLIT_ANALYSIS_DIR / "A_split_summary.csv")

    # ── bar chart ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=BG)
    fig.suptitle("A — Split-level Summaries", fontweight="bold", color=TEXT_P)

    metrics = [
        ("n_proteins",    "Proteins"),
        ("n_clusters",    "Clusters"),
        ("n_annotations", "Annotations"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        _style(ax)
        vals = [summary.loc[s, col] for s in SPLIT_ORDER if s in summary.index]
        bars = ax.bar(SPLIT_ORDER, vals, color=SPLIT_PAL, width=0.55,
                      zorder=3, linewidth=0)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + max(vals) * 0.01, f"{_kfmt(h, None)}",
                    ha="center", va="bottom", fontsize=9, color=TEXT_S)
        ax.set_title(label); ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
        ax.yaxis.grid(True); ax.xaxis.grid(False)

    plt.tight_layout()
    _save(fig, "A_split_summary")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
#  B — PROTEIN LENGTH DISTRIBUTIONS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_lengths(prot: pd.DataFrame) -> pd.DataFrame:
    _section("B — Protein length distributions")
    if "length" not in prot.columns:
        print("  ⚠ 'length' column not found — skipping"); return pd.DataFrame()

    # summary table
    stats = (prot.groupby("split")["length"]
               .agg(["count", "mean", "median", "std", "min", "max"])
               .loc[SPLIT_ORDER])
    stats.columns = ["N", "mean", "median", "std", "min", "max"]
    print(stats.round(1).to_string())
    stats.to_csv(SPLIT_ANALYSIS_DIR / "B_length_stats.csv")

    sub = prot[prot["split"].isin(SPLIT_ORDER)].copy()

    # ── Fig B1: KDE ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    _style(ax)
    for sp in SPLIT_ORDER:
        vals = sub.loc[sub["split"] == sp, "length"].clip(upper=2000)
        sns.kdeplot(vals, ax=ax, label=sp, color=SPLIT_COLORS[sp],
                    linewidth=1.8, fill=True, alpha=0.18)
    ax.set_title("B1 — Protein Length KDE (clipped at 2000 aa)")
    ax.set_xlabel("Length (aa)"); ax.set_ylabel("Density")
    ax.legend(frameon=False)
    plt.tight_layout(); _save(fig, "B1_length_kde")

    # ── Fig B2: violin ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    _style(ax)
    sample = sub.sample(min(len(sub), SAMPLE_N), random_state=42)
    sns.violinplot(data=sample, x="split", y="length", order=SPLIT_ORDER,
                   palette=SPLIT_COLORS, inner="box", ax=ax,
                   cut=0, linewidth=0.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))
    ax.set_title("B2 — Protein Length Distribution by Split (log scale)")
    ax.set_xlabel("Split"); ax.set_ylabel("Length (aa, log)")
    plt.tight_layout(); _save(fig, "B2_length_violin")

    return stats


# ═════════════════════════════════════════════════════════════════════════════
#  C — CLUSTER SIZE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_cluster_sizes(prot: pd.DataFrame):
    _section("C — Cluster size analysis")
    if "cluster_id" not in prot.columns:
        print("  ⚠ 'cluster_id' not found — skipping"); return

    clust_size = prot.groupby("cluster_id").size().rename("size")
    n_singletons = (clust_size == 1).sum()
    print(f"  Total clusters       : {len(clust_size):,}")
    print(f"  Singleton clusters   : {n_singletons:,}  "
          f"({n_singletons/len(clust_size)*100:.1f}%)")
    print(f"  Largest cluster size : {clust_size.max():,}")
    print(f"  Median cluster size  : {clust_size.median():.1f}")

    # top 20
    top20 = clust_size.nlargest(20).reset_index()
    top20.columns = ["cluster_id", "size"]
    top20.to_csv(SPLIT_ANALYSIS_DIR / "C_top20_clusters.csv", index=False)
    print("\n  Top 20 clusters:")
    print(top20.to_string(index=False))

    # cluster → split mapping
    clust_split = (prot.groupby("cluster_id")["split"]
                       .first().rename("split"))

    # ── Fig C1: cluster size histogram (log) ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=BG)
    for ax, log in zip(axes, [False, True]):
        _style(ax)
        ax.hist(clust_size.values, bins=60, color=SPLIT_COLORS["train"],
                alpha=0.80, zorder=3, linewidth=0, edgecolor=BG,
                log=log)
        ax.set_title(f"C1 — Cluster Size {'(log y)' if log else '(linear)'}")
        ax.set_xlabel("Cluster size (proteins)"); ax.set_ylabel("Count")
        ax.yaxis.grid(True); ax.xaxis.grid(False)
    plt.tight_layout(); _save(fig, "C1_cluster_size_hist")

    # ── Fig C2: boxplot by split ──────────────────────────────────────────────
    df_cs = clust_split.reset_index()
    df_cs = df_cs.merge(clust_size.reset_index(), on="cluster_id")
    df_cs = df_cs[df_cs["split"].isin(SPLIT_ORDER)]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    _style(ax)
    sns.boxplot(data=df_cs, x="split", y="size", order=SPLIT_ORDER,
                palette=SPLIT_COLORS, width=0.5, linewidth=0.8,
                flierprops=dict(marker=".", markersize=2,
                                alpha=0.3, color=TEXT_S), ax=ax)
    ax.set_yscale("log")
    ax.set_title("C2 — Cluster Size Distribution by Split")
    ax.set_xlabel("Split"); ax.set_ylabel("Cluster size (proteins, log)")
    plt.tight_layout(); _save(fig, "C2_cluster_size_by_split")

    # ── Fig C3: top-20 bar ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    _style(ax)
    colors = [SPLIT_COLORS.get(
                  clust_split.get(cid, "train"), SPLIT_COLORS["train"])
              for cid in top20["cluster_id"]]
    ax.bar(range(len(top20)), top20["size"], color=colors,
           zorder=3, linewidth=0)
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels([str(c)[:12] for c in top20["cluster_id"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_title("C3 — Top 20 Largest Clusters (color = split)")
    ax.set_ylabel("Cluster size"); ax.yaxis.grid(True); ax.xaxis.grid(False)
    # legend
    patches = [mpatches.Patch(color=SPLIT_COLORS[s], label=s)
               for s in SPLIT_ORDER]
    ax.legend(handles=patches, frameon=False, fontsize=9)
    plt.tight_layout(); _save(fig, "C3_top20_clusters")


# ═════════════════════════════════════════════════════════════════════════════
#  D — LEAKAGE SANITY CHECKS
# ═════════════════════════════════════════════════════════════════════════════

def sanity_checks(prot: pd.DataFrame, ann: pd.DataFrame) -> dict:
    _section("D — Leakage sanity checks")
    acc = _acc_col(prot)
    results = {}

    # 1. cluster leakage
    if "cluster_id" in prot.columns:
        clust_splits = (prot.groupby("cluster_id")["split"]
                            .nunique().rename("n_splits"))
        leaked = clust_splits[clust_splits > 1]
        results["cluster_leakage_count"] = len(leaked)
        if leaked.empty:
            print("  ✅ No cluster appears in more than one split (clean)")
        else:
            print(f"  ❌ {len(leaked):,} clusters span multiple splits — LEAKAGE!")
            leaked.to_csv(SPLIT_ANALYSIS_DIR / "D_leaked_clusters.csv")

    # 2. accession duplicates in proteins
    dup_acc = prot[acc].duplicated().sum()
    results["dup_accessions_proteins"] = int(dup_acc)
    print(f"  Duplicate accessions in proteins: {dup_acc:,}")

    # 3. accession in multiple splits
    acc_splits = prot.groupby(acc)["split"].nunique()
    multi_split = acc_splits[acc_splits > 1]
    results["accessions_in_multiple_splits"] = len(multi_split)
    if multi_split.empty:
        print("  ✅ No accession appears in multiple splits")
    else:
        print(f"  ❌ {len(multi_split):,} accessions appear in multiple splits!")
        multi_split.to_csv(SPLIT_ANALYSIS_DIR / "D_multi_split_accessions.csv")

    # 4. duplicate rows
    dup_rows_p = prot.duplicated().sum()
    dup_rows_a = ann.duplicated().sum()
    results["dup_rows_proteins"]    = int(dup_rows_p)
    results["dup_rows_annotations"] = int(dup_rows_a)
    print(f"  Duplicate rows — proteins: {dup_rows_p:,}, annotations: {dup_rows_a:,}")

    # 5. annotations in annotations_with_split vs proteins_with_split
    if "split" in ann.columns:
        ann_acc_col = _acc_col(ann)
        prot_acc_set = set(prot[acc])
        orphan = ~ann[ann_acc_col].isin(prot_acc_set)
        results["orphan_annotations"] = int(orphan.sum())
        print(f"  Annotations without matching protein: {orphan.sum():,}")

    # save results
    pd.Series(results).to_csv(SPLIT_ANALYSIS_DIR / "D_leakage_report.csv", header=["value"])
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  E — ANNOTATION BALANCE ACROSS SPLITS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_annotations(ann: pd.DataFrame):
    _section("E — Annotation balance across splits")

    top_types = (ann["feature_type"].value_counts()
                     .head(TOP_N_TYPES).index.tolist())
    sub = ann[ann["feature_type"].isin(top_types) &
              ann["split"].isin(SPLIT_ORDER)].copy()

    # pivot: counts
    pivot_counts = (sub.groupby(["feature_type", "split"])
                       .size().unstack(fill_value=0)
                       .reindex(columns=SPLIT_ORDER, fill_value=0))
    pivot_counts.to_csv(SPLIT_ANALYSIS_DIR / "E_annotation_counts.csv")

    # pivot: proportions (each type sums to 1 across splits)
    pivot_norm = pivot_counts.div(pivot_counts.sum(axis=1), axis=0)

    # ── Fig E1: heatmap counts ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 8), facecolor=BG)
    sns.heatmap(pivot_counts, ax=ax, cmap="Blues", fmt=",d",
                annot=True, annot_kws={"size": 8},
                linewidths=0.4, linecolor=GRID_C,
                cbar_kws={"label": "Count"})
    ax.set_title(f"E1 — Feature Type Counts by Split (top {TOP_N_TYPES})")
    ax.set_xlabel("Split"); ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9); ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout(); _save(fig, "E1_annotation_counts_heatmap")

    # ── Fig E2: normalised heatmap ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 8), facecolor=BG)
    sns.heatmap(pivot_norm, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.4, linecolor=GRID_C,
                cbar_kws={"label": "Fraction"})
    ax.set_title(f"E2 — Feature Type Proportions by Split (top {TOP_N_TYPES})")
    ax.set_xlabel("Split"); ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9); ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout(); _save(fig, "E2_annotation_proportions_heatmap")

    # ── Fig E3: grouped bar ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    _style(ax)
    x      = np.arange(len(top_types))
    width  = 0.26
    for i, (sp, col) in enumerate(zip(SPLIT_ORDER, SPLIT_PAL)):
        vals = [pivot_counts.loc[t, sp] if t in pivot_counts.index else 0
                for t in top_types]
        ax.bar(x + i * width, vals, width, label=sp, color=col,
               zorder=3, linewidth=0)
    ax.set_xticks(x + width)
    ax.set_xticklabels(top_types, rotation=40, ha="right", fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    ax.set_title(f"E3 — Annotation Counts by Split (top {TOP_N_TYPES})")
    ax.set_ylabel("Count"); ax.yaxis.grid(True); ax.xaxis.grid(False)
    ax.legend(frameon=False)
    plt.tight_layout(); _save(fig, "E3_annotation_grouped_bar")

    # ── Annotation density per protein per split ──────────────────────────────
    ann_acc = _acc_col(ann)
    density = (ann[ann["split"].isin(SPLIT_ORDER)]
               .groupby([ann_acc, "split"]).size()
               .reset_index(name="n_ann"))
    print("\n  Annotation density (annotations per protein):")
    print(density.groupby("split")["n_ann"].describe().round(2).to_string())


# ═════════════════════════════════════════════════════════════════════════════
#  F — ANNOTATION GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════

def analyze_annotation_geometry(ann: pd.DataFrame):
    _section("F — Annotation geometry across splits")

    sub = ann[ann["split"].isin(SPLIT_ORDER)].copy()

    # ── Fig F1: annotation length KDE by split ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    _style(ax)
    for sp in SPLIT_ORDER:
        vals = sub.loc[sub["split"] == sp, "ann_len"].clip(upper=1000)
        sns.kdeplot(vals, ax=ax, label=sp, color=SPLIT_COLORS[sp],
                    linewidth=1.8, fill=True, alpha=0.18)
    ax.set_title("F1 — Annotation Length KDE by Split (clipped at 1000 aa)")
    ax.set_xlabel("Annotation length (aa)"); ax.set_ylabel("Density")
    ax.legend(frameon=False)
    plt.tight_layout(); _save(fig, "F1_ann_length_kde")

    # ── Fig F2: boxplot by top feature types ──────────────────────────────────
    top_types = ann["feature_type"].value_counts().head(12).index.tolist()
    sub2 = sub[sub["feature_type"].isin(top_types)]
    sample = sub2.sample(min(len(sub2), SAMPLE_N), random_state=42)

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    _style(ax)
    sns.boxplot(data=sample, x="feature_type", y="ann_len",
                order=top_types, hue="split", hue_order=SPLIT_ORDER,
                palette=SPLIT_COLORS, width=0.65, linewidth=0.6,
                flierprops=dict(marker=".", markersize=1.5, alpha=0.2),
                ax=ax)
    ax.set_yscale("log")
    ax.set_title("F2 — Annotation Length by Feature Type and Split (log)")
    ax.set_xlabel("Feature type"); ax.set_ylabel("Length (aa, log)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha="right", fontsize=8)
    ax.legend(title="split", frameon=False, fontsize=8)
    plt.tight_layout(); _save(fig, "F2_ann_length_by_type")

    # stats
    print("\n  Annotation length stats by split:")
    print(sub.groupby("split")["ann_len"].describe().round(1).to_string())


# ═════════════════════════════════════════════════════════════════════════════
#  G — CLUSTER SIZE VS BIOLOGY
# ═════════════════════════════════════════════════════════════════════════════

def analyze_cluster_vs_biology(prot: pd.DataFrame):
    _section("G — Cluster size vs biology")
    if "cluster_id" not in prot.columns:
        print("  ⚠ 'cluster_id' not found — skipping"); return

    clust_size = prot.groupby("cluster_id").size().rename("clust_size")
    df = prot.merge(clust_size.reset_index(), on="cluster_id")

    sample = df.sample(min(len(df), SAMPLE_N), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    # ── G1: cluster size vs protein length ────────────────────────────────────
    ax = axes[0]; _style(ax)
    if "length" in sample.columns:
        ax.hexbin(np.log10(sample["clust_size"].clip(lower=1)),
                  np.log10(sample["length"].clip(lower=1)),
                  gridsize=40, cmap="Blues", mincnt=1, linewidths=0.1)
        # median trend
        sample["log_cs"] = np.log10(sample["clust_size"].clip(lower=1))
        bins_x = np.linspace(0, np.log10(sample["clust_size"].max()+1), 16)
        meds = []
        for lo, hi in zip(bins_x[:-1], bins_x[1:]):
            m = sample.loc[(sample["log_cs"] >= lo) & (sample["log_cs"] < hi),
                           "length"].median()
            if pd.notna(m): meds.append(((lo+hi)/2, np.log10(m)))
        if meds:
            mx, my = zip(*meds)
            ax.plot(mx, my, "o-", color="#E8593C", linewidth=1.8,
                    markersize=4, label="Median protein length")
            ax.legend(frameon=False, fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{10**v:.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{10**v:.0f}"))
        ax.set_title("G1 — Cluster Size vs Protein Length (hexbin, log)")
        ax.set_xlabel("Cluster size (proteins, log)")
        ax.set_ylabel("Protein length (aa, log)")

    # ── G2: cluster size vs annotation count ──────────────────────────────────
    ax = axes[1]; _style(ax)
    ax.hexbin(np.log10(sample["clust_size"].clip(lower=1)),
              np.log10(sample["n_annotations"].clip(lower=1)),
              gridsize=40, cmap="Greens", mincnt=1, linewidths=0.1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{10**v:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{10**v:.0f}"))
    ax.set_title("G2 — Cluster Size vs Annotation Count (hexbin, log)")
    ax.set_xlabel("Cluster size (proteins, log)")
    ax.set_ylabel("Annotations per protein (log)")

    plt.tight_layout(); _save(fig, "G_cluster_vs_biology")


# ═════════════════════════════════════════════════════════════════════════════
#  H — ANNOTATION BURDEN PER PROTEIN
# ═════════════════════════════════════════════════════════════════════════════

def analyze_annotation_burden(prot: pd.DataFrame):
    _section("H — Protein-level annotation burden")

    sub = prot[prot["split"].isin(SPLIT_ORDER)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

    # ── H1: n_annotations violin ──────────────────────────────────────────────
    ax = axes[0]; _style(ax)
    sample = sub.sample(min(len(sub), SAMPLE_N), random_state=42)
    sns.violinplot(data=sample, x="split", y="n_annotations",
                   order=SPLIT_ORDER, palette=SPLIT_COLORS,
                   inner="box", cut=0, linewidth=0.8, ax=ax)
    ax.set_yscale("log")
    ax.set_title("H1 — Annotations per Protein by Split (log)")
    ax.set_xlabel("Split"); ax.set_ylabel("N annotations (log)")

    # ── H2: density violin ────────────────────────────────────────────────────
    ax = axes[1]; _style(ax)
    if "ann_density" in sub.columns:
        sns.violinplot(data=sample, x="split", y="ann_density",
                       order=SPLIT_ORDER, palette=SPLIT_COLORS,
                       inner="box", cut=0, linewidth=0.8, ax=ax)
        ax.set_title("H2 — Annotation Density per 100 aa by Split")
        ax.set_xlabel("Split"); ax.set_ylabel("Features / 100 aa")

    plt.tight_layout(); _save(fig, "H_annotation_burden")

    # stats
    print("\n  Annotations per protein by split:")
    print(sub.groupby("split")["n_annotations"].describe().round(2).to_string())
    if "ann_density" in sub.columns:
        print("\n  Annotation density (/ 100 aa) by split:")
        print(sub.groupby("split")["ann_density"].describe().round(3).to_string())


# ═════════════════════════════════════════════════════════════════════════════
#  I — KEY FINDINGS
# ═════════════════════════════════════════════════════════════════════════════

def key_findings(prot: pd.DataFrame, ann: pd.DataFrame,
                 summary: pd.DataFrame, sanity: dict):
    _section("I — Key Findings (paper-ready)")

    lines = []

    # split sizes
    for sp in SPLIT_ORDER:
        if sp in summary.index:
            n  = int(summary.loc[sp, "n_proteins"])
            nc = summary.loc[sp, "n_clusters"]
            pct = summary.loc[sp, "pct_proteins"]
            lines.append(
                f"  {sp.capitalize():5s}: {n:>8,} proteins ({pct:.1f}%), "
                f"{int(nc):,} clusters"
            )

    lines.append("")

    # protein length medians
    if "length" in prot.columns:
        for sp in SPLIT_ORDER:
            med = prot.loc[prot["split"] == sp, "length"].median()
            lines.append(f"  Median protein length [{sp}]: {int(med):,} aa")
        lines.append("")

    # cluster stats
    if "cluster_id" in prot.columns:
        clust_size = prot.groupby("cluster_id").size()
        n_sing = (clust_size == 1).sum()
        lines.append(f"  Total clusters      : {len(clust_size):,}")
        lines.append(f"  Singleton clusters  : {n_sing:,} "
                     f"({n_sing/len(clust_size)*100:.1f}%)")
        lines.append(f"  Largest cluster     : {clust_size.max():,} proteins")
        lines.append("")

    # leakage
    lk = sanity.get("cluster_leakage_count", 0)
    if lk == 0:
        lines.append("  ✅ No homology leakage detected (all clusters confined to one split)")
    else:
        lines.append(f"  ❌ {lk:,} clusters detected with homology leakage across splits")

    dup_acc = sanity.get("dup_accessions_proteins", 0)
    if dup_acc == 0:
        lines.append("  ✅ No duplicate accessions in protein table")
    else:
        lines.append(f"  ⚠  {dup_acc:,} duplicate accessions found")

    multi = sanity.get("accessions_in_multiple_splits", 0)
    if multi == 0:
        lines.append("  ✅ No protein appears in more than one split")
    else:
        lines.append(f"  ❌ {multi:,} proteins appear in multiple splits")

    lines.append("")

    # annotation balance
    pivot = (ann[ann["split"].isin(SPLIT_ORDER)]
            .groupby(["feature_type", "split"]).size()
            .unstack(fill_value=0)
            .reindex(columns=SPLIT_ORDER, fill_value=0))

    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    expected = (
        ann[ann["split"].isin(SPLIT_ORDER)]["split"]
        .value_counts(normalize=True)
        .reindex(SPLIT_ORDER)
    )

    dev = pivot_norm.sub(expected, axis=1).abs()
    max_dev = dev.max(axis=1).max()
    feature_dev = dev.max(axis=1).sort_values(ascending=False)

    lines.append("  Expected annotation split proportions:")
    for s in SPLIT_ORDER:
        lines.append(f"    {s}: {expected[s]:.3f}")

    lines.append(
        "  Max annotation proportion deviation from expected "
        f"(using actual annotation split proportions): {max_dev:.3f}"
    )

    if max_dev < 0.05:
        lines.append("  ✅ Annotation distribution appears well-balanced across splits")
    else:
        lines.append("  ⚠  Some feature types are unevenly distributed across splits")

    lines.append("  Top 5 most uneven feature types:")
    for ft, d in feature_dev.head(5).items():
        lines.append(f"    {ft}: max deviation {d:.3f}")

    output = "\n".join(lines)
    print(output)

    # save to txt
    with open(SPLIT_ANALYSIS_DIR / "I_key_findings.txt", "w") as f:
        f.write("KEY FINDINGS — PIPLV2 Split Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(output + "\n")
    print(f"\n  Saved to {SPLIT_ANALYSIS_DIR / 'I_key_findings.txt'}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

import matplotlib.patches as mpatches  # needed in C3

def main():
    print("=" * 62)
    print("  PIPLV2 — Split Quality Analysis")
    print(f"  Output: {SPLIT_ANALYSIS_DIR}")
    print("=" * 62)

    prot, ann = load_data()

    summary = summarize_splits(prot, ann)         # A
    analyze_lengths(prot)                          # B
    analyze_cluster_sizes(prot)                    # C
    sanity  = sanity_checks(prot, ann)             # D
    analyze_annotations(ann)                       # E
    analyze_annotation_geometry(ann)               # F
    analyze_cluster_vs_biology(prot)               # G
    analyze_annotation_burden(prot)                # H
    key_findings(prot, ann, summary, sanity)       # I

    print(f"\n✨  All done — {len(list(SPLIT_ANALYSIS_DIR.glob('*.png')))} figures + "
          f"{len(list(SPLIT_ANALYSIS_DIR.glob('*.csv')))} CSV files saved to:\n   {SPLIT_ANALYSIS_DIR}")


if __name__ == "__main__":
    main()