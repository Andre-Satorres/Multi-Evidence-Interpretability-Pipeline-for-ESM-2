"""
ds_analisys.py — PIPLV2
Scales to 500k+ proteins / 5M+ annotations.

Usage
-----
  # original annotations:
  python src/dsPrep/ds_analisys.py

  # enriched annotations (29 types, 147k subtypes):
  python src/dsPrep/ds_analisys.py \\
      --annotations data/annotations_enriched.tsv \\
      --label enriched

  # only specific figures:
  python src/dsPrep/ds_analisys.py --figures 2 3 4

  # filter to test split only:
  python src/dsPrep/ds_analisys.py \\
      --annotations data/annotations_enriched.tsv \\
      --split test --label enriched_test
"""

import sys, argparse, warnings
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from constants import PROTEINS_TSV_PATH, FIGURES_DIR

PALETTE = [
    "#3266AD","#1D9E75","#E8593C","#BA7517","#9B59B6","#D45384",
    "#0F6E56","#993C1D","#5B7FBF","#48B89F","#C0392B","#2980B9",
    "#7F8C8D","#F39C12","#8E44AD","#16A085","#D35400","#27AE60",
    "#2C3E50","#E74C3C","#1ABC9C","#F1C40F","#34495E","#E67E22",
]
BG      = "#F8F7F4"
CARD_BG = "#FFFFFF"
GRID_C  = "#E8E6E0"
TEXT_P  = "#1A1917"
TEXT_S  = "#6B6964"
ACCENT  = "#3266AD"
HEAT_CMAP = LinearSegmentedColormap.from_list("ht", ["#F8F7F4","#3266AD","#1A3A6B"])
TOP_N        = 15
SAMPLE_COO   = 50_000
SAMPLE_LARGE = 200_000

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_C, "axes.labelcolor": TEXT_S,
    "axes.titlecolor": TEXT_P, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID_C, "grid.linewidth": 0.6,
    "xtick.color": TEXT_S, "ytick.color": TEXT_S,
    "font.family": "DejaVu Sans",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10,
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cmap(labels):
    return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labels)}

def _style_ax(ax):
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID_C); sp.set_linewidth(0.5)
    ax.set_axisbelow(True)

def _acc_col(df):
    return next((c for c in df.columns
                 if c in ("accession","entry","protein_id","seqname")), df.columns[0])

def _watermark(fig, label=""):
    txt = f"PIPLV2 · dsPrep{(' · ' + label) if label else ''}"
    fig.text(0.99, 0.002, txt, ha="right", va="bottom", fontsize=7.5, color=GRID_C)

def _top_types(ann, n=TOP_N):
    counts = ann["feature_type"].value_counts()
    return counts.index[:n].tolist(), counts

def _has_subtypes(ann):
    return ("annot_subtype" in ann.columns
            and ann["annot_subtype"].notna().any()
            and ann["annot_subtype"].nunique() > ann["feature_type"].nunique())


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(annotations_path, proteins_path):
    print(f"    annotations: {annotations_path.name}", flush=True)
    ann  = pd.read_csv(annotations_path, sep="\t", low_memory=False)
    print(f"    proteins:    {proteins_path.name}", flush=True)
    prot = pd.read_csv(proteins_path,    sep="\t", low_memory=False)
    ann.columns  = ann.columns.str.strip().str.lower()
    prot.columns = prot.columns.str.strip().str.lower()
    ann["feat_len"] = (ann["end"].astype(int) - ann["start"].astype(int)).abs() + 1
    acc     = _acc_col(ann)
    len_map = prot.set_index(prot.columns[0])["length"].to_dict()
    ann["prot_len"] = ann[acc].map(len_map)
    ann["rel_pos"]  = ((ann["start"].astype(int) + ann["end"].astype(int)) / 2) / ann["prot_len"].replace(0, np.nan)
    if "annot_subtype" not in ann.columns:
        ann["annot_subtype"] = ann["feature_type"]
    return ann, prot


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_distribution(ann, ax):
    counts = ann["feature_type"].value_counts()
    top    = counts.head(TOP_N)
    acc    = _acc_col(ann)
    prot_c = ann.groupby("feature_type")[acc].nunique().reindex(top.index)
    y = range(len(top))
    cm = _cmap(top.index.tolist())
    ax.barh(list(y), top.values, color=[cm[t] for t in top.index], alpha=0.85, zorder=3, linewidth=0)
    ax2 = ax.twiny()
    ax2.barh(list(y), prot_c.values, color=[cm[t] for t in top.index], alpha=0.35, zorder=2, linewidth=0)
    ax.set_yticks(list(y)); ax.set_yticklabels(top.index.tolist(), fontsize=8)
    ax.set_title(f"1 — Distribuição de Feature Types (top {TOP_N})")
    ax.set_xlabel("Nº de anotações"); ax2.set_xlabel("Nº de proteínas (sombreado)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v/1e3)}k" if v >= 1e3 else str(int(v))))
    ax.xaxis.grid(True); ax.yaxis.grid(False)

def plot_feature_sizes(ann, ax):
    top, _ = _top_types(ann)
    sub    = ann[ann["feature_type"].isin(top)].copy()
    order  = (sub.groupby("feature_type")["feat_len"].median().sort_values().index.tolist())
    cm     = _cmap(order)
    data   = [sub.loc[sub["feature_type"]==t, "feat_len"].clip(upper=2000).values for t in order]
    bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.55,
                    medianprops=dict(color="white", linewidth=1.8),
                    whiskerprops=dict(color=GRID_C), capprops=dict(color=GRID_C), boxprops=dict(linewidth=0))
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(cm[t]); patch.set_alpha(0.85)
    ax.set_yticks(range(1, len(order)+1)); ax.set_yticklabels(order, fontsize=8)
    ax.set_title(f"2 — Comprimento das Features (top {TOP_N})")
    ax.set_xlabel("Comprimento (aa)"); ax.xaxis.grid(True); ax.yaxis.grid(False)

def plot_coverage_per_protein(ann, ax):
    acc    = _acc_col(ann)
    top, _ = _top_types(ann, n=20)
    sub    = ann[ann["feature_type"].isin(top)]
    counts = (sub.groupby([acc, "feature_type"]).size().reset_index(name="n")
              .groupby("feature_type")["n"].mean().reindex(top).sort_values(ascending=True))
    cm = _cmap(counts.index.tolist())
    ax.barh(range(len(counts)), counts.values, color=[cm[t] for t in counts.index], alpha=0.85, zorder=3, linewidth=0)
    ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index.tolist(), fontsize=8)
    ax.set_title("3 — Média de Features por Proteína (top 20)"); ax.set_xlabel("Média")
    ax.xaxis.grid(True); ax.yaxis.grid(False)

def plot_balance_by_type(ann, ax):
    acc = _acc_col(ann)
    if "split" not in ann.columns:
        ax.text(0.5, 0.5, "Coluna 'split' não encontrada", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_S)
        ax.set_title("4 — Balanço por Split"); return
    top, _ = _top_types(ann, n=10)
    sub    = ann[ann["feature_type"].isin(top)]
    pivot  = (sub.groupby(["feature_type","split"])[acc].nunique().unstack(fill_value=0))
    splits = [c for c in ["train","val","test"] if c in pivot.columns]
    pivot  = pivot[splits].loc[top]
    x = np.arange(len(top)); w = 0.25
    colors = {"train":"#3266AD","val":"#1D9E75","test":"#E8593C"}
    for i, sp in enumerate(splits):
        ax.bar(x + i*w, pivot[sp], w, label=sp, color=colors.get(sp, PALETTE[i]), alpha=0.85, zorder=3, linewidth=0)
    ax.set_xticks(x + w); ax.set_xticklabels(top, rotation=35, ha="right", fontsize=8)
    ax.set_title("4 — Proteínas por Split e Feature Type"); ax.set_ylabel("Nº de proteínas")
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v/1e3)}k" if v >= 1e3 else str(int(v))))
    ax.yaxis.grid(True); ax.xaxis.grid(False)

def make_figure1(ann, prot, label=""):
    n_types = ann["feature_type"].nunique()
    fig = plt.figure(figsize=(18, max(12, 5 + n_types * 0.28)), facecolor=BG)
    fig.suptitle("Annotation Dataset — Análises Originais", fontsize=15, fontweight="bold", color=TEXT_P, y=0.995)
    gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38, left=0.18, right=0.97, top=0.96, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axes: _style_ax(ax)
    plot_feature_distribution(ann, axes[0]); plot_feature_sizes(ann, axes[1])
    plot_coverage_per_protein(ann, axes[2]); plot_balance_by_type(ann, axes[3])
    _watermark(fig, label); return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2
# ══════════════════════════════════════════════════════════════════════════════
def plot_density_boxplot(ann, ax):
    top, _ = _top_types(ann); acc = _acc_col(ann)
    sub = ann[ann["feature_type"].isin(top)].copy()
    prot_len = sub.groupby(acc)["prot_len"].first()
    counts = sub.groupby([acc, "feature_type"]).size().rename("n")
    df = counts.reset_index(); df["prot_len"] = df[acc].map(prot_len)
    df["density"] = df["n"] / df["prot_len"] * 100
    order = (df.groupby("feature_type")["density"].median().sort_values(ascending=True).index.tolist())
    cm = _cmap(order)
    bp = ax.boxplot([df.loc[df["feature_type"]==t, "density"].values for t in order],
                    vert=False, patch_artist=True, showfliers=False, widths=0.55,
                    medianprops=dict(color="white", linewidth=1.8),
                    whiskerprops=dict(color=GRID_C), capprops=dict(color=GRID_C), boxprops=dict(linewidth=0))
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(cm[t]); patch.set_alpha(0.85)
    ax.set_yticks(range(1, len(order)+1)); ax.set_yticklabels(order, fontsize=8)
    ax.set_title(f"5 — Densidade por Proteína (top {TOP_N})"); ax.set_xlabel("Features / 100 aa")
    ax.xaxis.grid(True); ax.yaxis.grid(False)

def plot_ridge(ann, ax):
    top, _ = _top_types(ann)
    sub = ann[ann["feature_type"].isin(top)].dropna(subset=["rel_pos"])
    order = (sub.groupby("feature_type")["rel_pos"].median().sort_values().index.tolist())
    cm = _cmap(order); spacing = 1.0
    for i, ft in enumerate(order):
        vals = sub.loc[sub["feature_type"]==ft, "rel_pos"].values
        if len(vals) < 5: continue
        xs = np.linspace(0, 1, 200)
        hist, edges = np.histogram(vals, bins=80, range=(0, 1), density=True)
        ys = np.interp(xs, (edges[:-1]+edges[1:])/2, hist)
        ys = ys / ys.max() * spacing * 0.85
        base = i * spacing
        ax.fill_between(xs, base, base+ys, color=cm[ft], alpha=0.75, zorder=i+1)
        ax.plot(xs, base+ys, color=cm[ft], linewidth=0.8, zorder=i+2)
        med = np.median(vals)
        ax.plot([med,med],[base, base+ys[np.argmin(np.abs(xs-med))]], color="white", linewidth=1.2, zorder=i+3)
    ax.set_yticks([i*spacing+spacing*0.4 for i in range(len(order))]); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlim(0, 1)
    ax.axvline(0.33, color=GRID_C, linewidth=0.8, linestyle="--"); ax.axvline(0.66, color=GRID_C, linewidth=0.8, linestyle="--")
    ax.text(0.01,-0.6,"N-term",fontsize=7.5,color=TEXT_S); ax.text(0.68,-0.6,"C-term",fontsize=7.5,color=TEXT_S)
    ax.set_title(f"6 — Posição Relativa (ridge KDE, top {TOP_N})"); ax.set_xlabel("Posição relativa (0=N → 1=C)")
    ax.xaxis.grid(False); ax.yaxis.grid(False); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(True)

def plot_cooccurrence(ann, ax):
    acc = _acc_col(ann); top, _ = _top_types(ann, n=TOP_N)
    sub = ann[ann["feature_type"].isin(top)]
    all_prots = sub[acc].unique()
    note = ""
    if len(all_prots) > SAMPLE_COO:
        sub = sub[sub[acc].isin(np.random.choice(all_prots, SAMPLE_COO, replace=False))]
        note = f"(amostrado {SAMPLE_COO:,} proteínas)"
    n = len(top); idx = {t:i for i,t in enumerate(top)}
    mat = np.zeros((n,n), dtype=np.int32)
    for _, grp in sub.groupby(acc):
        present = [t for t in grp["feature_type"].unique() if t in idx]
        for a in present:
            for b in present: mat[idx[a], idx[b]] += 1
    np.fill_diagonal(mat, 0)
    im = ax.imshow(mat, cmap=HEAT_CMAP, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_xticklabels(top, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(top, fontsize=7.5); ax.grid(False)
    vmax = mat.max()
    for i in range(n):
        for j in range(n):
            v = mat[i,j]
            if v > 0:
                ax.text(j,i, f"{v/1e3:.0f}k" if v>=1000 else str(v), ha="center", va="center", fontsize=6,
                        color="white" if v > vmax*0.5 else TEXT_P)
    plt.colorbar(im, ax=ax, fraction=0.032, pad=0.02, label="Proteínas em comum")
    ax.set_title(f"7 — Co-ocorrência{chr(10)+note if note else ''}", fontsize=10)

def plot_hexbin(ann, ax):
    top, _ = _top_types(ann)
    sub = ann[ann["feature_type"].isin(top)].dropna(subset=["prot_len"])
    if len(sub) > SAMPLE_LARGE: sub = sub.sample(SAMPLE_LARGE, random_state=42)
    x = np.log10(sub["prot_len"].clip(lower=1)); y = np.log10(sub["feat_len"].clip(lower=1))
    hb = ax.hexbin(x, y, gridsize=45, cmap=HEAT_CMAP, mincnt=1, linewidths=0.1, zorder=3)
    plt.colorbar(hb, ax=ax, label="Contagem", fraction=0.032, pad=0.02)
    lim = max(x.max(), y.max())
    ax.plot([0,lim],[0,lim],"--",color="#E8593C",linewidth=1.2,label="feat=prot",zorder=4)
    ax.legend(fontsize=8, frameon=False)
    for a in (ax.xaxis,ax.yaxis):
        a.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{10**v:.0f}" if v<3 else f"{10**v/1e3:.0f}k"))
    ax.set_title("8 — Comprimento Proteína × Feature (hexbin, log)")
    ax.set_xlabel("Comprimento proteína (aa, log)"); ax.set_ylabel("Comprimento feature (aa, log)")
    ax.xaxis.grid(True); ax.yaxis.grid(True)

def make_figure2(ann, prot, label=""):
    n_types = min(ann["feature_type"].nunique(), TOP_N)
    fig = plt.figure(figsize=(18, max(14, 6+n_types*0.35)), facecolor=BG)
    fig.suptitle("Annotation Dataset — Análises de Anotação", fontsize=15, fontweight="bold", color=TEXT_P, y=0.995)
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.40, left=0.16, right=0.97, top=0.95, bottom=0.06)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]
    for ax in axes: _style_ax(ax)
    plot_density_boxplot(ann, axes[0]); plot_ridge(ann, axes[1])
    plot_cooccurrence(ann, axes[2]); plot_hexbin(ann, axes[3])
    _watermark(fig, label); return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3
# ══════════════════════════════════════════════════════════════════════════════
def plot_protein_lengths(prot, ax):
    lengths = prot["length"].dropna()
    use_log = lengths.max() / (lengths.min()+1) > 100
    bins = np.logspace(np.log10(lengths.min()+1), np.log10(lengths.max()+1), 50) if use_log else 40
    if use_log: ax.set_xscale("log")
    ax.hist(lengths, bins=bins, color=ACCENT, alpha=0.80, zorder=3, linewidth=0, edgecolor=BG)
    med = lengths.median()
    ax.axvline(med, color="#E8593C", linewidth=1.5, linestyle="--", label=f"Mediana: {int(med):,} aa", zorder=4)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("9 — Distribuição do Comprimento das Proteínas")
    ax.set_xlabel("Comprimento (aa)"); ax.set_ylabel("Nº de proteínas")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1e3)}k" if x>=1e3 else str(int(x))))
    ax.yaxis.grid(True); ax.xaxis.grid(False)

def plot_aa_composition(prot, ax):
    seq_col = next((c for c in prot.columns if c in ("sequence","seq")), None)
    if seq_col is None:
        ax.text(0.5,0.5,"Coluna 'sequence' não encontrada",ha="center",va="center",transform=ax.transAxes,color=TEXT_S)
        ax.set_title("10 — Composição de Aminoácidos"); return
    AAs = list("ACDEFGHIKLMNPQRSTVWY"); cnt = Counter(); total = 0
    seqs = prot[seq_col].dropna()
    if len(seqs) > 50_000: seqs = seqs.sample(50_000, random_state=42)
    for seq in seqs:
        s = seq.upper(); cnt.update(c for c in s if c in AAs); total += sum(1 for c in s if c in AAs)
    if total == 0:
        ax.text(0.5,0.5,"Sequências vazias",ha="center",va="center",transform=ax.transAxes,color=TEXT_S); return
    ordered = sorted({aa: cnt[aa]/total*100 for aa in AAs}.items(), key=lambda x: -x[1])
    labels, vals = zip(*ordered)
    bars = ax.bar(labels, vals, color=[PALETTE[i%len(PALETTE)] for i in range(len(labels))], width=0.7, zorder=3, linewidth=0)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+ax.get_ylim()[1]*0.01, f"{h:.1f}",
                ha="center", va="bottom", fontsize=6.5, color=TEXT_S)
    ax.set_title("10 — Composição de Aminoácidos (freq. média)")
    ax.set_xlabel("Aminoácido"); ax.set_ylabel("Frequência (%)")
    ax.yaxis.grid(True); ax.xaxis.grid(False)

def plot_length_vs_features_hexbin(ann, prot, ax):
    acc = _acc_col(ann)
    counts = ann.groupby(acc).size().rename("n")
    lengths = prot.set_index(prot.columns[0])["length"]
    merged = pd.concat([counts, lengths], axis=1).dropna()
    if len(merged) > SAMPLE_LARGE: merged = merged.sample(SAMPLE_LARGE, random_state=42)
    x = np.log10(merged["length"].clip(lower=1)); y = np.log10(merged["n"].clip(lower=1))
    hb = ax.hexbin(x, y, gridsize=40, cmap=HEAT_CMAP, mincnt=1, linewidths=0.1, zorder=3)
    plt.colorbar(hb, ax=ax, label="Nº de proteínas", fraction=0.032, pad=0.02)
    bins_x = np.linspace(x.min(), x.max(), 20); meds = []
    for lo, hi in zip(bins_x[:-1], bins_x[1:]):
        mask = (x>=lo) & (x<hi)
        if mask.sum() > 10: meds.append(((lo+hi)/2, np.median(y[mask])))
    if meds:
        mx, my = zip(*meds)
        ax.plot(mx, my, "o-", color="#E8593C", linewidth=1.8, markersize=4, label="Mediana", zorder=5)
        ax.legend(fontsize=8.5, frameon=False)
    for a in (ax.xaxis,ax.yaxis):
        a.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{10**v:.0f}" if v<3 else f"{10**v/1e3:.0f}k"))
    ax.set_title("11 — Comprimento da Proteína × Nº de Features (hexbin)")
    ax.set_xlabel("Comprimento (aa, log)"); ax.set_ylabel("Nº de features (log)")
    ax.xaxis.grid(True); ax.yaxis.grid(True)

def make_figure3(ann, prot, label=""):
    fig = plt.figure(figsize=(18, 6), facecolor=BG)
    fig.suptitle("Annotation Dataset — Análises de Sequência", fontsize=15, fontweight="bold", color=TEXT_P, y=1.01)
    gs = GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.42, left=0.07, right=0.98, top=0.90, bottom=0.14)
    axes = [fig.add_subplot(gs[0,c]) for c in range(3)]
    for ax in axes: _style_ax(ax)
    plot_protein_lengths(prot, axes[0]); plot_aa_composition(prot, axes[1])
    plot_length_vs_features_hexbin(ann, prot, axes[2])
    _watermark(fig, label); return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4  — Enriched Subtypes (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def plot_top_subtypes(ann, ax, top_n=30):
    acc = _acc_col(ann)
    # prefer subtypes that differ from their feature_type (i.e. have descriptions)
    sub_ann = ann[ann["annot_subtype"] != ann["feature_type"]]
    if sub_ann.empty: sub_ann = ann
    counts = sub_ann.groupby("annot_subtype")[acc].nunique().sort_values(ascending=False).head(top_n)
    s2t = ann.drop_duplicates("annot_subtype").set_index("annot_subtype")["feature_type"].to_dict()
    top_types = ann["feature_type"].value_counts().head(len(PALETTE)).index.tolist()
    tc = {t: PALETTE[i%len(PALETTE)] for i,t in enumerate(top_types)}
    labels = [s[:55]+"…" if len(s)>55 else s for s in counts.index]
    colors = [tc.get(s2t.get(s,""), "#7F8C8D") for s in counts.index]
    ax.barh(range(len(counts)), counts.values, color=colors, alpha=0.85, zorder=3, linewidth=0)
    ax.set_yticks(range(len(counts))); ax.set_yticklabels(labels, fontsize=7.5); ax.invert_yaxis()
    ax.set_title(f"12 — Top {top_n} Subtypes por Nº de Proteínas"); ax.set_xlabel("Nº de proteínas")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v/1e3)}k" if v>=1e3 else str(int(v))))
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    handles = [mpatches.Patch(color=c, label=t, alpha=0.85) for t,c in list(tc.items())[:10]]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right", ncol=2)

def plot_subtypes_per_type(ann, ax):
    sub_ann = ann[ann["annot_subtype"] != ann["feature_type"]]
    if sub_ann.empty: sub_ann = ann
    counts = sub_ann.groupby("feature_type")["annot_subtype"].nunique().sort_values(ascending=True)
    cm = _cmap(counts.index.tolist())
    ax.barh(range(len(counts)), counts.values, color=[cm[t] for t in counts.index], alpha=0.85, zorder=3, linewidth=0)
    ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index.tolist(), fontsize=8)
    ax.set_title("13 — Subtypes Distintos por Feature Type"); ax.set_xlabel("Nº de subtypes únicos")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v/1e3)}k" if v>=1e3 else str(int(v))))
    ax.xaxis.grid(True); ax.yaxis.grid(False)

def plot_subtype_frequency_dist(ann, ax, min_proteins=5):
    acc = _acc_col(ann)
    counts = ann.groupby("annot_subtype")[acc].nunique()
    counts = counts[counts >= min_proteins]
    bins = np.logspace(np.log10(max(counts.min(),1)), np.log10(counts.max()), 40)
    ax.hist(counts.values, bins=bins, color=ACCENT, alpha=0.80, zorder=3, linewidth=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    for thresh, color, lbl in [(100,"#E8593C","≥100"),(1000,"#BA7517","≥1000")]:
        n = (counts >= thresh).sum()
        ax.axvline(thresh, color=color, linewidth=1.2, linestyle="--", label=f"{lbl}: {n:,} subtypes")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title(f"14 — Distribuição de Frequência dos Subtypes\n({len(counts):,} subtypes com ≥{min_proteins} proteínas)")
    ax.set_xlabel("Nº de proteínas por subtype (log)"); ax.set_ylabel("Nº de subtypes (log)")
    ax.xaxis.grid(True); ax.yaxis.grid(True)

def plot_top5_per_type(ann, ax):
    acc = _acc_col(ann)
    interesting = ["Modified residue","Binding site","Zinc finger","Topological domain",
                   "Compositional bias","Region","Transit peptide","Active site","Domain","Repeat"]
    available = [t for t in interesting if t in ann["feature_type"].values][:8]
    type_colors = {t: PALETTE[i%len(PALETTE)] for i,t in enumerate(available)}
    yticks=[]; yticklabels=[]; pos=0; gap=2
    for ft in available:
        sub = ann[(ann["feature_type"]==ft) & (ann["annot_subtype"]!=ft)]
        if sub.empty: continue
        top5 = sub.groupby("annot_subtype")[acc].nunique().sort_values(ascending=False).head(5)
        for subtype, cnt in top5.items():
            lbl = subtype.replace(f"{ft}: ","")[:42]
            ax.barh(pos, cnt, color=type_colors[ft], alpha=0.85, zorder=3, linewidth=0)
            yticks.append(pos); yticklabels.append(lbl); pos+=1
        pos += gap
    ax.set_yticks(yticks); ax.set_yticklabels(yticklabels, fontsize=7); ax.invert_yaxis()
    ax.set_title("15 — Top 5 Subtypes por Feature Type"); ax.set_xlabel("Nº de proteínas")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v/1e3)}k" if v>=1e3 else str(int(v))))
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    handles = [mpatches.Patch(color=type_colors[t], label=t, alpha=0.85) for t in available]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower right")

def make_figure4(ann, prot, label=""):
    if not _has_subtypes(ann):
        print("  ⚠ Figure 4 skipped: no annot_subtype or same as feature_type.")
        print("    Run with --annotations data/annotations_enriched.tsv")
        return None
    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle("Enriched Annotation Subtypes — Granularity Analysis",
                 fontsize=15, fontweight="bold", color=TEXT_P, y=0.995)
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.50,
                  left=0.22, right=0.97, top=0.95, bottom=0.05)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]
    for ax in axes: _style_ax(ax)
    plot_top_subtypes(ann, axes[0], top_n=30)
    plot_subtypes_per_type(ann, axes[1])
    plot_subtype_frequency_dist(ann, axes[2])
    plot_top5_per_type(ann, axes[3])
    _watermark(fig, label); return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CLI + Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    from constants import ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--annotations", type=Path, default=ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH,
                   help="Annotation TSV. Use data/annotations_enriched.tsv for enriched subtypes.")
    p.add_argument("--proteins",    type=Path, default=PROTEINS_TSV_PATH)
    p.add_argument("--outdir",      type=Path, default=FIGURES_DIR)
    p.add_argument("--label",       default="",
                   help="Label added to filenames and watermark (e.g. 'enriched').")
    p.add_argument("--figures",     nargs="+", type=int, default=[1,2,3,4], choices=[1,2,3,4])
    p.add_argument("--split",       default=None,
                   help="Filter to specific split (train/val/test). Default: all.")
    p.add_argument("--dpi",         type=int, default=160)
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""

    print("📂  Carregando dados...")
    ann, prot = load_data(args.annotations, args.proteins)

    if args.split and "split" in ann.columns:
        ann = ann[ann["split"] == args.split].copy()
        print(f"    Filtered to split='{args.split}'")

    acc = _acc_col(ann)
    has_sub = _has_subtypes(ann)
    print(f"    {len(ann):,} anotações  |  "
          f"{ann['feature_type'].nunique()} feature types  |  "
          f"{ann['annot_subtype'].nunique() if has_sub else 'N/A'} subtypes  |  "
          f"{ann[acc].nunique():,} proteínas")

    fig_map = {
        1: (f"fig1_original{suffix}",   lambda: make_figure1(ann, prot, args.label)),
        2: (f"fig2_annotation{suffix}", lambda: make_figure2(ann, prot, args.label)),
        3: (f"fig3_sequence{suffix}",   lambda: make_figure3(ann, prot, args.label)),
        4: (f"fig4_subtypes{suffix}",   lambda: make_figure4(ann, prot, args.label)),
    }

    for fig_n in args.figures:
        name, make_fn = fig_map[fig_n]
        print(f"\n📊  Figura {fig_n} ({name})...")
        fig = make_fn()
        if fig is None: continue
        path = args.outdir / f"{name}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"    ✅ {path}")

    print(f"\n✨  Pronto! Arquivos em: {args.outdir}")


if __name__ == "__main__":
    main()