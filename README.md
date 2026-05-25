# SAE Interpretability Pipeline for Protein Language Models

[![Dataset on Zenodo](https://img.shields.io/badge/dataset-10.5281%2Fzenodo.20379180-blue)](https://doi.org/10.5281/zenodo.20379180)

Residue-level interpretability of ESM-2 embeddings via Sparse Autoencoders (SAEs).
Aligns learned SAE features against curated Swiss-Prot annotations using a
homology-aware evaluation protocol, with multi-evidence verification and
attention baselines.

## Requirements

- Python 3.10+
- CUDA GPU (A5500 or better recommended)
- ~60 GB disk for outputs, ~4 GB for data

```bash
pip install torch transformers pandas numpy scipy scikit-learn matplotlib seaborn tqdm
pip install fair-esm  # optional, falls back to HuggingFace
```

---

## Project Structure

```
src/
  constants.py              # Shared paths (DATA_DIR, OUT_DIR, etc.)
  dsPrep/                   # Phase 0: dataset preparation
  embeddings/               # Phase 1: ESM-2 embedding extraction
  stream/                   # Phase 2-3: SAE training + feature alignment
  verification/             # Phase 4: multi-evidence verification + figures
data/                       # Proteins, annotations, clusters, splits
outputs/                    # All pipeline outputs (gitignored)
```

---

## Full Pipeline

### Phase 0 — Dataset Preparation

All scripts under `src/dsPrep/`. Run from the project root.

#### 0.1 Convert raw annotations

```bash
python src/dsPrep/gff_to_tsv.py
```

Reads `data/annotations.gff` → writes `data/annotations.tsv`.

#### 0.2 Convert proteins to FASTA

```bash
python src/dsPrep/tsv_to_fasta.py
```

Reads `data/proteins.tsv` → writes `data/proteins.fasta`.

#### 0.3 Homology clustering + train/val/test split

Cluster proteins with MMseqs2 (external), then split by cluster:

```bash
# MMseqs2 clustering (run externally, results in data/clusters.tsv)
python src/dsPrep/split_clusters.py 0.70 0.15
```

Reads `data/clusters.tsv` → writes `data/protein_splits.tsv` (70/15/15 split).

#### 0.4 Merge split info into protein/annotation tables

```bash
python src/dsPrep/add_split_info.py
```

Writes `data/proteins_with_split.tsv` and `data/annotations_with_split.tsv`.

#### 0.5 Deduplicate annotations

```bash
python src/dsPrep/remove_annotation_duplicates.py
```

Writes `data/annotations_dedup_with_split.tsv`.

#### 0.6 Download enriched Swiss-Prot features

```bash
python src/dsPrep/download_swissprot_features.py
# or, to parse only (if raw TSV already downloaded):
python src/dsPrep/download_swissprot_features.py --parse-only
```

Downloads all Swiss-Prot features via UniProt REST API. Writes
`data/annotations_enriched.tsv` with columns: `accession, feature_type,
start, end, description, annot_subtype, split, cluster_id`.

#### 0.7 (Optional) Dataset analysis figures

```bash
python src/dsPrep/ds_analisys.py --figures 1 2 3 4
python src/dsPrep/split_analysis.py
```

Writes analysis figures and CSVs to `outputs/figures/` and `outputs/split_analysis/`.

---

### Phase 1 — Embeddings

#### 1.1 (Optional) Create a subset for fast iteration

```bash
python src/embeddings/create_subset.py --fraction 0.10
```

Writes cluster-preserving subset to `outputs/subsets/`.

#### 1.2 Extract ESM-2 embeddings

```bash
python src/embeddings/extract_embeddings.py \
    --input data/proteins_with_split.tsv \
    --outdir outputs/embeddings/esm2_650m \
    --model facebook/esm2_t33_650M_UR50D \
    --splits train val test \
    --resume
```

Writes per-split `.pt` shard files to `outputs/embeddings/esm2_650m/`.
Each shard contains `{accessions, splits, lengths, embeddings [L,D]}`.

> **Note:** Pre-extracted embeddings are only needed for the attention
> baseline (`extract_attention.py`). SAE training and alignment use
> streaming ESM-2 inference directly.

---

### Phase 2 — SAE Training

```bash
python src/stream/train_sae_variants.py \
    --architecture topk \
    --latent-dim 16384 \
    --topk-k 512 \
    --fasta data/proteins.fasta \
    --proteins data/proteins_with_split.tsv \
    --num-epochs 20 \
    --early-stop-patience 10 \
    --outdir outputs/sae_variants/topk_16384_k512
```

Supported architectures: `vanilla`, `topk`, `matryoshka`, `tsae`.

To train all 4 variants at once:

```bash
bash src/stream/run_all_variants.sh
# or specific archs: bash src/stream/run_all_variants.sh vanilla topk
```

**Outputs:** `best.pt` (checkpoint), `config.json`, `training_log.csv`.

Key hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `--latent-dim` | 8192 | SAE dictionary size |
| `--topk-k` | 128 | Active features per token (TopK) |
| `--l1-lambda` | 3e-5 | L1 penalty (Vanilla/T-SAE) |
| `--target-l0` | 0 | Adaptive L1 target (0 = disabled) |
| `--esm2-layer` | 33 | ESM-2 layer for embeddings |

---

### Phase 3 — Feature–Annotation Alignment

Two-pass streaming alignment on GPU. Pass 1 collects per-feature activation
statistics; Pass 2 computes per-protein AUPRC, odds ratio, and enrichment
for every (feature, annotation) pair.

#### 3.1 Run alignment on test split

```bash
python src/stream/feature_alignment_streaming.py \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --fasta data/proteins.fasta \
    --split test \
    --all-features \
    --outdir outputs/feature_alignment_test
```

#### 3.2 (Recommended) Unbiased val→test evaluation

Run alignment separately on val and test, then use val for feature
selection and test for frozen evaluation:

```bash
# Val run
python src/stream/feature_alignment_streaming.py \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --fasta data/proteins.fasta \
    --split val \
    --all-features \
    --outdir outputs/feature_alignment_val

# Test run
python src/stream/feature_alignment_streaming.py \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --fasta data/proteins.fasta \
    --split test \
    --all-features \
    --outdir outputs/feature_alignment_test

# Unbiased selection: picks best feature per annotation on val,
# reports frozen test AUPRC
python src/stream/val_test_selection.py \
    --val-scores  outputs/feature_alignment_val/alignment_scores.parquet \
    --test-scores outputs/feature_alignment_test/alignment_scores.parquet \
    --outdir      outputs/feature_alignment_unbiased
```

**Alignment outputs:**

| File | Description |
|---|---|
| `alignment_scores.parquet` | All (feature, annot_type) AUPRC/OR pairs |
| `per_annot_summary.tsv` | Best feature per annotation (test-selected) |
| `per_annot_summary_robust.tsv` | Filtered: n≥100, p<0.01, clusters≥30 |
| `feature_triage_summary.tsv` | Per-feature best annotation |
| `novel_candidates.tsv` | Features with low best AUPRC (novel candidates) |

**Unbiased selection outputs:**

| File | Description |
|---|---|
| `per_annot_val_test.tsv` | Val-selected, test-evaluated per annotation |
| `feature_triage_val_test.tsv` | Per-feature summary (val-selected) |
| `selection_summary.json` | Bias estimate: val vs test AUPRC gap |

#### 3.3 Multi-GPU sharding

For large runs, shard across GPUs and merge:

```bash
# GPU 0
python src/stream/feature_alignment_streaming.py \
    --checkpoint best.pt --fasta proteins.fasta \
    --num-shards 2 --shard-idx 0 --outdir outputs/alignment

# GPU 1
python src/stream/feature_alignment_streaming.py \
    --checkpoint best.pt --fasta proteins.fasta \
    --num-shards 2 --shard-idx 1 --outdir outputs/alignment

# Merge
python src/stream/feature_alignment_streaming.py \
    --merge-shards --num-shards 2 --outdir outputs/alignment
```

---

### Phase 4 — Verification & Figures

All scripts under `src/verification/`. Each requires the SAE checkpoint
and alignment outputs from Phase 3.

#### 4.1 Attention baseline

Extract ESM-2 attention profiles and compare with SAE alignment:

```bash
python src/verification/extract_attention.py \
    --shard-dir outputs/embeddings/esm2_650m \
    --alignment outputs/feature_alignment_test/per_annot_summary.tsv \
    --sae-scores outputs/feature_alignment_test/alignment_scores.parquet \
    --outdir outputs/attention
```

Outputs: `convergence_with_sae.tsv`, attention `alignment_scores.parquet`.

#### 4.2 Multi-evidence analysis

Three evidence layers per (feature, annotation) pair: physicochemical
properties (hydrophobicity, charge, volume inside vs outside), causal
ablation (zero feature → measure embedding delta), and cross-feature
convergence.

```bash
python src/verification/multi_evidence.py \
    --alignment outputs/feature_alignment_test/per_annot_summary.tsv \
    --all-scores outputs/feature_alignment_test/alignment_scores.parquet \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --outdir outputs/multi_evidence
```

Outputs: per-pair directories with `physicochemical.tsv`, `ablation.tsv`,
`convergence.tsv`, plus `full_evidence_matrix.tsv`.

#### 4.3 Qualitative feature analysis

Positional profiles, activation distributions, per-protein heatmaps:

```bash
python src/verification/feature_qualitative.py \
    --alignment outputs/feature_alignment_test/per_annot_summary.tsv \
    --all-scores outputs/feature_alignment_test/alignment_scores.parquet \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --outdir outputs/feature_qualitative
```

#### 4.4 Novel feature triage

Identify features with high causal signal but no known annotation match:

```bash
python src/verification/triage_novel_features.py \
    --alignment-dir outputs/feature_alignment_test \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --run-ablation \
    --outdir outputs/novel_features
```

#### 4.5 Paper figures

Generate all publication figures (heatmap, multi-evidence bars, convergence
scatter, structural features, layer-wise attention, protein case studies):

```bash
python src/verification/paper_figures.py \
    --alignment-dir outputs/feature_alignment_test \
    --multi-ev-dir outputs/multi_evidence \
    --attention-dir outputs/attention \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --outdir outputs/paper_figures
```

Outputs: `fig1_alignment_heatmap.png`, `fig2_multi_evidence.png`,
`fig3_convergence.png`, `fig4_structural.png`, `fig5_layerwise.png`,
`fig6_*.png` (case studies).

#### 4.6 Enriched annotation figures

Bar charts, scatter plots, and heatmaps for enriched annotation subtypes:

```bash
python src/verification/enriched_figures.py \
    --alignment-dir outputs/feature_alignment_test \
    --outdir outputs/paper_figures_enriched
```

---

## Data Files Reference

Processed dataset files are available on Zenodo: [10.5281/zenodo.20379180](https://doi.org/10.5281/zenodo.20379180)
(includes `clusters.tsv`, `protein_splits.tsv`, `proteins_with_split.tsv`, `annotations_enriched.tsv`)

Raw Swiss-Prot files (`proteins.tsv`, `annotations.gff`, `swissprot_features_raw.tsv`) should be downloaded directly from UniProt.

| File | Description |
|---|---|
| `data/proteins.tsv` | Swiss-Prot proteins (accession, sequence, length) — download from UniProt |
| `data/proteins.fasta` | Same proteins in FASTA format |
| `data/annotations.gff` | Raw UniProt GFF annotations — download from UniProt |
| `data/annotations_enriched.tsv` | Enriched annotations with `annot_subtype`, `split`, `cluster_id` — Zenodo |
| `data/clusters.tsv` | MMseqs2 homology clusters — Zenodo |
| `data/protein_splits.tsv` | Cluster-level train/val/test assignment — Zenodo |
| `data/proteins_with_split.tsv` | Proteins with split and cluster_id columns — Zenodo |
| `data/annotations_dedup_with_split.tsv` | Deduplicated annotations with split |

---

## Quick Start

```bash
# 1. Prepare data (assumes data/*.tsv and data/*.gff already exist)
python src/dsPrep/gff_to_tsv.py
python src/dsPrep/tsv_to_fasta.py
python src/dsPrep/split_clusters.py 0.70 0.15
python src/dsPrep/add_split_info.py
python src/dsPrep/remove_annotation_duplicates.py
python src/dsPrep/download_swissprot_features.py --parse-only

# 2. Train SAE (TopK, k=512, 16384 latents)
python src/stream/train_sae_variants.py \
    --architecture topk --latent-dim 16384 --topk-k 512 \
    --fasta data/proteins.fasta --proteins data/proteins_with_split.tsv \
    --outdir outputs/sae_variants/topk_16384_k512

# 3. Align features (test split, all 16384 features)
python src/stream/feature_alignment_streaming.py \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --fasta data/proteins.fasta --split test --all-features \
    --outdir outputs/feature_alignment_test

# 4. Multi-evidence verification
python src/verification/multi_evidence.py \
    --alignment outputs/feature_alignment_test/per_annot_summary.tsv \
    --all-scores outputs/feature_alignment_test/alignment_scores.parquet \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --outdir outputs/multi_evidence

# 5. Paper figures
python src/verification/paper_figures.py \
    --alignment-dir outputs/feature_alignment_test \
    --multi-ev-dir outputs/multi_evidence \
    --checkpoint outputs/sae_variants/topk_16384_k512/best.pt \
    --outdir outputs/paper_figures
```