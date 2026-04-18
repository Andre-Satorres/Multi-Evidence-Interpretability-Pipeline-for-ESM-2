from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

ANNOTATIONS_GFF_PATH = ROOT / "data" / "annotations.gff"
ANNOTATIONS_TSV_PATH = DATA_DIR / "annotations.tsv"
ANNOTATIONS_DEDUP_TSV_PATH = DATA_DIR / "annotations_dedup.tsv"
PROTEINS_TSV_PATH    = DATA_DIR / "proteins.tsv"

PROTEINS_FASTA_PATH = DATA_DIR / "proteins.fasta"

CLUSTERS_TSV_PATH = DATA_DIR / "clusters.tsv"
SPLITS_TSV_PATH = DATA_DIR / "protein_splits.tsv"
PROTEINS_WITH_SPLIT_TSV_PATH = DATA_DIR / "proteins_with_split.tsv"
ANNOTATIONS_WITH_SPLIT_TSV_PATH = DATA_DIR / "annotations_with_split.tsv"
ANNOTATIONS_DEDUP_WITH_SPLIT_TSV_PATH = DATA_DIR / "annotations_dedup_with_split.tsv"

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATASET_ANALYSIS_DIR = OUT_DIR / "raw_dataset_analysis"
RAW_DATASET_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_ANALYSIS_DIR = OUT_DIR / "split_analysis"
SPLIT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SUBSET_OUT_DIR = OUT_DIR / "subsets"
SUBSET_OUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)