import sys
from pathlib import Path
from split_analysis import load_data

from dsPrep.split_analysis import _acc_col

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

prot, ann = load_data()

dup_cols = ["accession", "feature_type", "start", "end"]
dups = ann.duplicated(subset=dup_cols, keep=False)

print(ann[dups].sort_values(dup_cols).head(50))

print(ann.groupby(
    ["accession", "feature_type", "start", "end"]
).size().sort_values(ascending=False).head(20))
