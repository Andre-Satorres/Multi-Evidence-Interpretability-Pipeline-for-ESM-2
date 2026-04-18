import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import PROTEINS_TSV_PATH, PROTEINS_FASTA_PATH

df = pd.read_csv(PROTEINS_TSV_PATH, sep="\t")

with open(PROTEINS_FASTA_PATH, "w") as f:
    for _, row in df.iterrows():
        entry = row["Entry"]
        seq = row["Sequence"]
        if isinstance(seq, str) and seq:
            f.write(f">{entry}\n{seq}\n")