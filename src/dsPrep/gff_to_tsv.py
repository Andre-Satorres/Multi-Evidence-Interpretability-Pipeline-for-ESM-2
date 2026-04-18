import pandas as pd
import constants

records = []

with open(constants.ANNOTATIONS_GFF_PATH) as f:
    for line in f:
        if line.startswith("#"):
            continue
        
        parts = line.strip().split("\t")
        if len(parts) < 9:
            continue
        
        accession = parts[0]
        feature_type = parts[2]
        start = int(parts[3])
        end = int(parts[4])
        attributes = parts[8]
        
        # extrair descrição
        desc = ""
        for attr in attributes.split(";"):
            if attr.startswith("Note="):
                desc = attr.replace("Note=", "")
        
        records.append([
            accession,
            feature_type,
            start,
            end,
            desc
        ])

df = pd.DataFrame(records, columns=[
    "accession", "feature_type", "start", "end", "description"
])

df.to_csv(constants.ANNOTATIONS_TSV_PATH, sep="\t", index=False)