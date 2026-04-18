"""
download_swissprot_features.py — PIPLV2 / dsPrep
==================================================
Download ALL Swiss-Prot features in a single streaming request and parse
into the same annotation format used by the rest of the pipeline.

Instead of making 500k individual API requests (44+ hours), this downloads
the complete Swiss-Prot feature table in one TSV request (~5-10 minutes for
~570k proteins, ~700MB).

The UniProt TSV format for features looks like:
  ft_act_site:  "ACT_SITE 100; /note="Proton acceptor"; ACT_SITE 200; ..."
  ft_binding:   "BINDING 10..20; /ligand="ATP"; BINDING 50; /ligand="Mg(2+)"; ..."
  ft_mod_res:   "MOD_RES 15; /note="Phosphoserine"; ..."

This script parses all feature fields into rows with:
  accession, feature_type, start, end, description, annot_subtype

Then filters to only proteins in proteins_with_split.tsv and adds
split/cluster_id columns.

Usage
-----
  # Step 1: download (run once, ~5-10 min, ~700MB)
  python src/dsPrep/download_swissprot_features.py --download-only

  # Step 2: parse into annotation TSV (fast, local)
  python src/dsPrep/download_swissprot_features.py --parse-only

  # Both steps:
  python src/dsPrep/download_swissprot_features.py
"""

import re
import sys
import logging
import argparse
import gzip
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import DATA_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# =============================================================================
#  UNIPROT FIELD → FEATURE TYPE MAPPING
# =============================================================================

# UniProt TSV field name → canonical feature type
FIELD_TO_TYPE = {
    "ft_act_site":    "Active site",
    "ft_binding":     "Binding site",
    # ft_ca_bind removed — not a valid UniProt field; calcium binding covered by ft_binding
    "ft_dna_bind":    "DNA binding",
    "ft_site":        "Site",
    "ft_mod_res":     "Modified residue",
    "ft_carbohyd":    "Glycosylation",  # correct field name for glycosylation
    "ft_lipid":       "Lipidation",
    "ft_crosslnk":    "Cross-link",
    "ft_disulfid":    "Disulfide bond",
    "ft_helix":       "Helix",
    "ft_strand":      "Beta strand",
    "ft_turn":        "Turn",
    "ft_coiled":      "Coiled coil",
    "ft_region":      "Region",
    "ft_motif":       "Motif",
    "ft_repeat":      "Repeat",
    "ft_compbias":    "Compositional bias",
    "ft_intramem":    "Intramembrane",
    "ft_transmem":    "Transmembrane",
    "ft_topo_dom":    "Topological domain",
    "ft_signal":      "Signal peptide",
    "ft_transit":     "Transit peptide",
    "ft_propep":      "Propeptide",
    "ft_peptide":     "Peptide",
    "ft_chain":       "Chain",
    "ft_init_met":    "Initiator methionine",
    "ft_domain":      "Domain",
    "ft_zn_fing":     "Zinc finger",
    "ft_variant":     "Natural variant",
    "ft_var_seq":     "Alternative sequence",
    "ft_mutagen":     "Mutagenesis",
    "ft_conflict":    "Sequence conflict",
    "ft_unsure":      "Sequence uncertainty",
    "ft_non_cons":    "Non-adjacent residues",
    "ft_non_ter":     "Non-terminal residue",
}

# Fields to exclude from the final annotation file
SKIP_TYPES = {"Chain", "Initiator methionine"}

# All feature fields to request from UniProt
ALL_FIELDS = list(FIELD_TO_TYPE.keys())


# =============================================================================
#  DOWNLOAD
# =============================================================================

def build_download_url(cursor: str = None, page_size: int = 500) -> str:
    """Build UniProt search URL for paginated download of all Swiss-Prot features."""
    fields = "accession," + ",".join(ALL_FIELDS)
    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        "?format=tsv"
        "&compressed=true"
        "&query=reviewed:true"
        f"&size={page_size}"
        f"&fields={fields}"
    )
    if cursor:
        url += f"&cursor={cursor}"
    return url

def _build_download_url_old() -> str:  # kept for reference
    """Build UniProt streaming URL for all Swiss-Prot features."""
    fields = "accession," + ",".join(ALL_FIELDS)
    # reviewed:true = Swiss-Prot only
    # Note: stream endpoint does not accept &size= parameter
    # It streams the complete Swiss-Prot (~570k proteins)
    url = (
        "https://rest.uniprot.org/uniprotkb/stream"
        "?format=tsv"
        "&compressed=false"
        "&query=(reviewed:true)"
        f"&fields={fields}"
    )
    return url


def download_swissprot(out_path: Path, page_size: int = 500) -> None:
    """
    Download Swiss-Prot TSV via paginated search API.
    ~570k proteins in ~500-entry pages. Handles cursor pagination via Link header.
    """
    import re as _re
    log.info("Downloading Swiss-Prot features (paginated search) ...")
    log.info(f"  Output: {out_path}")
    log.info(f"  Expected: ~570k proteins, ~10-20 min")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cursor = None
    page   = 0
    total_proteins = 0
    header_written = False

    with open(out_path, 'w', encoding='utf-8') as f:
        while True:
            url = build_download_url(cursor=cursor, page_size=page_size)
            try:
                req = urllib.request.Request(
                    url, headers={"Accept": "text/plain"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    raw = response.read()
                    # decompress only if actually gzip-encoded
                    encoding = response.headers.get("Content-Encoding", "")
                    if encoding == "gzip" or (raw[:2] == b'\x1f\x8b'):
                        try:
                            body = gzip.decompress(raw).decode('utf-8')
                        except (OSError, Exception):
                            body = raw.decode('utf-8')
                    else:
                        body = raw.decode('utf-8')
                    link_header = response.headers.get("Link", "")
                    next_cursor = None
                    if link_header:
                        m = _re.search(r'cursor=([^&> "]+).*?rel="next"',
                                       link_header)
                        if m:
                            next_cursor = m.group(1)
            except urllib.error.URLError as e:
                log.error(f"  Page {page} failed: {e}")
                raise

            lines = body.rstrip("\n").split("\n")

            if not header_written:
                f.write(lines[0] + "\n")
                header_written = True
                data_lines = lines[1:]
            else:
                # skip repeated header line on subsequent pages
                data_lines = lines[1:] if (lines and
                    lines[0].startswith("Entry")) else lines

            for line in data_lines:
                if line.strip():
                    f.write(line + "\n")
                    total_proteins += 1

            page += 1
            if page % 100 == 0 or page <= 2:
                log.info(f"  Page {page}: {total_proteins:,} proteins, "
                         f"cursor={'found' if next_cursor else 'NONE (last page)'}")

            if not next_cursor:
                log.info(f"  No more pages after page {page}. Done.")
                break
            cursor = next_cursor

    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info(f"  Done: {page} pages, {total_proteins:,} proteins, "
             f"{size_mb:.0f} MB → {out_path.name}")


def _parse_position(pos_str: str) -> tuple:
    """Parse "100" → (100, 100) or "10..20" → (10, 20)."""
    if '..' in pos_str:
        parts = pos_str.split('..')
        return int(parts[0]), int(parts[1])
    else:
        v = int(pos_str)
        return v, v


def parse_feature_field(acc: str, field_name: str,
                         field_value: str) -> list[dict]:
    """
    Parse one feature field value into a list of annotation rows.

    field_value example:
      "100; /note="Proton acceptor"; /evidence="ECO:0000255""
      "10..20; /ligand="ATP"; 50; /ligand="Mg(2+)";"
    """
    if not field_value or field_value.strip() in ("", "-"):
        return []

    ftype = FIELD_TO_TYPE.get(field_name, field_name)
    if ftype in SKIP_TYPES:
        return []

    rows = []

    # Split multiple entries within the same field.
    # Each entry starts with a position (digits), possibly preceded by
    # a feature-type keyword that we can ignore (it's redundant with the column).
    # Strategy: split on semicolons, reconstruct entries.
    raw = field_value.strip()

    # UniProt stream format includes the feature type keyword before each entry
    # e.g. "BINDING 10..20; /ligand=...; BINDING 50; /ligand=..."
    # Split on the keyword boundaries first
    # Pattern: split before uppercase keyword + space + digit
    entries_raw = re.split(r'\s+(?=[A-Z_]{2,}\s+\d)', raw)

    # Also handle the case where there's no keyword (already stripped)
    entries = []
    for e in entries_raw:
        # Remove leading keyword if present
        e_clean = re.sub(r'^[A-Z_]+\s+', '', e.strip())
        if e_clean:
            entries.append(e_clean)

    if not entries:
        entries = re.split(r'(?<=;)\s+(?=\d)', raw)

    for entry in entries:
        entry = entry.strip().rstrip(';')
        if not entry:
            continue

        # extract position
        pos_match = re.match(r'(\d+(?:\.\.\d+)?)', entry)
        if not pos_match:
            continue
        try:
            start, end = _parse_position(pos_match.group(1))
        except ValueError:
            continue

        if start <= 0 or end <= 0 or end < start:
            continue

        # extract description: combine /ligand + /note for binding sites,
        # use /note alone for others, fallback to /description
        desc = ""
        lig_m  = re.search(r'/ligand="([^"]*)"', entry)
        note_m = re.search(r'/note="([^"]*)"', entry)
        desc_m = re.search(r'/description="([^"]*)"', entry)

        if lig_m and note_m:
            # e.g. Binding site: "heme b; distal binding residue"
            desc = f"{lig_m.group(1).strip()}; {note_m.group(1).strip()}"
        elif lig_m:
            desc = lig_m.group(1).strip()
        elif note_m:
            desc = note_m.group(1).strip()
        elif desc_m:
            desc = desc_m.group(1).strip()

        # clean description
        desc = re.sub(r'\[ECO:[^\]]+\]', '', desc).strip()
        desc = re.sub(r'^(By similarity|Probable|Potential)[;,]?\s*', '',
                      desc, flags=re.IGNORECASE).strip()
        # truncate very long descriptions
        if len(desc) > 120:
            desc = desc[:120]

        # For some feature types, the description is mutation/variant-specific
        # (e.g. "E->A: Lack of activity") — not a generalizable biological concept.
        # For these, use the generic feature_type as subtype (ignore description).
        NO_SUBTYPE_TYPES = {
            "Mutagenesis", "Natural variant", "Alternative sequence",
            "Sequence conflict", "Sequence uncertainty", "Cross-link",
            "Disulfide bond",  # position-specific, not a named subtype
        }
        if ftype in NO_SUBTYPE_TYPES:
            annot_subtype = ftype
            desc = ""  # clear desc so it doesn't mislead
        elif desc:
            annot_subtype = f"{ftype}: {desc}"
        else:
            annot_subtype = ftype

        rows.append({
            "accession":     acc,
            "feature_type":  ftype,
            "start":         start,
            "end":           end,
            "description":   desc,
            "annot_subtype": annot_subtype,
        })

    return rows


def parse_swissprot_tsv(tsv_path: Path,
                         proteins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the downloaded Swiss-Prot TSV into annotation rows.
    Filters to only proteins present in proteins_df.
    """
    log.info(f"Parsing {tsv_path.name} ...")

    acc_set    = set(proteins_df["accession"].tolist())
    split_map  = dict(zip(proteins_df["accession"], proteins_df["split"]))
    cluster_map = dict(zip(proteins_df["accession"],
                           proteins_df.get("cluster_id",
                                           pd.Series(dtype=str))))

    all_rows = []
    n_proteins_found = 0
    n_proteins_total = 0

    # read header to get column indices
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().rstrip('\n')
        col_names = [c.strip().lower() for c in header_line.split('\t')]

        # UniProt TSV uses "Entry" as the accession column name
        if "entry" in col_names:
            acc_idx = col_names.index("entry")
        elif "accession" in col_names:
            acc_idx = col_names.index("accession")
        else:
            acc_idx = 0
            log.warning(f"Accession column not found, using column 0. "
                        f"Headers: {col_names[:5]}")

        # map field names to column indices
        # UniProt TSV uses display names like "Active site", "Binding site", etc.
        # Build reverse map: display_name_lower → field_name
        display_to_field = {}
        for field, ftype in FIELD_TO_TYPE.items():
            display_to_field[ftype.lower()] = field
            display_to_field[field.lower()] = field
            display_to_field[field.replace("ft_", "").lower()] = field

        field_col_idx = {}
        for col_i, col in enumerate(col_names):
            col_clean = col.strip().lower().rstrip(" [ft]").strip()
            if col_clean in display_to_field:
                field = display_to_field[col_clean]
                if field not in field_col_idx:  # first match wins
                    field_col_idx[field] = col_i

        log.info(f"  Mapped {len(field_col_idx)} columns: "
                 f"{sorted(field_col_idx.keys())[:5]}...")

        log.info(f"  Columns found: {len(field_col_idx)} feature fields "
                 f"out of {len(ALL_FIELDS)} requested")
        if not field_col_idx:
            log.error("  No feature columns found! Check column names:")
            log.error(f"  Header: {col_names[:10]}")
            return pd.DataFrame()

        # parse line by line
        for line_i, line in enumerate(f):
            n_proteins_total += 1
            if n_proteins_total % 50000 == 0:
                log.info(f"  {n_proteins_total:,} proteins processed, "
                         f"{n_proteins_found:,} matched, "
                         f"{len(all_rows):,} annotations ...")

            parts = line.rstrip('\n').split('\t')
            if len(parts) <= acc_idx:
                continue

            acc = parts[acc_idx].strip()
            if acc not in acc_set:
                continue

            n_proteins_found += 1

            for field, col_idx in field_col_idx.items():
                if col_idx >= len(parts):
                    continue
                value = parts[col_idx].strip()
                rows = parse_feature_field(acc, field, value)
                all_rows.extend(rows)

    log.info(f"  Parsed {n_proteins_total:,} proteins total, "
             f"{n_proteins_found:,} matched to dataset, "
             f"{len(all_rows):,} annotation rows")

    if not all_rows:
        log.warning("No annotations found! Check accession format.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["split"]      = df["accession"].map(split_map).fillna("unknown")
    df["cluster_id"] = df["accession"].map(cluster_map).fillna("")

    # deduplicate
    before = len(df)
    df = df.drop_duplicates(
        subset=["accession", "feature_type", "start", "end", "description"])
    log.info(f"  After dedup: {len(df):,} rows (removed {before-len(df):,} dups)")

    return df


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--proteins",      type=Path,
                   default=DATA_DIR / "proteins_with_split.tsv")
    p.add_argument("--raw-tsv",       type=Path,
                   default=DATA_DIR / "swissprot_features_raw.tsv",
                   help="Where to save/load the raw Swiss-Prot download.")
    p.add_argument("--outfile",       type=Path,
                   default=DATA_DIR / "annotations_enriched.tsv")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--parse-only",    action="store_true",
                   help="Skip download, parse existing --raw-tsv.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── download ───────────────────────────────────────────────────────────────
    if not args.parse_only:
        if args.raw_tsv.exists():
            size_mb = args.raw_tsv.stat().st_size / 1024 / 1024
            log.info(f"Raw TSV already exists ({size_mb:.0f} MB): {args.raw_tsv.name}")
            log.info("  Use --parse-only to re-parse, or delete to re-download.")
        else:
            download_swissprot(args.raw_tsv)

    if args.download_only:
        log.info("Download complete. Run with --parse-only to parse.")
        return

    # ── parse ──────────────────────────────────────────────────────────────────
    if not args.raw_tsv.exists():
        log.error(f"Raw TSV not found: {args.raw_tsv}")
        log.error("Run without --parse-only first to download.")
        sys.exit(1)

    log.info("Loading protein list ...")
    prot_df = pd.read_csv(args.proteins, sep="\t", low_memory=False)
    prot_df.columns = prot_df.columns.str.strip().str.lower()
    log.info(f"  {len(prot_df):,} proteins")

    ann_df = parse_swissprot_tsv(args.raw_tsv, prot_df)

    if ann_df.empty:
        log.error("Parsing produced no results.")
        sys.exit(1)

    # ── save ───────────────────────────────────────────────────────────────────
    ann_df.to_csv(args.outfile, sep="\t", index=False)
    log.info(f"\nSaved: {args.outfile.name}")

    # summary
    n_types    = ann_df["feature_type"].nunique()
    n_subtypes = ann_df["annot_subtype"].nunique()
    log.info(f"  {len(ann_df):,} annotations")
    log.info(f"  {n_types} feature types")
    log.info(f"  {n_subtypes} unique subtypes (annot_subtype)")

    print("\nTop subtypes per feature type:")
    for ftype in sorted(ann_df["feature_type"].unique()):
        sub = ann_df[(ann_df["feature_type"] == ftype) &
                     (ann_df["description"] != "")]
        if sub.empty:
            continue
        top = sub["annot_subtype"].value_counts().head(3)
        print(f"\n  {ftype} ({ann_df[ann_df['feature_type']==ftype]['accession'].nunique():,} proteins):")
        for subtype, cnt in top.items():
            print(f"    {cnt:>6,}x  {subtype}")

    log.info(f"\n✅ Done. Use {args.outfile.name} as --annotations in the pipeline.")


if __name__ == "__main__":
    main()