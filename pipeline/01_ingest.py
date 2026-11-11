"""Ingest IEDB and CEDAR from raw JSON, applying identical MHC-I filters to both."""

import re
import sys
import json
from pathlib import Path

import ijson
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import IEDB_JSON, CEDAR_JSON, DATA_DIR, MIN_LENGTH, MAX_LENGTH

DATA_DIR.mkdir(parents=True, exist_ok=True)

VALID_AA        = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
POSITIVE_LABELS = {"Positive", "Positive-High", "Positive-Intermediate", "Positive-Low"}
NEGATIVE_LABELS = {"Negative"}
HLA_PATTERN     = re.compile(r'^HLA-[ABC]\*\d{2}:\d{2,3}[NLSQACG]?$')

CANCER_VIRUSES = {
    "human gammaherpesvirus 4",
    "human papillomavirus 16", "human papillomavirus 18",
    "hepatitis b virus", "hepacivirus hominis",
    "human t-lymphotropic virus 1", "merkel cell polyomavirus",
    "human betaherpesvirus 5",
}


def source_category(organism: str) -> str:
    if not organism:
        return "Unknown"
    low = organism.lower()
    if low == "homo sapiens":
        return "TAA"
    if any(cv in low for cv in CANCER_VIRUSES):
        return "Cancer-viral"
    return "Other-viral"


def parse_allele(raw: str) -> str | None:
    allele = raw.strip()
    return allele if HLA_PATTERN.match(allele) else None


def ingest_iedb(path: Path) -> pd.DataFrame:
    print(f"Streaming IEDB: {path} ...")
    rows = []
    skipped = {k: 0 for k in
               ("not_linear", "invalid_seq", "length_oob", "not_mhc1",
                "non_human_host", "bad_allele", "ambiguous_outcome")}
    n_raw = 0

    with open(path, "rb") as f:
        for r in ijson.items(f, "Data.item"):
            n_raw += 1
            if n_raw % 100_000 == 0:
                print(f"  ... {n_raw:,} scanned, {len(rows):,} kept")

            if r.get("Epitope - Object Type", "").strip() != "Linear peptide":
                skipped["not_linear"] += 1; continue

            seq = r.get("Epitope - Name", "").strip().upper()
            if not seq or not VALID_AA.match(seq):
                skipped["invalid_seq"] += 1; continue
            if not (MIN_LENGTH <= len(seq) <= MAX_LENGTH):
                skipped["length_oob"] += 1; continue

            if r.get("MHC Restriction - Class", "").strip() != "I":
                skipped["not_mhc1"] += 1; continue

            host = r.get("Host - Name", "").strip()
            if not (host.startswith("Homo sapiens") or "HLA" in host):
                skipped["non_human_host"] += 1; continue

            allele = parse_allele(r.get("MHC Restriction - Name", ""))
            if allele is None:
                skipped["bad_allele"] += 1; continue

            qual = r.get("Assay - Qualitative Measurement", "").strip()
            if qual in POSITIVE_LABELS:
                outcome = "Positive"
            elif qual in NEGATIVE_LABELS:
                outcome = "Negative"
            else:
                skipped["ambiguous_outcome"] += 1; continue

            organism = r.get("Epitope - Species", "").strip()
            rows.append({
                "epitope":          seq,
                "hla_allele":       allele,
                "outcome":          outcome,
                "source_organism":  organism,
                "source_category":  source_category(organism),
            })

    df = pd.DataFrame(rows)
    _report("IEDB", n_raw, skipped, df)
    return df


def ingest_cedar(path: Path) -> pd.DataFrame:
    print(f"Loading CEDAR: {path} ...")
    with open(path) as f:
        data = json.load(f)
    records = data["Data"]
    print(f"  Total records: {len(records):,}")

    rows = []
    skipped = {k: 0 for k in
               ("not_linear", "invalid_seq", "length_oob", "not_mhc1",
                "non_human_host", "bad_allele", "ambiguous_outcome")}

    for r in records:
        if r.get("Epitope - Object Type", "").strip() != "Linear peptide":
            skipped["not_linear"] += 1; continue

        seq = r.get("Epitope - Name", "").strip().upper()
        if not seq or not VALID_AA.match(seq):
            skipped["invalid_seq"] += 1; continue
        if not (MIN_LENGTH <= len(seq) <= MAX_LENGTH):
            skipped["length_oob"] += 1; continue

        if r.get("MHC Restriction - Class", "").strip() != "I":
            skipped["not_mhc1"] += 1; continue

        host = r.get("Host - Name", "").strip()
        if not host.startswith("Homo sapiens"):
            skipped["non_human_host"] += 1; continue

        allele = parse_allele(r.get("MHC Restriction - Name", ""))
        if allele is None:
            skipped["bad_allele"] += 1; continue

        qual = r.get("Assay - Qualitative Measurement", "").strip()
        if qual in POSITIVE_LABELS:
            outcome = "Positive"
        elif qual in NEGATIVE_LABELS:
            outcome = "Negative"
        else:
            skipped["ambiguous_outcome"] += 1; continue

        organism = r.get("Epitope - Source Organism", "").strip()
        rows.append({
            "epitope":          seq,
            "hla_allele":       allele,
            "outcome":          outcome,
            "source_organism":  organism,
            "source_category":  source_category(organism),
        })

    df = pd.DataFrame(rows)
    _report("CEDAR", len(records), skipped, df)
    return df


def _report(name: str, n_raw: int, skipped: dict, df: pd.DataFrame) -> None:
    print(f"\n{name}: {n_raw:,} raw → {len(df):,} kept")
    for reason, n in skipped.items():
        if n:
            print(f"  skipped {reason}: {n:,}")
    print(f"  unique epitopes : {df['epitope'].nunique():,}")
    print(f"  unique alleles  : {df['hla_allele'].nunique():,}")
    print(f"  outcome         : {df['outcome'].value_counts().to_dict()}")


def main():
    iedb  = ingest_iedb(IEDB_JSON)
    cedar = ingest_cedar(CEDAR_JSON)

    out_iedb  = DATA_DIR / "iedb_raw.tsv"
    out_cedar = DATA_DIR / "cedar_raw.tsv"
    iedb.to_csv(out_iedb,   sep="\t", index=False)
    cedar.to_csv(out_cedar, sep="\t", index=False)
    print(f"\nSaved:\n  {out_iedb}\n  {out_cedar}")


if __name__ == "__main__":
    main()
