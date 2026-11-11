"""Run NetMHCpan on all (epitope, allele) pairs; saves rank_el and score_el."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import NETMHCPAN, DATA_DIR

CHECKPOINT_EVERY = 20


def run_netmhcpan(peptides: list[str], allele: str, tmp_dir: str) -> pd.DataFrame:
    pept_file = os.path.join(tmp_dir, "peptides.txt")
    out_file  = os.path.join(tmp_dir, "output.txt")

    with open(pept_file, "w") as fh:
        for p in peptides:
            fh.write(p + "\n")

    allele_fmt = allele.replace("*", "")
    cmd = [
        str(NETMHCPAN),
        "-a", allele_fmt,
        "-f", pept_file,
        "-inptype", "1",
        "-BA",
        "-xls", "-xlsfile", out_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_file):
        print(f"    WARNING: NetMHCpan failed for {allele}: {result.stderr[:200]}")
        return pd.DataFrame()

    rows = []
    with open(out_file) as fh:
        lines = fh.readlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("Pos")), None)
    if header_idx is None:
        return pd.DataFrame()
    headers = lines[header_idx].strip().split("\t")
    for line in lines[header_idx + 1:]:
        parts = line.strip().split("\t")
        if len(parts) < len(headers):
            continue
        rows.append(dict(zip(headers, parts)))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    col_lower = {c.lower(): c for c in df.columns}

    pept_col  = col_lower.get("peptide")
    core_col  = col_lower.get("core") or col_lower.get("mhcii_core")
    rank_col  = (col_lower.get("el_rank") or col_lower.get("%rank_el")
                 or col_lower.get("rank_el") or col_lower.get("%rank"))
    score_col = (col_lower.get("el_score") or col_lower.get("score_el")
                 or col_lower.get("score") or col_lower.get("1-log50k(aff)"))

    if not all([pept_col, core_col, rank_col]):
        print(f"    WARNING: missing columns for {allele}: {list(df.columns)}")
        return pd.DataFrame()

    out = df[[pept_col, core_col, rank_col]].copy()
    out.columns = ["peptide", "binding_core", "rank_el"]
    out["score_el"] = pd.to_numeric(df[score_col], errors="coerce") if score_col else float("nan")
    out["hla_allele"] = allele
    out["rank_el"] = pd.to_numeric(out["rank_el"], errors="coerce")
    return out


def process_dataset(name: str, raw_path: Path, out_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Processing {name.upper()}: {raw_path}")
    raw = pd.read_csv(raw_path, sep="\t")
    pairs = raw[["epitope", "hla_allele"]].drop_duplicates()
    print(f"  {len(raw):,} records  |  {len(pairs):,} unique (epitope, allele) pairs")

    scored_alleles: set[str] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, sep="\t")
        scored_alleles = set(existing["hla_allele"].unique())
        print(f"  Resuming: {len(scored_alleles)} alleles already scored")

    by_allele: dict[str, list[str]] = {}
    for _, row in pairs.iterrows():
        if row["hla_allele"] not in scored_alleles:
            by_allele.setdefault(row["hla_allele"], []).append(row["epitope"])

    if not by_allele:
        print("  All alleles already scored — nothing to do.")
        return

    results: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory() as tmp:
        alleles = sorted(by_allele.keys())
        for i, allele in enumerate(alleles):
            peptides = list(set(by_allele[allele]))
            print(f"  [{i+1}/{len(alleles)}] {allele}: {len(peptides):,} peptides")
            df = run_netmhcpan(peptides, allele, tmp)
            if not df.empty:
                results.append(df)

            if results and (i + 1) % CHECKPOINT_EVERY == 0:
                _append_results(results, out_path, scored_alleles)
                results = []

    if results:
        _append_results(results, out_path, scored_alleles)

    if out_path.exists():
        final = pd.read_csv(out_path, sep="\t")
        print(f"\nDone. {len(final):,} rows → {out_path}")
        print(f"  {final['epitope'].nunique():,} unique epitopes")
        print(f"  score_el coverage: {final['score_el'].notna().mean():.1%}")
    else:
        print(f"\nDone. No output written (all alleles may have failed).")


def _append_results(results: list[pd.DataFrame], out_path: Path,
                    scored_alleles: set[str]) -> None:
    chunk = pd.concat(results, ignore_index=True)
    chunk = chunk.rename(columns={"peptide": "epitope"})
    chunk = chunk.drop_duplicates(subset=["epitope", "hla_allele"])
    file_exists = out_path.exists()
    chunk.to_csv(out_path, sep="\t", mode="a", header=not file_exists, index=False)
    scored_alleles.update(chunk["hla_allele"].unique())
    print(f"    Checkpoint: wrote {len(chunk):,} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iedb", "cedar", "both"], default="both")
    args = parser.parse_args()

    if args.dataset in ("iedb", "both"):
        process_dataset("iedb",
                        DATA_DIR / "iedb_raw.tsv",
                        DATA_DIR / "iedb_mhc.tsv")
    if args.dataset in ("cedar", "both"):
        process_dataset("cedar",
                        DATA_DIR / "cedar_raw.tsv",
                        DATA_DIR / "cedar_mhc.tsv")


if __name__ == "__main__":
    main()
