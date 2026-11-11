"""
Score WT self-proteome peptides with NetMHCpan so wt_rank_el carries real signal.

For each dataset, joins the proteome scan output with the common set to get the
HLA allele per binding core, then runs NetMHCpan on all unique (WT peptide, allele)
pairs grouped by allele.  Saves data/{name}_wt_mhc.tsv with columns:
  self_peptide, hla_allele, wt_rank_el, wt_score_el
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import NETMHCPAN, DATA_DIR

CHECKPOINT_EVERY = 10


def run_netmhcpan_wt(peptides: List[str], allele: str, tmp_dir: str) -> pd.DataFrame:
    pept_file = os.path.join(tmp_dir, "wt_peptides.txt")
    out_file  = os.path.join(tmp_dir, "wt_output.txt")

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
    rank_col  = (col_lower.get("el_rank") or col_lower.get("%rank_el")
                 or col_lower.get("rank_el") or col_lower.get("%rank"))
    score_col = (col_lower.get("el_score") or col_lower.get("score_el")
                 or col_lower.get("score") or col_lower.get("1-log50k(aff)"))

    if not pept_col or not rank_col:
        print(f"    WARNING: missing columns for {allele}: {list(df.columns)}")
        return pd.DataFrame()

    out = df[[pept_col]].copy()
    out.columns = ["self_peptide"]
    out["hla_allele"]   = allele
    out["wt_rank_el"]   = pd.to_numeric(df[rank_col],  errors="coerce")
    out["wt_score_el"]  = pd.to_numeric(df[score_col], errors="coerce") if score_col else float("nan")
    return out.drop_duplicates(subset=["self_peptide", "hla_allele"])


def _append(results: List[pd.DataFrame], out_path: Path, done: Set[str]) -> None:
    chunk = pd.concat(results, ignore_index=True)
    chunk.to_csv(out_path, sep="\t", mode="a", header=not out_path.exists(), index=False)
    done.update(chunk["hla_allele"].unique())
    print(f"    Checkpoint: {len(chunk):,} rows → {out_path}")


def build_pairs(name: str) -> pd.DataFrame:
    """Return unique (self_peptide, hla_allele) pairs for a dataset."""
    proteome = pd.read_csv(DATA_DIR / f"{name}_proteome.tsv", sep="\t")
    common   = pd.read_csv(DATA_DIR / f"{name}_common.tsv",   sep="\t")
    pairs = (proteome
             .merge(common[["binding_core", "hla_allele"]], on="binding_core")
             [["self_peptide", "hla_allele"]]
             .drop_duplicates())
    return pairs


def process_dataset(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"Scoring WT peptides: {name.upper()}")

    pairs    = build_pairs(name)
    out_path = DATA_DIR / f"{name}_wt_mhc.tsv"

    done: Set[str] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, sep="\t")
        done = set(existing["hla_allele"].unique())
        print(f"  Resuming: {len(done)} alleles already scored")

    by_allele: Dict[str, List[str]] = {}
    for _, row in pairs.iterrows():
        if row["hla_allele"] not in done:
            by_allele.setdefault(row["hla_allele"], []).append(row["self_peptide"])

    total_pairs = sum(len(v) for v in by_allele.values())
    print(f"  {len(pairs):,} total pairs  |  {len(by_allele)} alleles remaining  |  ~{total_pairs:,} peptides")

    if not by_allele:
        print("  All alleles already scored — nothing to do.")
        return

    results: List[pd.DataFrame] = []
    with tempfile.TemporaryDirectory() as tmp:
        alleles = sorted(by_allele.keys())
        for i, allele in enumerate(alleles):
            peptides = list(set(by_allele[allele]))
            print(f"  [{i+1}/{len(alleles)}] {allele}: {len(peptides):,} peptides")
            df = run_netmhcpan_wt(peptides, allele, tmp)
            if not df.empty:
                results.append(df)
            if results and (i + 1) % CHECKPOINT_EVERY == 0:
                _append(results, out_path, done)
                results = []

    if results:
        _append(results, out_path, done)

    if out_path.exists():
        final = pd.read_csv(out_path, sep="\t")
        total = len(pairs)
        covered = final.drop_duplicates(["self_peptide","hla_allele"])
        print(f"\nDone. {len(final):,} rows in {out_path}")
        print(f"  Coverage: {len(covered):,}/{total:,} pairs ({len(covered)/total:.1%})")
        print(f"  wt_rank_el: mean={final.wt_rank_el.mean():.1f}  median={final.wt_rank_el.median():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iedb", "cedar", "both"], default="both")
    args = parser.parse_args()

    for name in (["iedb", "cedar"] if args.dataset == "both" else [args.dataset]):
        process_dataset(name)


if __name__ == "__main__":
    main()
