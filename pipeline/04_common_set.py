"""Build the shared record set used by all methods, with per-filter audit logging."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import NEOPRECIS_DIR, DATA_DIR

def _load_crd_alleles() -> set[str]:
    allele_map_path = NEOPRECIS_DIR / "allele_code_mapping.json"
    with open(allele_map_path) as f:
        mapping = json.load(f)
    alleles: set[str] = set()
    for gene, codes in mapping.items():
        if isinstance(codes, dict):
            for code in codes.keys():
                if len(code) >= 4:
                    alleles.add(f"HLA-{gene}*{code[:2]}:{code[2:4]}")
    return alleles


def build_common_set(name: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Building common set: {name.upper()}")

    raw       = pd.read_csv(DATA_DIR / f"{name}_raw.tsv",      sep="\t")
    mhc       = pd.read_csv(DATA_DIR / f"{name}_mhc.tsv",      sep="\t")
    proteome  = pd.read_csv(DATA_DIR / f"{name}_proteome.tsv", sep="\t")

    print(f"  raw: {len(raw):,} rows, {raw['epitope'].nunique():,} unique epitopes")
    print(f"  mhc: {len(mhc):,} rows")
    print(f"  proteome: {proteome['binding_core'].nunique():,} cores with neighbor")

    audit: list[dict] = []

    n_raw_epitopes = raw["epitope"].nunique()
    # Inner join on epitope×allele — drops epitopes that failed NetMHCpan prediction.
    # Only MT binding cores are scored; WT self-proteome peptides are NOT scored here,
    # so wt_rank_el in bags defaults to 50.0 for virtually all instances.
    merged = raw.merge(
        mhc[["epitope", "hla_allele", "binding_core", "rank_el", "score_el"]],
        on=["epitope", "hla_allele"],
        how="inner",
    )
    n_no_mhc = n_raw_epitopes - merged["epitope"].nunique()
    audit.append({"step": "no_mhc_score", "reason": "epitope×allele not in NetMHCpan output",
                  "n_epitopes_lost": n_no_mhc, "n_cores_lost": "—"})
    print(f"  After MHC merge: {len(merged):,} rows")

    before = len(merged)
    merged = merged.dropna(subset=["rank_el", "score_el"])
    n_missing = before - len(merged)
    audit.append({"step": "missing_mhc_values", "reason": "rank_el or score_el is NaN",
                  "n_epitopes_lost": n_missing, "n_cores_lost": "—"})
    print(f"  After dropping missing MHC scores: {len(merged):,} rows")

    def majority_vote(x):
        pos = (x == "Positive").sum()
        neg = (x == "Negative").sum()
        if pos == neg:
            return float("nan")
        return "Positive" if pos > neg else "Negative"

    core_meta = (
        merged.groupby("binding_core")
        .agg(
            outcome         = ("outcome",          majority_vote),
            rank_el         = ("rank_el",          "mean"),
            score_el        = ("score_el",         "mean"),
            hla_allele      = ("hla_allele",        "first"),
            source_category = ("source_category",  "first"),
        )
        .reset_index()
    )
    n_before_ties = len(core_meta)
    core_meta = core_meta.dropna(subset=["outcome"])
    n_ties = n_before_ties - len(core_meta)
    audit.append({"step": "conflicting_outcome", "reason": "equal Pos/Neg votes for binding_core",
                  "n_epitopes_lost": "—", "n_cores_lost": n_ties})
    core_meta["label"] = (core_meta["outcome"] == "Positive").astype(int)
    print(f"  After dedup to core level: {len(core_meta):,} unique cores")
    print(f"  Positive rate: {core_meta['label'].mean():.1%}")

    cores_with_neighbor = set(proteome["binding_core"].unique())
    n_before = len(core_meta)
    core_meta = core_meta[core_meta["binding_core"].isin(cores_with_neighbor)]
    n_no_neighbor = n_before - len(core_meta)
    audit.append({"step": "no_proteome_neighbor",
                  "reason": f"no human proteome peptide within H≤{3} of binding_core",
                  "n_epitopes_lost": "—", "n_cores_lost": n_no_neighbor})
    print(f"  After requiring proteome neighbor: {len(core_meta):,} cores")

    try:
        crd_alleles = _load_crd_alleles()
        n_before = len(core_meta)
        core_meta = core_meta[core_meta["hla_allele"].isin(crd_alleles)]
        n_bad_allele = n_before - len(core_meta)
        audit.append({"step": "unsupported_allele",
                      "reason": "HLA allele not in NeoPrecis allele_code_mapping.json",
                      "n_epitopes_lost": "—", "n_cores_lost": n_bad_allele})
        print(f"  After requiring CRD-compatible allele: {len(core_meta):,} cores "
              f"(dropped {n_before - len(core_meta):,})")
    except FileNotFoundError:
        print("  WARNING: CRD allele list not found — skipping allele filter")

    audit_path = DATA_DIR / f"{name}_filter_audit.tsv"
    pd.DataFrame(audit).to_csv(audit_path, sep="\t", index=False)
    print(f"\nFilter audit → {audit_path}")

    dts = proteome.groupby("binding_core")["distoself_blosum"].first().reset_index()
    core_meta = core_meta.merge(dts, on="binding_core", how="left")

    n_neighbors = proteome.groupby("binding_core").size().rename("n_neighbors").reset_index()
    core_meta = core_meta.merge(n_neighbors, on="binding_core", how="left")
    core_meta["n_neighbors"] = core_meta["n_neighbors"].fillna(0).astype(int)

    out_cols = ["binding_core", "hla_allele", "label", "rank_el", "score_el",
                "distoself_blosum", "n_neighbors", "source_category"]
    out = core_meta[out_cols].reset_index(drop=True)

    out_path = DATA_DIR / f"{name}_common.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    print(f"\nFinal common set: {len(out):,} binding cores")
    print(f"  Positive: {out['label'].sum():,} ({out['label'].mean():.1%})")
    print(f"  Bag size: mean={out['n_neighbors'].mean():.1f}, "
          f"median={out['n_neighbors'].median():.0f}, "
          f"singleton={( out['n_neighbors']==1).mean():.1%}")
    print(f"Saved → {out_path}")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iedb", "cedar", "both"], default="both")
    args = parser.parse_args()

    for name in (["iedb", "cedar"] if args.dataset == "both" else [args.dataset]):
        build_common_set(name)


if __name__ == "__main__":
    main()
