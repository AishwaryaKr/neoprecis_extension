"""Build padded bag arrays for MIL training (all/best/worst WT variants)."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MAX_HAMMING, MIN_BAG, MAX_PAD

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {aa: i for i, aa in enumerate(AA)}

_BLOSUM_FLAT = [
     4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2,
     0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2,
    -2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3,
    -1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2,
    -2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3,
     0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3,
    -2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2,
    -1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1,
    -1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2,
    -1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1,
    -1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1,
    -2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2,
    -1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3,
    -1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1,
    -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2,
     1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2,
     0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2,
     0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1,
    -3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2,
    -2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7,
]
BLOSUM62 = np.array(_BLOSUM_FLAT, dtype=np.float32).reshape(20, 20)

FEAT_COLS = [f"blosum_pos_{i}" for i in range(9)] + [
    "blosum_sim", "hamming_dist", "wt_rank_el", "mt_rank_el", "delta_mhc"
]
# 14 features per instance:
#   blosum_pos_0..8  — BLOSUM62 score of MT[i] vs WT[i] at each position (9-mer)
#   blosum_sim       — total BLOSUM62 similarity between full MT and WT strings
#   hamming_dist     — Hamming distance (H ≤ MAX_HAMMING=3)
#   wt_rank_el       — MHC %rank for the WT self-peptide; defaults to 50.0 when
#                      the WT was not scored by NetMHCpan (~99.5% of instances)
#   mt_rank_el       — MHC %rank for the MT binding core (from common set)
#   delta_mhc        — mt_rank_el - wt_rank_el; corr(delta, mt_rank_el) ≈ 0.96
#                      because wt_rank_el is effectively constant (50.0)
#
# PARITY NOTE: mt_rank_el and delta_mhc are MHC-binding features. All LR and MIL
# methods that use bag features (lr_best/worst/avg, meanpool, abmil) implicitly
# receive MHC rank. Only the scalar methods isolate individual signals.


def _blos(a: str, b: str) -> float:
    i = AA_IDX.get(a, -1); j = AA_IDX.get(b, -1)
    return float(BLOSUM62[i, j]) if i >= 0 and j >= 0 else 0.0


def _ss(seq: str) -> float:
    return sum(_blos(a, a) for a in seq)


def _per_pos(mt: str, wt: str):
    return [_blos(m, w) for m, w in zip(mt, wt)]


def _instance_features(mt: str, wt: str, h: int, blosum_sim: float,
                        wt_rank: float, mt_rank: float) -> np.ndarray:
    pos = _per_pos(mt, wt)
    return np.array(pos + [blosum_sim, h, wt_rank, mt_rank, mt_rank - wt_rank],
                    dtype=np.float32)


def build_variant(variant: str, common: pd.DataFrame, proteome: pd.DataFrame,
                  wt_rank_lookup: dict) -> dict:
    """Build bag arrays for one variant: 'all', 'best', or 'worst'."""
    print(f"  Building variant '{variant}' ...")

    prot_sorted = (
        proteome[proteome["hamming_dist"] >= 0]
        .sort_values(["binding_core", "blosum_sim"], ascending=[True, True])
    )
    grouped = prot_sorted.groupby("binding_core")

    bag_feats    = []
    bag_labels   = []
    bag_epitopes = []
    bag_sizes    = []
    skipped      = 0
    PAD_VEC = np.zeros(len(FEAT_COLS), dtype=np.float32)

    for _, row in common.iterrows():
        core     = row["binding_core"]
        label    = int(row["label"])
        mt_rank  = float(row["rank_el"])
        allele   = row["hla_allele"]

        if core not in grouped.groups:
            skipped += 1; continue

        grp = grouped.get_group(core)

        if variant == "best":
            instances = grp.iloc[[-1]]   # highest blosum_sim row
        elif variant == "worst":
            instances = grp.iloc[[0]]    # lowest blosum_sim row
        else:
            instances = grp              # all neighbors

        feats = []
        for _, inst in instances.iterrows():
            wt       = inst["self_peptide"]
            h        = int(inst["hamming_dist"])
            bsim     = float(inst["blosum_sim"])
            # Look up by (self_peptide, allele) if real WT MHC ranks are available
            wt_rank  = wt_rank_lookup.get((wt, allele), 50.0)
            feats.append(_instance_features(core, wt, h, bsim, wt_rank, mt_rank))

        n = len(feats)
        if n < MIN_BAG:
            skipped += 1; continue

        rows = np.array(feats, dtype=np.float32)
        if n < MAX_PAD:
            pad  = np.tile(PAD_VEC, (MAX_PAD - n, 1))
            rows = np.vstack([rows, pad])
        else:
            rows = rows[:MAX_PAD]
            n    = MAX_PAD

        bag_feats.append(rows)
        bag_labels.append(label)
        bag_epitopes.append(core)
        bag_sizes.append(min(len(feats), MAX_PAD))

    if skipped:
        print(f"    Skipped {skipped} cores (no neighbor or too small)")

    features   = np.stack(bag_feats)                           # (N, MAX_PAD, n_feat)
    labels     = np.array(bag_labels)
    bag_sizes  = np.array(bag_sizes)
    padding_mask = np.arange(MAX_PAD)[None, :] < bag_sizes[:, None]

    # PARITY NOTE — global normalization computed here, before any CV splits.
    # MIL models (meanpool, abmil) consume features_norm at training time, so their
    # normalization statistics are fit on the entire dataset including every val fold.
    # This is a mild data-leakage: val-fold mean/std inform the scaler.
    # LR methods use features_raw and apply a per-fold StandardScaler inside
    # run_ablation.py (clean). For strict parity, MIL models should normalize
    # per fold using the training split only.
    real = features[padding_mask]
    mean = real.mean(axis=0)
    std  = real.std(axis=0) + 1e-8
    features_norm = (features - mean) / std
    features_norm[~padding_mask] = 0.0

    print(f"    {len(labels):,} bags | pos={labels.mean():.1%} | "
          f"mean_size={bag_sizes.mean():.1f} | median={int(np.median(bag_sizes))}")

    return {
        "features":      features_norm,
        "features_raw":  features,
        "padding_mask":  padding_mask,
        "bag_sizes":     bag_sizes,
        "labels":        labels,
        "epitopes":      bag_epitopes,
        "feature_names": FEAT_COLS,
        "norm_mean":     mean,
        "norm_std":      std,
        "MAX_PAD":       MAX_PAD,
        "MAX_HAMMING":   MAX_HAMMING,
        "variant":       variant,
    }


def build_dataset(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"Building bags: {name.upper()}")

    common   = pd.read_csv(DATA_DIR / f"{name}_common.tsv",   sep="\t")
    proteome = pd.read_csv(DATA_DIR / f"{name}_proteome.tsv", sep="\t")

    # Load real WT MHC ranks from step 07 (pipeline/07_wt_mhc.py).
    # Keyed by (self_peptide, hla_allele) because the same WT peptide can appear
    # with different alleles and have different binding affinities.
    wt_mhc_path = DATA_DIR / f"{name}_wt_mhc.tsv"
    if wt_mhc_path.exists():
        wt_mhc = pd.read_csv(wt_mhc_path, sep="\t")
        # Join allele info from proteome scan; keyed by (self_peptide, hla_allele)
        prot_allele = proteome.merge(
            common[["binding_core", "hla_allele"]], on="binding_core", how="left"
        )[["self_peptide", "hla_allele"]].drop_duplicates()
        wt_joined = prot_allele.merge(wt_mhc, on=["self_peptide", "hla_allele"], how="left")
        wt_rank_lookup: dict[tuple[str, str], float] = {
            (row["self_peptide"], row["hla_allele"]): float(row["wt_rank_el"])
            for _, row in wt_joined.dropna(subset=["wt_rank_el"]).iterrows()
        }
        n_covered = len(wt_rank_lookup)
        n_total   = len(prot_allele)
        print(f"  wt_rank_lookup: {n_covered:,}/{n_total:,} (self_peptide, allele) pairs "
              f"have real MHC rank ({n_covered/max(n_total,1):.1%})")
    else:
        wt_rank_lookup = {}
        print("  WARNING: wt_mhc.tsv not found — wt_rank_el will default to 50.0. "
              "Run pipeline/07_wt_mhc.py first.")

    payload = {}
    for variant in ("all", "best", "worst"):
        payload[variant] = build_variant(variant, common, proteome, wt_rank_lookup)

    out_path = DATA_DIR / f"{name}_bags.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved → {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iedb", "cedar", "both"], default="both")
    args = parser.parse_args()

    for name in (["iedb", "cedar"] if args.dataset == "both" else [args.dataset]):
        build_dataset(name)


if __name__ == "__main__":
    main()
