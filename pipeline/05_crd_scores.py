"""Score each binding_core with PeptCRD (best/worst/avg WT) using score_el."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import NEOPRECIS_DIR, DATA_DIR

sys.path.insert(0, str(NEOPRECIS_DIR))
from CRD.CRD import PeptCRD  # noqa: E402

CRD_REF    = NEOPRECIS_DIR / "CRD" / "ref.h5"
CRD_CONFIG = NEOPRECIS_DIR / "CRD" / "PeptCRD_config.yaml"
CRD_CKPT   = NEOPRECIS_DIR / "CRD" / "PeptCRD_checkpoint.ckpt"

CHECKPOINT = 500


def _load_model() -> PeptCRD:
    print("Loading PeptCRD model ...")
    return PeptCRD(str(CRD_REF), str(CRD_CONFIG), str(CRD_CKPT))


def _score_one(model: PeptCRD, wt: str, mt: str, allele: str,
               score_el: float) -> tuple[float, float]:
    """Returns (pept_crd, immgen). NaN on failure."""
    if not wt or not mt or len(wt) != len(mt):
        return float("nan"), float("nan")
    try:
        allele_fmt = allele.replace("HLA-", "")
        preds = model.score_peptide(wt, mt, allele_fmt, float(score_el))
        return float(preds[-2]), float(preds[-1])
    except Exception:
        return float("nan"), float("nan")


def score_dataset(name: str, model: PeptCRD) -> None:
    print(f"\n{'='*60}")
    print(f"CRD scoring: {name.upper()}")

    common   = pd.read_csv(DATA_DIR / f"{name}_common.tsv",   sep="\t")
    proteome = pd.read_csv(DATA_DIR / f"{name}_proteome.tsv", sep="\t")
    out_path = DATA_DIR / f"{name}_crd.tsv"

    done: set[str] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, sep="\t")
        done = set(existing["binding_core"].unique())
        print(f"  Resuming: {len(done):,} cores already scored")

    wt_lookup = (
        proteome[proteome["hamming_dist"] >= 0]
        .sort_values(["binding_core", "blosum_sim"])
        .groupby("binding_core")
    )

    rows_out: list[dict] = []
    todo = common[~common["binding_core"].isin(done)]
    total = len(todo)
    print(f"  {total:,} cores to score")

    t0 = time.time()
    for i, (_, row) in enumerate(todo.iterrows()):
        core    = row["binding_core"]
        allele  = row["hla_allele"]
        sel     = float(row["score_el"])

        if core not in wt_lookup.groups:
            rows_out.append({
                "binding_core": core,
                "best_wt": "", "worst_wt": "",
                "crd_best": float("nan"), "immgen_best": float("nan"),
                "crd_worst": float("nan"), "immgen_worst": float("nan"),
                "crd_avg": float("nan"),   "immgen_avg": float("nan"),
                "n_wt": 0,
            })
            continue

        grp = wt_lookup.get_group(core)
        worst_wt = grp.iloc[0]["self_peptide"]   # lowest blosum_sim
        best_wt  = grp.iloc[-1]["self_peptide"]  # highest blosum_sim
        n_wt     = len(grp)

        c_best,  i_best  = _score_one(model, best_wt,  core, allele, sel)
        c_worst, i_worst = _score_one(model, worst_wt, core, allele, sel)

        c_all = [_score_one(model, r["self_peptide"], core, allele, sel)
                 for _, r in grp.iterrows()]
        valid_c = [c for c, _ in c_all if not np.isnan(c)]
        valid_i = [ig for _, ig in c_all if not np.isnan(ig)]
        c_avg = float(np.mean(valid_c)) if valid_c else float("nan")
        i_avg = float(np.mean(valid_i)) if valid_i else float("nan")

        rows_out.append({
            "binding_core": core,
            "best_wt":      best_wt,
            "worst_wt":     worst_wt,
            "crd_best":     c_best,  "immgen_best":  i_best,
            "crd_worst":    c_worst, "immgen_worst": i_worst,
            "crd_avg":      c_avg,   "immgen_avg":   i_avg,
            "n_wt":         n_wt,
        })

        if (i + 1) % CHECKPOINT == 0 or (i + 1) == total:
            df_chunk = pd.DataFrame(rows_out)
            file_exists = out_path.exists()
            df_chunk.to_csv(out_path, sep="\t", mode="a",
                            header=not file_exists, index=False)
            rows_out = []
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate / 60
            print(f"  {i+1:,}/{total:,} ({100*(i+1)/total:.1f}%) "
                  f"| {rate:.0f} cores/s | ~{remaining:.0f} min left")

    if rows_out:
        df_chunk = pd.DataFrame(rows_out)
        df_chunk.to_csv(out_path, sep="\t", mode="a",
                        header=not out_path.exists(), index=False)

    final = pd.read_csv(out_path, sep="\t")
    print(f"\nDone. {len(final):,} rows → {out_path}")
    for col in ["crd_best", "crd_worst", "crd_avg"]:
        cov = final[col].notna().mean()
        print(f"  {col} coverage: {cov:.1%}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iedb", "cedar", "both"], default="both")
    args = parser.parse_args()

    model = _load_model()
    for name in (["iedb", "cedar"] if args.dataset == "both" else [args.dataset]):
        score_dataset(name, model)


if __name__ == "__main__":
    main()
