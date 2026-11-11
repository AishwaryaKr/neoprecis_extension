"""Scan the human proteome for H≤MAX_HAMMING neighbors of each binding core."""

from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROTEOME_FASTA, DATA_DIR, MAX_HAMMING

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

_BLOSUM62_FLAT = [
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
BLOSUM62 = np.array(_BLOSUM62_FLAT, dtype=np.int8).reshape(20, 20)

_SHM_NAME = ""
_SHM_L    = 0
_SHM_N    = 0
_MAX_H    = MAX_HAMMING


def _init_worker(shm_name: str, L: int, N: int, max_h: int) -> None:
    global _SHM_NAME, _SHM_L, _SHM_N, _MAX_H
    _SHM_NAME, _SHM_L, _SHM_N, _MAX_H = shm_name, L, N, max_h


def _encode(seq: str) -> np.ndarray | None:
    arr = np.empty(len(seq), dtype=np.int8)
    for i, aa in enumerate(seq):
        idx = AA_INDEX.get(aa)
        if idx is None:
            return None
        arr[i] = idx
    return arr


def _worker_fn(core: str) -> list[dict]:
    from multiprocessing import shared_memory as shm_mod
    shm = shm_mod.SharedMemory(name=_SHM_NAME)
    arr = np.ndarray((_SHM_L, _SHM_N), dtype=np.int8, buffer=shm.buf)

    enc = _encode(core)
    if enc is None:
        shm.close()
        return []

    hamming = (arr != enc[:, np.newaxis]).sum(axis=0).astype(np.int16)

    blosum_scores = np.zeros(_SHM_N, dtype=np.int32)
    for i in range(_SHM_L):
        blosum_scores += BLOSUM62[int(enc[i])][arr[i]].astype(np.int32)
    ss = int(BLOSUM62[enc, enc].sum())
    max_sim = float(blosum_scores.max()) / ss if ss > 0 else 0.0
    distoself_blosum = 1.0 - max_sim

    within = np.where(hamming <= _MAX_H)[0]
    rows = []
    for idx in within:
        wt = "".join(AA_ORDER[int(arr[i, idx])] for i in range(_SHM_L))
        h  = int(hamming[idx])
        sim_blosum = sum(
            int(BLOSUM62[int(enc[i])][int(arr[i, idx])]) for i in range(_SHM_L)
        ) / ss if ss > 0 else 0.0
        rows.append({
            "binding_core":     core,
            "self_peptide":     wt,
            "hamming_dist":     h,
            "blosum_sim":       round(sim_blosum, 6),
            "distoself_blosum": round(distoself_blosum, 6),
        })

    if not rows:
        rows.append({
            "binding_core":     core,
            "self_peptide":     "",
            "hamming_dist":     -1,
            "blosum_sim":       float("nan"),
            "distoself_blosum": round(distoself_blosum, 6),
        })

    shm.close()
    return rows


def load_proteome(fasta_gz: Path) -> list[str]:
    seqs, current = [], []
    with gzip.open(fasta_gz, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                current = []
            else:
                current.append(line.strip())
    if current:
        seqs.append("".join(current))
    return seqs


def build_kmer_index(seqs: list[str], k: int) -> np.ndarray:
    seen: set[bytes] = set()
    kmers: list[np.ndarray] = []
    for seq in seqs:
        enc = np.full(len(seq), -1, dtype=np.int8)
        for i, aa in enumerate(seq):
            idx = AA_INDEX.get(aa)
            if idx is not None:
                enc[i] = idx
        for i in range(len(enc) - k + 1):
            w = enc[i:i + k]
            if np.any(w < 0):
                continue
            key = w.tobytes()
            if key not in seen:
                seen.add(key)
                kmers.append(w.copy())
    return np.array(kmers, dtype=np.int8).T  # (k, N_unique)


CHECKPOINT = 500


def scan_dataset(name: str, mhc_path: Path, out_path: Path, workers: int) -> None:
    print(f"\n{'='*60}")
    print(f"Proteome scan: {name.upper()}")
    mhc = pd.read_csv(mhc_path, sep="\t")
    cores = [c for c in mhc["binding_core"].dropna().unique() if len(c) == 9]
    print(f"  {len(cores):,} unique 9-mer binding cores")

    already: set[str] = set()
    if out_path.exists():
        ex = pd.read_csv(out_path, sep="\t")
        already = set(ex["binding_core"].unique())
        print(f"  Resuming: {len(already):,} cores already scanned")

    todo = [c for c in cores if c not in already]
    if not todo:
        print("  All cores already scanned.")
        return

    print(f"  Loading proteome ...")
    seqs = load_proteome(PROTEOME_FASTA)
    print(f"  {len(seqs):,} proteins")

    arr = build_kmer_index(seqs, 9)
    N   = arr.shape[1]
    print(f"  {N:,} unique 9-mers indexed")

    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray((9, N), dtype=np.int8, buffer=shm.buf)
    shm_arr[:] = arr

    pool = mp.Pool(workers, initializer=_init_worker,
                   initargs=(shm.name, 9, N, MAX_HAMMING))

    file_exists = out_path.exists()
    total_done  = len(already)
    try:
        for start in range(0, len(todo), CHECKPOINT):
            chunk = todo[start:start + CHECKPOINT]
            results = pool.map(_worker_fn, chunk, chunksize=50)
            rows = [r for sublist in results for r in sublist]
            df_chunk = pd.DataFrame(rows)
            df_chunk.to_csv(out_path, sep="\t", mode="a",
                            header=not file_exists, index=False)
            file_exists = True
            total_done += len(chunk)
            pct = 100 * total_done / (len(already) + len(todo))
            print(f"  Checkpoint {total_done:,}/{len(already)+len(todo):,} ({pct:.1f}%)")
    finally:
        pool.close(); pool.join()
        shm.close(); shm.unlink()

    final = pd.read_csv(out_path, sep="\t")
    final = final[final["hamming_dist"] >= 0].drop_duplicates(
        subset=["binding_core", "self_peptide"]
    )
    final.to_csv(out_path, sep="\t", index=False)

    n_covered = final["binding_core"].nunique()
    print(f"\nDone. {n_covered:,}/{len(cores):,} cores have ≥1 H≤{MAX_HAMMING} neighbor")
    print(f"  {len(final):,} total rows → {out_path}")
    h_dist = final["hamming_dist"].value_counts().sort_index()
    for h, n in h_dist.items():
        print(f"  H={h}: {n:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iedb", "cedar", "both"], default="both")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    if args.dataset in ("iedb", "both"):
        scan_dataset("iedb",
                     DATA_DIR / "iedb_mhc.tsv",
                     DATA_DIR / "iedb_proteome.tsv",
                     args.workers)
    if args.dataset in ("cedar", "both"):
        scan_dataset("cedar",
                     DATA_DIR / "cedar_mhc.tsv",
                     DATA_DIR / "cedar_proteome.tsv",
                     args.workers)


if __name__ == "__main__":
    main()
