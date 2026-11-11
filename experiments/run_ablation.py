"""Run 10-method MIL ablation with 5-fold stratified CV on IEDB and CEDAR."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, RESULTS_DIR, N_FOLDS, RANDOM_SEED
from config import EPOCHS, LR as LR_RATE, HIDDEN_DIM, ATTN_DIM, DROPOUT, BATCH_SIZE
from models.abmil import MeanPool, ABMIL

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def cv_splits(labels: np.ndarray):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return list(skf.split(np.zeros(len(labels)), labels))


def auc(y_true, y_score) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask], y_score[mask]))


def run_scalar(name: str, scores: np.ndarray, labels: np.ndarray, splits):
    aucs = []
    for train_idx, val_idx in splits:
        aucs.append(auc(labels[val_idx], scores[val_idx]))
    return aucs


def run_lr(X: np.ndarray, labels: np.ndarray, splits):
    aucs = []
    for train_idx, val_idx in splits:
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = labels[train_idx], labels[val_idx]
        sc = StandardScaler()
        X_tr  = sc.fit_transform(X_tr)
        X_val = sc.transform(X_val)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=RANDOM_SEED)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)[:, 1]
        aucs.append(auc(y_val, proba))
    return aucs


def train_mil(model: nn.Module, features: np.ndarray, mask: np.ndarray,
              labels: np.ndarray, train_idx: np.ndarray) -> None:
    X = torch.tensor(features[train_idx], dtype=torch.float32)
    M = torch.tensor(mask[train_idx],     dtype=torch.bool)
    y = torch.tensor(labels[train_idx],   dtype=torch.float32)

    pos_w = (y == 0).sum() / (y == 1).sum().clamp(min=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    ds = TensorDataset(X, M, y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for xb, mb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def eval_mil(model: nn.Module, features: np.ndarray, mask: np.ndarray,
             labels: np.ndarray, val_idx: np.ndarray) -> float:
    X = torch.tensor(features[val_idx], dtype=torch.float32)
    M = torch.tensor(mask[val_idx],     dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        logits = model(X, M).numpy()
    return auc(labels[val_idx], logits)


def _normalize_bags_per_fold(features_raw: np.ndarray, mask: np.ndarray,
                              train_idx: np.ndarray) -> np.ndarray:
    """Fit mean/std on training instances only, apply to all, zero out padding."""
    real_train = features_raw[train_idx][mask[train_idx]]   # (N_train_real, F)
    mean = real_train.mean(axis=0)
    std  = real_train.std(axis=0) + 1e-8
    normed = (features_raw - mean) / std
    normed[~mask] = 0.0
    return normed


def run_mil(arch: str, bags: dict, splits):
    # Use features_raw + per-fold normalization to match LR's clean CV discipline.
    # Previously used bags["features"] (globally pre-normalized), which caused mild
    # data leakage because val-fold statistics informed the scaler.
    features_raw = bags["features_raw"]
    mask         = bags["padding_mask"]
    labels       = bags["labels"]
    n_feat       = features_raw.shape[2]
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        features = _normalize_bags_per_fold(features_raw, mask, train_idx)
        if arch == "meanpool":
            model = MeanPool(n_feat, hidden=HIDDEN_DIM, dropout=DROPOUT)
        else:
            model = ABMIL(n_feat, hidden=HIDDEN_DIM, attn_dim=ATTN_DIM, dropout=DROPOUT)
        train_mil(model, features, mask, labels, train_idx)
        fold_auc = eval_mil(model, features, mask, labels, val_idx)
        print(f"    fold {fold+1}/{N_FOLDS}: AUC={fold_auc:.4f}")
        aucs.append(fold_auc)
    return aucs


def run_dataset(name: str, method_filter) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"Dataset: {name.upper()}")

    common   = pd.read_csv(DATA_DIR / f"{name}_common.tsv", sep="\t")
    crd      = pd.read_csv(DATA_DIR / f"{name}_crd.tsv",    sep="\t")

    with open(DATA_DIR / f"{name}_bags.pkl", "rb") as f:
        bags = pickle.load(f)

    bag_cores = bags["all"]["epitopes"]
    core_set  = set(bag_cores)
    common    = common[common["binding_core"].isin(core_set)].copy()
    common    = common.set_index("binding_core").loc[bag_cores].reset_index()
    crd       = crd.set_index("binding_core").reindex(bag_cores).reset_index()

    labels = bags["all"]["labels"]
    splits = cv_splits(labels)

    print(f"  {len(labels):,} bags  |  pos={labels.mean():.1%}  |  {N_FOLDS}-fold CV")

    rows = []

    def _record(method: str, fold_aucs) -> None:
        mean = float(np.nanmean(fold_aucs))
        std  = float(np.nanstd(fold_aucs))
        print(f"  {method:<25}: {mean:.4f} ± {std:.4f}  {[f'{a:.4f}' for a in fold_aucs]}")
        for fold, a in enumerate(fold_aucs):
            rows.append({"dataset": name, "method": method,
                         "fold": fold + 1, "auc": a})

    if not method_filter or "scalar" in method_filter:
        print("\n  [Scalar]")
        _record("scalar_distoself",
                run_scalar(name, common["distoself_blosum"].values, labels, splits))
        _record("scalar_rank_el",
                run_scalar(name, -common["rank_el"].values, labels, splits))

        for col, tag in [("crd_best", "scalar_crd_best"),
                         ("crd_worst", "scalar_crd_worst"),
                         ("crd_avg",   "scalar_crd_avg")]:
            vals = crd[col].reindex(range(len(bag_cores))).values
            _record(tag, run_scalar(name, vals, labels, splits))

    if not method_filter or "lr_best" in method_filter:
        print("\n  [LR — best WT BLOSUM features]")
        # features_raw[:, 0, :]: first (and only real) instance per bag for the
        # "best" variant. Per-fold StandardScaler applied inside run_lr — clean CV.
        # Note: feature vector includes mt_rank_el (index 12) and delta_mhc (index 13),
        # so LR receives implicit MHC rank signal alongside positional BLOSUM features.
        X_best = bags["best"]["features_raw"][:, 0, :]   # (N, n_feat)
        _record("lr_best_blosum", run_lr(X_best, labels, splits))

    if not method_filter or "lr_worst" in method_filter:
        print("\n  [LR — worst WT BLOSUM features]")
        # Same structure as lr_best but selects lowest-BLOSUM-sim WT neighbor.
        X_worst = bags["worst"]["features_raw"][:, 0, :]
        _record("lr_worst_blosum", run_lr(X_worst, labels, splits))

    if not method_filter or "lr_avg" in method_filter:
        print("\n  [LR — bag average BLOSUM features]")
        # Averages features_raw over real (non-padded) instances, then feeds the
        # mean vector to LR. Uses the same "all" bag set as MeanPool/ABMIL, but:
        #   • uses features_raw (unnormalized), not features (globally normalized)
        #   • normalization is per-fold StandardScaler (clean CV)
        # PARITY GAP vs MIL: MIL sees globally normalized features; LR does not.
        feats = bags["all"]["features_raw"]                 # (N, PAD, F)
        mask  = bags["all"]["padding_mask"]                 # (N, PAD)
        X_avg = (feats * mask[:, :, None]).sum(axis=1) / \
                mask.sum(axis=1, keepdims=True).clip(min=1) # (N, F)
        _record("lr_avg_blosum", run_lr(X_avg, labels, splits))

    if not method_filter or "meanpool" in method_filter:
        print("\n  [MeanPool MIL]")
        # Uses bags["all"]["features"] — globally normalized (all 835k instances).
        # PARITY GAP: normalization was fit before CV splits (mild leakage).
        # Feature vector includes mt_rank_el and delta_mhc (implicit MHC rank).
        _record("meanpool_blosum", run_mil("meanpool", bags["all"], splits))

    if not method_filter or "abmil" in method_filter:
        print("\n  [ABMIL]")
        # Same data as MeanPool; adds learned per-instance attention weights.
        # Same normalization leakage and implicit MHC rank caveats apply.
        _record("abmil_blosum", run_mil("abmil", bags["all"], splits))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iedb", "cedar", "both"], default="both")
    parser.add_argument("--methods", nargs="*",
                        help="subset to run: scalar lr_best lr_worst lr_avg meanpool abmil")
    args = parser.parse_args()

    datasets = ["iedb", "cedar"] if args.dataset == "both" else [args.dataset]
    all_rows = []
    for name in datasets:
        df = run_dataset(name, args.methods)
        all_rows.append(df)
        out = RESULTS_DIR / f"{name}_results.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved → {out}")

    if len(all_rows) > 1:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "all_results.csv", index=False)


if __name__ == "__main__":
    main()
