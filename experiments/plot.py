"""
Plot ablation results.

Produces:
  figures/auc_comparison_{dataset}.png  — grouped bar chart per dataset
  figures/auc_heatmap.png               — dataset × method heatmap

Run: python experiments/plot.py [--dataset iedb|cedar|both]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Method display names and grouping
METHOD_META = {
    "scalar_distoself":   ("DisToSelf",          "Scalar baseline"),
    "scalar_rank_el":     ("MHC rank",            "Scalar baseline"),
    "scalar_crd_best":    ("CRD best WT",         "Scalar baseline"),
    "scalar_crd_worst":   ("CRD worst WT",        "Scalar baseline"),
    "scalar_crd_avg":     ("CRD avg bag",         "Scalar baseline"),
    "lr_best_blosum":     ("LR — best WT",        "Single instance"),
    "lr_worst_blosum":    ("LR — worst WT",       "Single instance"),
    "lr_avg_blosum":      ("LR — bag avg",        "Bag aggregation"),
    "meanpool_blosum":    ("MeanPool MIL",        "MIL"),
    "abmil_blosum":       ("ABMIL",               "MIL"),
}

GROUP_COLORS = {
    "Scalar baseline": "#7fb3d3",
    "Single instance": "#f0a500",
    "Bag aggregation": "#e06c75",
    "MIL":             "#98c379",
}


def plot_dataset(name: str, df: pd.DataFrame) -> None:
    summary = (
        df.groupby("method")["auc"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    # Order by METHOD_META key order
    order = [m for m in METHOD_META if m in summary["method"].values]
    summary["method_order"] = summary["method"].map({m: i for i, m in enumerate(order)})
    summary = summary.sort_values("method_order")

    labels  = [METHOD_META[m][0] for m in summary["method"]]
    groups  = [METHOD_META[m][1] for m in summary["method"]]
    means   = summary["mean"].values
    stds    = summary["std"].values
    colors  = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
           color=colors, edgecolor="white", linewidth=0.5)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.003, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("5-fold CV AUC", fontsize=11)
    ax.set_title(f"{name.upper()} — MIL Ablation", fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, min(1.0, means.max() + 0.12))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
               if g in groups]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / f"auc_comparison_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_heatmap(all_df: pd.DataFrame) -> None:
    summary = all_df.groupby(["dataset", "method"])["auc"].mean().reset_index()
    pivot   = summary.pivot(index="method", columns="dataset", values="auc")
    order   = [m for m in METHOD_META if m in pivot.index]
    pivot   = pivot.reindex(order)
    row_labels = [METHOD_META[m][0] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(4 + len(pivot.columns), 0.5 * len(pivot)))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn",
                   vmin=0.45, vmax=0.75, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.upper() for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if 0.5 < val < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="AUC", shrink=0.8)
    ax.set_title("AUC by method and dataset", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "auc_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iedb", "cedar", "both"], default="both")
    args = parser.parse_args()

    datasets = ["iedb", "cedar"] if args.dataset == "both" else [args.dataset]
    all_dfs = []
    for name in datasets:
        path = RESULTS_DIR / f"{name}_results.csv"
        if not path.exists():
            print(f"  Missing {path} — run run_ablation.py first")
            continue
        df = pd.read_csv(path)
        plot_dataset(name, df)
        all_dfs.append(df)

    if len(all_dfs) > 1:
        plot_heatmap(pd.concat(all_dfs, ignore_index=True))


if __name__ == "__main__":
    main()
