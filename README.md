# MIL Ablation Study

Systematic comparison of multiple instance learning (MIL) approaches for neoantigen immunogenicity prediction.

## What is compared

Three independent dimensions:

| Dimension | Options |
|---|---|
| **Instance selection** | best WT (closest BLOSUM), worst WT (most different), avg / full bag |
| **Model architecture** | scalar (no model), LR, bag avg → LR, MeanPool MIL, ABMIL |
| **Dataset** | IEDB, CEDAR |

Every method runs on the **same set of binding cores** with the **same CV folds** for a fair comparison.

## Pipeline

Run steps in order:

```bash
# 1. Ingest raw IEDB and CEDAR JSON → filtered TSVs
python pipeline/01_ingest.py

# 2. Run NetMHCpan → binding_core, rank_el, score_el  (~30 min IEDB, ~5 min CEDAR)
python pipeline/02_netmhcpan.py

# 3. Proteome scan H<=3 — find all human proteome neighbors  (~2-3h IEDB, ~30 min CEDAR)
python pipeline/03_proteome_scan.py --workers 7

# 4. Build the common record set (intersection of all data sources)
python pipeline/04_common_set.py

# 5. Score with PeptCRD using score_el (best, worst, avg WT)  (~1h)
python pipeline/05_crd_scores.py

# 6. Build padded bag arrays (all / best / worst variants)
python pipeline/06_build_bags.py

# 7. Run the ablation experiments
python experiments/run_ablation.py

# 8. Plot results
python experiments/plot.py
```

Each script accepts `--dataset iedb|cedar|both` and long-running steps checkpoint automatically and resume on re-run.

## Configuration

Edit `config.py` to set:
- `NEOPRECIS_DIR` — path to NeoPrecis installation
- `NETMHCPAN` — path to NetMHCpan binary
- `MAX_HAMMING`, `MAX_PAD`, `MIN_BAG` — bag construction parameters
- `N_FOLDS`, `EPOCHS`, `HIDDEN_DIM`, etc. — training hyperparameters

## Requirements

```
numpy pandas scikit-learn torch ijson
```

NeoPrecis must be installed separately (see NeoPrecis README).
NetMHCpan-4.1+ must be installed and the binary path set in `config.py`.

## Output

- `data/`    — generated data files (not committed to git)
- `results/` — per-dataset CSV with method × fold AUC values
- `figures/` — bar charts and heatmap

## Methods

| Method | Description |
|---|---|
| `scalar_distoself` | AUC of distoself_blosum alone |
| `scalar_rank_el` | AUC of MHC %rank alone |
| `scalar_crd_best` | AUC of CRD score (best WT by BLOSUM) |
| `scalar_crd_worst` | AUC of CRD score (worst WT by BLOSUM) |
| `scalar_crd_avg` | AUC of CRD score (avg across all H≤3 neighbors) |
| `lr_best_blosum` | 14 per-position BLOSUM features of best WT → LR |
| `lr_worst_blosum` | 14 per-position BLOSUM features of worst WT → LR |
| `lr_avg_blosum` | average of per-instance features across bag → LR |
| `meanpool_blosum` | per-instance features, mean pooled → FF classifier |
| `abmil_blosum` | per-instance features, attention pooled → FF classifier |
