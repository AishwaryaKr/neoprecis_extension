from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NEOPRECIS_DIR  = Path.home() / "NeoPrecis" / "neoprecis"
NETMHCPAN      = ROOT / "netMHCpan-4.2" / "netMHCpan"

IEDB_JSON      = ROOT / "iedb_full.json"
CEDAR_JSON     = ROOT / "cedar_full.json"
PROTEOME_FASTA = ROOT / "human_proteome.fasta.gz"

DATA_DIR    = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

MIN_LENGTH = 8
MAX_LENGTH = 14

MAX_HAMMING = 3
MIN_BAG     = 1
MAX_PAD     = 50

N_FOLDS     = 5
RANDOM_SEED = 42

EPOCHS     = 100
LR         = 1e-3
HIDDEN_DIM = 64
ATTN_DIM   = 32
DROPOUT    = 0.1
BATCH_SIZE = 256
