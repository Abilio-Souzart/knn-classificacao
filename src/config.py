"""Caminhos e parâmetros globais."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "base_pejota.xls"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

SEED = 42
TEST_SIZE = 0.30
N_FOLDS = 10
K_ENUNCIADO = list(range(1, 16))          # item 11(c)
K_AMPLO = list(range(1, 102, 2))          # análise complementar (k ímpar até 101)
K_IRIS = [1, 5, 15]
TEST_SIZE_IRIS = 0.25

COLS_EXCLUIDAS = ["COD_CLI", "STATUS", "DEFAULT"]
COLS_CATEGORICAS = ["UF"]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
