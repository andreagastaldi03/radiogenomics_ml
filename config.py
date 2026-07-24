"""
Configurazione centrale del progetto.
Modifica qui i percorsi e i parametri principali: il resto del codice li importa da qui.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# PERCORSI
# ---------------------------------------------------------------------------
# Il dataset atteso è un CSV con:
#   - una colonna "patient_id"
#   - una colonna "label" (es. "ADK" / "SCC")
#   - tutte le altre colonne = feature radiomiche (107) + feature genomiche (750)
# Se hai già due file separati (radiomica / genomica), impostali qui e uniscili
# in data_utils.load_data(). Il codice è già predisposto per entrambi i casi.

DATA_DIR = Path("/home/claude/radiogenomics_ml/data")
OUTPUT_DIR = Path("/home/claude/radiogenomics_ml/outputs")

RADIOMICS_PATH = DATA_DIR / "radiomics_features.csv"   # 107 feature
GENOMICS_PATH = DATA_DIR / "genomics_features.csv"      # 750 geni
LABELS_PATH = DATA_DIR / "labels.csv"                   # patient_id, label

ID_COL = "patient_id"
LABEL_COL = "label"
POSITIVE_CLASS = "ADK"   # classe codificata come 1 nelle metriche (scelta arbitraria, per coerenza)

# ---------------------------------------------------------------------------
# QUALE SORGENTE DATI USARE PER IL MODELLO ML
# ---------------------------------------------------------------------------
# "radiomics"  -> solo feature TAC
# "genomics"   -> solo feature genomiche
# "both"       -> early fusion (concatenazione)
# Ti consiglio di lanciare la pipeline 3 volte, una per ciascuna opzione,
# e confrontare i risultati: è di per sé un risultato scientifico.
DATA_SOURCE = "both"

# ---------------------------------------------------------------------------
# RIDUZIONE FEATURE "NEUTRA" (indipendente dalla label)
# ---------------------------------------------------------------------------
VARIANCE_THRESHOLD = 0.01          # rimuove feature quasi costanti (dopo standardizzazione su varianza relativa)
REDUNDANCY_CORR_THRESHOLD = 0.90   # soglia di correlazione Spearman per raggruppare feature ridondanti
REDUNDANCY_METHOD = "spearman"     # "spearman" o "pearson"

# ---------------------------------------------------------------------------
# NESTED CROSS-VALIDATION
# ---------------------------------------------------------------------------
N_OUTER_FOLDS = 5          # con n=54 valuta anche Leave-One-Out (vedi ml_pipeline.py)
N_INNER_FOLDS = 5
N_REPEATS_OUTER = 10       # ripeti la CV esterna con seed diversi per stimare la varianza della stima
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# STABILITY SELECTION (per la parte di rete / interpretazione)
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 200
STABILITY_SELECTION_THRESHOLD = 0.6  # una feature è "stabile" se selezionata in >=60% dei bootstrap

RANDOM_SEEDS_MULTI_RUN = list(range(10))  # per la ripetizione della nested CV con vari seed
