"""
Configurazione centrale del progetto.
Modifica qui i percorsi e i parametri principali: il resto del codice li importa da qui.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# PERCORSI
# ---------------------------------------------------------------------------
DATA_DIR = Path("/content/drive/MyDrive/problem_solving/project/")
OUTPUT_DIR = Path("/content/drive/MyDrive/problem_solving/project/outputs")

RADIOMICS_PATH = DATA_DIR / "out_CTinvivo_roiOrig.csv"          # 107 feature
GENOMICS_PATH = DATA_DIR / "geni_normalizzati_R_250226.xlsx"    # 750 geni
LABELS_PATH = DATA_DIR / "Tabella_di_conversione_completa.xlsx" # patient_id, label

ID_COL = "Nome micro CT"
LABEL_COL = "fenotipo"
POSITIVE_CLASS = "ADK"   # classe codificata come 1 nelle metriche 

# ---------------------------------------------------------------------------
# QUALE SORGENTE DATI USARE PER IL MODELLO ML
# ---------------------------------------------------------------------------
# "radiomics"  -> solo feature TAC
# "genomics"   -> solo feature genomiche
# "both"       -> early fusion (concatenazione)
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

# ---------------------------------------------------------------------------
# DIAGNOSTICA: test di permutazione e controllo di batch effect
#
# prima di fidarti di un buon punteggio del modello,questi due controlli verificano
# che non sia "falso": il test di permutazione controlla che il modello non stia
# indovinando per caso; il controllo di batch effect controlla che non stia 
# riconoscendo un artefatto tecnico (es. giorno di scansione) invece della vera
# differenza biologica.
# ---------------------------------------------------------------------------
N_PERMUTATIONS = 200         # quante volte rimescolare le etichette a caso
PERMUTATION_N_FOLDS = 5      # k-fold semplice (non nested) usata per velocità

# Nome della colonna in Tabella_di_conversione_completa.xlsx che indica un
# lotto tecnico (es. data di estrazione RNA, run di sequenziamento, piastra).
# Lascia None se non hai questa informazione: il controllo funzionerà
# comunque, solo in versione "solo fenotipo" invece di "fenotipo + batch".
BATCH_COL = None

# ---------------------------------------------------------------------------
# SELEZIONE GENI ALTERNATIVA
#
# invece di scartare i geni "troppo piatti" con la varianza (criterio attuale), 
# questi metodi selezionano un numero fisso di geni tra i più "variabili" tra 
# pazienti secondo l'IQR (la differenza tra il 75° e il 25° percentile), più robusto
# agli outlier della semplice varianza.
# ---------------------------------------------------------------------------
# "variance"       -> criterio attuale (soglia di varianza)
# "iqr_top_n"      -> tiene i GENE_IQR_TOP_N geni con IQR più alta
# "iqr_top_pct"    -> tiene la percentuale GENE_IQR_TOP_PCT di geni più variabili
# "iqr_threshold"  -> tiene i geni con IQR sopra GENE_IQR_THRESHOLD
GENE_SELECTION_METHOD = "iqr_top_pct"
GENE_IQR_TOP_N = 50
GENE_IQR_TOP_PCT = 0.05
GENE_IQR_THRESHOLD = 1000

# ---------------------------------------------------------------------------
# FEATURE RADIOMICHE "DI FORMA" (shape)
#
# le feature che descrivono volume, diametri, sfericità del tumore si comportano 
# in modo diverso dalle feature di texture. Nello studio statistico pregresso 
# venivano ridotte SEPARATAMENTE. Se True, questa pipeline fa lo stesso invece di 
# trattarle tutte insieme.
# ---------------------------------------------------------------------------
EXCLUDE_SHAPE_FROM_MAIN_REDUCTION = True

