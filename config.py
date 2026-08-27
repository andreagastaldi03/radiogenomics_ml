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

# CI bootstrap sulla AUC pooled out-of-fold: ricampiona i pazienti (con
# reinserimento) dalle predizioni OOF già salvate, nessun nuovo fit. 
N_BOOTSTRAP_AUC_CI = 2000

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
# "variance"       -> soglia di varianza
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

# ---------------------------------------------------------------------------
# TEST DI SIGNIFICATIVITÀ CONGIUNTO SULLA SPECIFICATION CURVE (permutazione)
#
# Parte "inferenziale" della specification curve analysis (Simonsohn et al.
# 2020): si permutano le etichette e si rifà l'intera curva (tutte le
# specifiche) molte volte, poi si confronta una statistica riassuntiva della
# curva reale con la distribuzione della stessa statistica sulle curve
# permutate. Costoso (n_specifiche x N_PERMUTATIONS_SPEC_CURVE fit), quindi
# si usa di default una griglia ridotta (vedi REDUCED_SPEC_GRID in
# specification_curve.py) e un N di permutazioni contenuto.
# ---------------------------------------------------------------------------
N_PERMUTATIONS_SPEC_CURVE = 100     # Simonsohn consiglia 500+, ma con n=54 e una
                                    # griglia di specifiche già 50-100 è un 
                                    # compromesso ragionevole
SPEC_CURVE_SUMMARY_STAT = "median"  # statistica riassuntiva della curva: "median" 
                                    # o "mean" di auc_pooled
    
# Quante permutazioni valutare in parallelo (processi separati) nei test di
# significatività congiunti (joint_significance_test e la sua variante
# source_comparison): -1 = usa tutti i core disponibili, 1 = sequenziale. 
# Durante l'esecuzione parallela il parallelismo interno del Random Forest 
# (n_estimators) viene forzato a 1 automaticamente, altrimenti i processi si 
# contenderebbero gli stessi core inutilmente:
# vedi la nota su n_jobs_model in specification_curve.py.
N_JOBS_SPEC_CURVE_PERMUTATIONS = -1

# ---------------------------------------------------------------------------
# RADIOGENOMICA: associazione radiomica <-> genomica, INDIPENDENTE dalla label
#
# Punto diverso da tutto il resto della pipeline: qui non si chiede "questa
# feature aiuta a predire ADK/SCC?" ma "le feature radiomiche stabili e i
# geni stabili si muovono insieme tra pazienti?" — la classificazione
# binaria ADK/SCC potrebbe far passare inosservata un'associazione
# immagine-genoma reale, perché ogni feature deve prima "farsi strada"
# attraverso il collo di bottiglia di un'unica etichetta.
# ---------------------------------------------------------------------------
# quali feature considerare "stabili": righe di feature_consensus.csv (vedi
# feature_consensus.py) con almeno questo numero di criteri (stability
# selection, SHAP, voti spec curve) su 3 in cui la feature compare.
RADIOGENOMICS_MIN_CRITERIA = 2
 
RADIOGENOMICS_CORR_METHOD = "spearman"   # "spearman" (monotona, robusta) o "pearson"
RADIOGENOMICS_FDR_ALPHA = 0.05           # soglia sul q-value (Benjamini-Hochberg) per i singoli test
 
# permutazioni per il test globale (coefficiente RV) sull'intero blocco
# radiomica vs blocco genomica insieme, non sulle singole coppie
RADIOGENOMICS_N_PERMUTATIONS_RV = 5000

# ---------------------------------------------------------------------------
# CORREZIONE PER TEST MULTIPLI SUL CONSENSUS SCORE (feature_consensus.py)
#
# Con centinaia di feature scremate da tre criteri indipendenti (stability
# selection, SHAP, voti spec curve), guardare solo il ranking assoluto del
# consensus_score rischia falsi positivi: anche senza nessun segnale reale
# qualche feature otterrà comunque un consenso alto tra i tre criteri per
# puro caso. Si rigira l'intera pipeline di consenso su etichette permutate
# un piccolo numero di volte (ESPANSIVO: bootstrap stability + SHAP + intera
# specification curve, ripetuti per intero ad ogni permutazione) e si
# guarda la distribuzione nulla del consensus_score PIÙ ALTO ottenuto per
# caso (max-statistic / Westfall-Young): una feature reale deve battere
# quella soglia, non solo le altre feature del dataset.
# ---------------------------------------------------------------------------
CONSENSUS_N_PERMUTATIONS = 10   # "un paio di volte": costoso, si tiene basso di proposito.
                                # Con m permutazioni il p-value minimo ottenibile è 1/(m+1):
                                # con 3 permutazioni non si scende sotto 0.25 — usalo come
                                # controllo di plausibilità, non come test ad alta risoluzione.
N_JOBS_CONSENSUS_PERMUTATIONS = -1  # permutazioni in parallelo (stessa logica di
                                    # N_JOBS_SPEC_CURVE_PERMUTATIONS)