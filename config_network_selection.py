"""
Configurazione della selezione di feature radiomiche guidata dalla rete
(file NUOVO: nessun altro file di configurazione viene toccato).

I risultati vanno in OUTPUT_DIR/network_selection/, separati da quelli delle
altre pipeline.
"""

import config

# Reti NEUTRAL (costruite senza etichetta): niente leakage nella selezione.
# Percorsi come in plot_tpi1_zoom.py; adattali se i file sono altrove.
_NET = config.OUTPUT_DIR / "network" / "neutral"
NETWORK_GRAPHML = {
    "separate": _NET / "separate" / "network_confirmed.graphml",  # FDR per blocco
    "unified":  _NET / "unified" / "network_confirmed.graphml",   # FDR unico su tutte le coppie
}

OUTPUT_ROOT = config.OUTPUT_DIR / "network_selection"

# Quante feature radiomiche tenere (budget). Con ~54 pazienti: 3-8.
K_RAD_LEVELS = [3, 5, 8]

# Criteri di selezione basati sulla rete (vedi network_feature_selection.py)
NET_SELECTORS = ["gene_strength", "betweenness", "eigenvector", "community_rep"]

# Riferimenti senza rete (reduction_v2.BlockReducer sulle 107 feature radiomiche)
BASELINE_SELECTORS = ["v2_medoid", "v2_pca"]

# Due feature collegate da un arco con |r| >= soglia sono considerate
# ridondanti: la seconda non viene scelta. (Nei grafi |r| rad-rad: mediana ~0.5,
# 75° percentile ~0.7.)
REDUNDANCY_MAX_ABS_R = 0.7

# Blocco genomico usato quando data_source = "both": uguale per TUTTE le
# specifiche, così le differenze dipendono solo dalla parte radiomica.
GENE_REDUCTION = dict(k_gen=5, mode="pca")

# Controllo: quante estrazioni casuali di k feature dai nodi radiomici della
# rete (stesso pool di candidati dei selettori di rete).
N_RANDOM_DRAWS = 20

N_FOLDS = 5
N_REPEATS = 1   # >1: ripete la CV con seed diversi (riduce il rumore dello split)
