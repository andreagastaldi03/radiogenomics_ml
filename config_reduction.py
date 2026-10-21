"""
Configurazione della riduzione feature v2.

Percorsi, DATA_SOURCE, classe positiva ecc. restano quelli di config.py;
qui ci sono solo i parametri specifici della riduzione restrittiva.
I risultati vanno in OUTPUT_DIR/reduction_v2/<DATA_SOURCE>/<nome_specifica>/,
quindi non sovrascrivono mai quelli della pipeline precedente.
"""

import config

OUTPUT_ROOT = config.OUTPUT_DIR / "reduction_v2"

# ---------------------------------------------------------------------------
# SPECIFICHE DA CONFRONTARE
#
# Dichiara la griglia PRIMA di guardare i risultati e riportala per intero:
# scegliere a posteriori la specifica con l'AUC migliore è tuning sui dati di
# valutazione, e con n=54 la stima risulterebbe ottimistica.
#
# Chiavi = parametri di reduction_v2.BlockReducer:
#   k_texture / k_shape / k_gen : feature finali per blocco (None = usa corr_threshold)
#   mode                        : "medoid" | "pca"
#   corr_threshold              : usato solo dai blocchi con k=None
# Budget indicativo: ~ (n. pazienti classe minoritaria) / 10 feature in totale.
# ---------------------------------------------------------------------------
SPECS = {
    "t4s1_g5_medoid": dict(k_texture=4, k_shape=1, k_gen=5, mode="medoid"),
    "t4s1_g5_pca":    dict(k_texture=4, k_shape=1, k_gen=5, mode="pca"),
    "t8s2_g8_medoid": dict(k_texture=8, k_shape=2, k_gen=8, mode="medoid"),
    # stessa logica "a soglia" della versione precedente, ma con filtro di
    # dispersione e rappresentante corretti: isola l'effetto delle correzioni
    "thr050_medoid":  dict(k_texture=None, k_shape=None, k_gen=None,
                           corr_threshold=0.5, mode="medoid"),
}

# Per queste specifiche si eseguono anche stability selection e SHAP (costosi);
# per tutte le altre solo nested CV + metriche pooled.
SPECS_FULL_ANALYSIS = ["t4s1_g5_medoid"]