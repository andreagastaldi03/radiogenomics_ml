"""
Caricamento dati e riduzione feature INDIPENDENTE dalla label.

Principio chiave: qui NON si guarda mai la variabile target. Questo step serve
a togliere ridondanza e rumore prima ancora di pensare alla classificazione,
in modo che lo stesso set ridotto possa essere riusato coerentemente sia per
il modello ML sia per lo studio di rete, senza circolarità statistica.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import config


# ---------------------------------------------------------------------------
# CARICAMENTO
# ---------------------------------------------------------------------------
def normalize_id(patient_id):
    if pd.isna(patient_id):
        return None
    patient_str = str(patient_id)
    match = re.search(r'\d+', patient_str)
    return match.group(0) if match else patient_str.strip()

def load_data(source: str = "both"):
    """
    Carica radiomica/genomica/entrambe e le allinea alla label per patient_id.

    Parametri
    -------
    source: str ("radiomics", "genomics", "both")

    Ritorna
    -------
    X : pd.DataFrame (righe=pazienti, colonne=feature normalizzate e pre-fissate)
    y : pd.Series (valori 0 per ADK, 1 per SCC)
    """
    # 1. Caricamento e allineamento LABELS
    labels_raw = pd.read_excel(LABELS_PATH, index_col=0).T
    labels_raw.columns = labels_raw.columns.str.strip()
    
    # Applichiamo la normalizzazione e mappiamo le classi (ADK=0, SCC=1)
    labels_raw['ID_Normalizzato'] = labels_raw[ID_COL].apply(normalize_id)
    labels_raw['Target'] = labels_raw[LABEL_COL].map({'ADK': 0, 'SCC': 1})
    
    # Creiamo la Series finale pulita
    y_all = labels_raw.set_index('ID_Normalizzato')['Target'].dropna()

    frames = []

    # 2. Caricamento e pulizia RADIOMICA
    if source in ("radiomics", "both"):
        rad = pd.read_csv(RADIOMICS_PATH, index_col=0)
        # Sostituzione virgole e cast a float
        rad = rad.replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce')
        # Normalizzazione indici e rinomina colonne
        rad.index = [normalize_id(idx) for idx in rad.index]
        rad.columns = [f"rad__{c}" for c in rad.columns]
        frames.append(rad)

    # 3. Caricamento e pulizia GENOMICA
    if source in ("genomics", "both"):
        gen = pd.read_excel(GENOMICS_PATH, index_col=0)
        # Rimuoviamo le ultime due colonne non utili e trasponiamo (pazienti sulle righe)
        gen = gen.iloc[:, :-2].T
        # Normalizzazione indici e rinomina colonne
        gen.index = [normalize_id(idx) for idx in gen.index]
        gen.columns = [f"gen__{c}" for c in gen.columns]
        frames.append(gen)

    if not frames:
        raise ValueError(f"Parametro 'source' non valido: {source}")

    # 4. Concatenazione dei blocchi feature
    X_all = pd.concat(frames, axis=1, join="inner")

    # 5. Inner Join finale tra Feature (X) e Target (y)
    common_idx = X_all.index.intersection(y_all.index)
    X = X_all.loc[common_idx].sort_index()
    y = y_all.loc[common_idx].sort_index()

    # Controllo di integrità
    assert X.shape[0] == y.shape[0], "Errore critico: Disallineamento tra feature e label dopo il merge."
    
    print(f"--- [load_data] completato ---")
    print(f"Source richiesto  : {source.upper()}")
    print(f"Pazienti allineati: {X.shape[0]}")
    print(f"Predittori totali : {X.shape[1]}")
    print(f"Distribuzione     :\n{y.value_counts().to_string()}\n")

    return X, y

# ---------------------------------------------------------------------------
# FILTRO PER VARIANZA
# ---------------------------------------------------------------------------
def variance_filter(X: pd.DataFrame, threshold: float = config.VARIANCE_THRESHOLD) -> pd.DataFrame:
    """
    Rimuove feature con varianza (calcolata su dati standardizzati per range)
    sotto soglia. Usa il coefficiente di variazione robusto per non favorire
    feature con scale diverse (radiomica vs genomica).
    """
    X_norm = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    variances = X_norm.var(ddof=0)
    keep = variances[variances > threshold].index

    print(f"[variance_filter] {X.shape[1]} -> {len(keep)} feature (soglia={threshold})")
    return X[keep]


# ---------------------------------------------------------------------------
# RIDUZIONE RIDONDANZA (clustering gerarchico su correlazione)
# ---------------------------------------------------------------------------
def redundancy_reduction(
    X: pd.DataFrame,
    corr_threshold: float = config.REDUNDANCY_CORR_THRESHOLD,
    method: str = config.REDUNDANCY_METHOD,
) -> pd.DataFrame:
    """
    Raggruppa feature altamente correlate tra loro (Spearman di default) e
    tiene un solo rappresentante per cluster (quello con varianza maggiore,
    come proxy di contenuto informativo).

    Questo NON usa la label: è una riduzione di ridondanza strutturale,
    non di rilevanza predittiva.
    """
    corr = X.corr(method=method).abs()
    dist = (1 - corr).copy()
    # rendi la matrice di distanza simmetrica e a diagonale nulla (tolleranza numerica)
    dist_values = dist.values.copy()
    np.fill_diagonal(dist_values, 0)
    dist = pd.DataFrame(dist_values, index=dist.index, columns=dist.columns)
    dist = (dist + dist.T) / 2

    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")

    # distanza di cutoff = 1 - soglia di correlazione
    cluster_ids = fcluster(Z, t=1 - corr_threshold, criterion="distance")

    clusters = pd.Series(cluster_ids, index=X.columns)
    representatives = []
    for cluster_id, members in clusters.groupby(clusters):
        cols = members.index
        if len(cols) == 1:
            representatives.append(cols[0])
        else:
            # rappresentante = feature con varianza massima nel cluster
            best = X[cols].var().idxmax()
            representatives.append(best)

    print(
        f"[redundancy_reduction] {X.shape[1]} -> {len(representatives)} feature "
        f"(clustering su corr {method}, soglia={corr_threshold})"
    )
    return X[representatives]


# ---------------------------------------------------------------------------
# PIPELINE COMPLETA DI RIDUZIONE NEUTRA
# ---------------------------------------------------------------------------
def neutral_feature_reduction(X: pd.DataFrame) -> pd.DataFrame:
    """
    Applica in sequenza variance filter + redundancy reduction.
    Da usare SEMPRE prima di qualsiasi step supervisionato, e da riusare
    identica nello studio di rete per garantire coerenza tra i due task.
    """
    X_var = variance_filter(X)
    X_red = redundancy_reduction(X_var)
    return X_red


if __name__ == "__main__":
    X, y = load_data()
    X_reduced = neutral_feature_reduction(X)
    print(X_reduced.shape)
