"""
Caricamento dati e riduzione feature INDIPENDENTE dalla label.

Principio chiave: qui NON si guarda mai la variabile target. Questo step serve
a togliere ridondanza e rumore prima ancora di pensare alla classificazione,
in modo che lo stesso set ridotto possa essere riusato coerentemente sia per
il modello ML sia per lo studio di rete, senza circolarità statistica.
"""

import numpy as np
import pandas as pd
import re
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import config


# ---------------------------------------------------------------------------
# CARICAMENTO
# ---------------------------------------------------------------------------
def normalize_id(patient_id):
    if pd.isna(patient_id): # controlla se la casella indicata sia vuota (NaN, None,..)  
                            # e restituisce un bool
        return None
    patient_str = str(patient_id) # converte qualsiasi tipo di dato in stringa
    match = re.search(r'\d+', patient_str) # usa Regular Expression, search scorre la stringa 
                                           # da sx a dx e cattura il primo blocco di numeri, 
                                           # dettato da "\d" che significa "cerca qualsiasi 
                                           # cifra da 0 a 9", e "+" significa "prendi anche 
                                           # tutte quelle consecutive"
    return match.group(0) if match else patient_str.strip()
    # match.group(0) ritorna la completa frase che ha matchato la ricerca
    # .strips() rimuove spazi bianchi e gap nel testo, restituendo una stringa più pulita possibile

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
    labels_raw = pd.read_excel(config.LABELS_PATH, index_col=0).T # usa la colonna 0 come indice di 
                                                                  # riga per il dataframe
    labels_raw.columns = labels_raw.columns.str.strip() # labels delle colonne del dataframe, 
                                                        # stringhe senza spazi e margini ai lati
    
    # Applichiamo la normalizzazione e mappiamo le classi (ADK=0, SCC=1)
    labels_raw['ID_Normalizzato'] = labels_raw[config.ID_COL].apply(normalize_id)
    # Mantiene il testo originale (es. "ADK", "SCC"), rimuovendo eventuali spazi invisibili
    labels_raw['Target'] = labels_raw[config.LABEL_COL].astype(str).str.strip()
    
    # Creiamo la Series finale pulita
    y_all = labels_raw.set_index('ID_Normalizzato')['Target'].dropna() # dropna() rimuove ogni elemento
        # con valori mancanti (NaN, None). set_index usa per il target il label dato da ID_Normalizzato

    frames = []

    # 2. Caricamento e pulizia RADIOMICA
    if source in ("radiomics", "both"):
        rad = pd.read_csv(config.RADIOMICS_PATH, index_col=0) # usa la prima colonna del file
            # direttamente come indice della tabella, anziché creare una colonna numerica generica 
        # Sostituzione virgole e cast a float
        rad = rad.replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce') # sostituisce 
            # ogni virgola in punto e il contenuto di ogni cella in un numero, mettendo NaN in caso
            # di testo strano o errore di estrazione
        # Normalizzazione indici e rinomina colonne
        rad.index = [normalize_id(idx) for idx in rad.index]
        rad.columns = [f"rad__{c}" for c in rad.columns] # aggiunge il prefisso rad__ a tutti i nomi 
            # delle colonne radiomiche, così da distinguerle da quelle genomiche se both usate
        frames.append(rad)

    # 3. Caricamento e pulizia GENOMICA
    if source in ("genomics", "both"):
        gen = pd.read_excel(config.GENOMICS_PATH, index_col=0)
        # Rimuoviamo le ultime due colonne non utili e trasponiamo (pazienti sulle righe)
        gen = gen.iloc[:, :-2].T # iloc seleziona colonne o righe in base alla loro posizione
        # Normalizzazione indici e rinomina colonne
        gen.index = [normalize_id(idx) for idx in gen.index]
        gen.columns = [f"gen__{c}" for c in gen.columns]
        frames.append(gen)

    if not frames:
        raise ValueError(f"Parametro 'source' non valido: {source}")

    # 4. Concatenazione dei blocchi feature
    X_all = pd.concat(frames, axis=1, join="inner") # concatena oggeti df lungo direzione 1 (orizz)
        # join='inner' takes the intersection of the axis values. quindi concatena in orizzontale, 
        # fianco a fianco, prendendo intersezione assi evitando di trascinarsi dietro valori nulli. 
        # intersezione viene presa sull'asse dei pazienti, selezionati solo paz comuni

    # 5. Inner Join finale tra Feature (X) e Target (y)
    common_idx = X_all.index.intersection(y_all.index) # trova indici (paz) presenti sia in X che in Y
    X = X_all.loc[common_idx].sort_index() # sotto-seleziona da X e y quei pazienti in comune, li ordina
        # secondo l'ordine dei paz (both X e y)
    y = y_all.loc[common_idx].sort_index()

    # Controllo di integrità
    assert X.shape[0] == y.shape[0], "Errore critico: Disallineamento tra feature e label dopo il merge."
        # esatto numero di paz per X e y
    
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
    X_norm = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan) # centra ogni feature rispetto alla media
        # (sottrae media di colonna a ciascun valore), divide per std calcolata con divisione per 
        # (N-ddof) con N num di elementi. se feature cost, dev nulla, quindi evito divisione per 0
        # rimpiazzando 0 di dev std con NaN. Produce colonna di NaN. Scalo dati prima di calcolare Var 
        # per non avere calcolo su scale numeriche diverse.
    variances = X_norm.var(ddof=0) # calcola varianza, dividendo per N - ddof, usually 1.
    keep = variances[variances > threshold].index # confronto var con soglia, e ne tengo gli indici

    print(f"[variance_filter] {X.shape[1]} -> {len(keep)} feature (soglia={threshold})")
    return X[keep] # ritorna la matrice di input solo nelle colonne corrispondenti agli indici calc
                   # precedentemente


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
    corr = X.corr(method=method).abs() # calcola correlazione di colonne, pairwise. salva i moduli. 
        # crea matrice di correlazione
    dist = (1 - corr).copy() # introduce concetto di distanza: se alta correlazione, bassa distanza
    # rendi la matrice di distanza simmetrica e a diagonale nulla (tolleranza numerica)
    dist_values = dist.values.copy() # estrae valori di distanza 
    np.fill_diagonal(dist_values, 0) # riempie diagonale di 0 
    dist = pd.DataFrame(dist_values, index=dist.index, columns=dist.columns)
    dist = (dist + dist.T) / 2 # rende perfettamente simmetrica la matrice/df delle distanze

    condensed = squareform(dist.values, checks=False) # trasforma la matrice in un vettore, no check
        # su diagonale nulla o simmetria, contiene solo elem matrice triang sup
    Z = linkage(condensed, method="average") # Costruisce un "albero genealogico" (dendrogramma) 
        # delle feature raggruppando quelle più vicine, calcolando la distanza tra gruppi come 
        # media delle distanze delle singole coppie (distanza gruppo-feature è media delle distanze
        # tra ogni elemento del gruppo e la feature)

    # distanza di cutoff = 1 - soglia di correlazione
    cluster_ids = fcluster(Z, t=1 - corr_threshold, criterion="distance") # Tutte le feature che si 
        # trovano a una distanza reciproca inferiore a 1-soglia vengono assegnate allo stesso 
        # gruppo/cluster. restituisce un array con gli id (per ogni feature) del cluster di appartenenza

    clusters = pd.Series(cluster_ids, index=X.columns) # series è un array 1d. il valore è l'ID del 
        # cluster, l'indice è il nome della feature
    representatives = []
    for cluster_id, members in clusters.groupby(clusters): # per ogni id di cluster e membro
        cols = members.index # cerco gli index dei membri del cluster
        if len(cols) == 1: # se solo un elemento nel cluster, lo salvo
            representatives.append(cols[0])
        else: # se più elementi, salvo solo quello con varianza maggiore
            # rappresentante = feature con varianza massima nel cluster
            best = X[cols].var().idxmax()
            representatives.append(best)

    print(
        f"[redundancy_reduction] {X.shape[1]} -> {len(representatives)} feature "
        f"(clustering su corr {method}, soglia={corr_threshold})"
    )
    return X[representatives]

# .groupby() su un oggetto indicando se stesso come argomento (clusters.groupby(clusters)), raggruppa 
# la Series in base ai suoi stessi valori. Series.groupby group Series using a mapper. un ciclo for 
# su un groupby, restituisce ad ogni iterazione una coppia di elementi:  cluster_id, la chiave del 
# gruppo (l'ID del cluster), e .members: La sotto-Series contenente solo le righe che appartengono 
# a quel gruppo.

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
