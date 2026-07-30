"""
Diagnostica: test di permutazione e controllo di batch effect.

QUESTI DUE CONTROLLI VANNO FATTI PRIMA DI FIDARSI DI QUALUNQUE RISULTATO.

--------------------------------------------------------------------------
TEST DI PERMUTAZIONE 
--------------------------------------------------------------------------
Immagina di scrivere "ADK" o "SCC" su 54 bigliettini e di mescolarli a caso,
come le carte di un mazzo, per poi riattaccarli ai pazienti. Ora l'etichetta
di ogni paziente non ha più niente a che fare con la sua vera malattia.

Se il modello, allenato su queste etichette FINTE, riesce comunque a
"indovinare" bene (AUC alta), vuol dire che il modello sta trovando un
pattern nei dati che non dipende dal tumore — probabilmente rumore, o un
artefatto tecnico. Ripetendo questo mescolamento centinaia di volte,
otteniamo una distribuzione di "quanto in alto può arrivare l'AUC per puro
caso". Se l'AUC ottenuta con le etichette VERE è chiaramente più alta di
quella distribuzione, abbiamo una prova che il modello ha trovato qualcosa
di reale.

--------------------------------------------------------------------------
CONTROLLO DI BATCH EFFECT 
--------------------------------------------------------------------------
Un "batch" è un lotto tecnico: per esempio, se metà dei campioni sono stati
processati in laboratorio in un giorno diverso, con un'altra macchina o un
altro operatore rispetto all'altra metà. Se per sfortuna il batch coincide
anche solo in parte con ADK/SCC, il modello può imparare a riconoscere il
batch (un artefatto) invece del tumore (la cosa che ci interessa davvero).

Qui riassumiamo tutti i pazienti in un grafico a 2 assi (PCA: prende tante
colonne e le "comprime" nei 2 assi che spiegano più variabilità) e li
coloriamo prima per fenotipo, poi (se disponibile) per batch, per vedere ad
occhio se si separano in modo sospetto.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non interattivo, necessario per salvare plot da script
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

import config


# ---------------------------------------------------------------------------
# TEST DI PERMUTAZIONE
# ---------------------------------------------------------------------------
def _cv_auc(X: pd.DataFrame, y_bin: pd.Series, pipe, n_folds: int, random_state: int) -> float:
    """
    Calcola l'AUC media su una k-fold CV semplice (NON nested), usando SEMPRE
    la stessa "ricetta" di iperparametri (niente grid search dentro il ciclo).

    Perché semplificare: rifare la nested CV completa (con grid search) per
    ognuna delle centinaia di permutazioni richiederebbe ore/giorni di
    calcolo. Qui vogliamo solo sapere: "con la ricetta già scelta come
    migliore, il modello riesce a separare le classi anche quando le
    etichette sono casuali?" — per questa domanda una k-fold semplice basta.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs = []
    for train_idx, test_idx in cv.split(X, y_bin):
        pipe.fit(X.iloc[train_idx], y_bin.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y_bin.iloc[test_idx], proba))
    return float(np.mean(aucs))


def permutation_test(X: pd.DataFrame, y: pd.Series, pipe,
                      n_permutations: int = config.N_PERMUTATIONS,
                      n_folds: int = config.PERMUTATION_N_FOLDS,
                      random_state: int = config.RANDOM_STATE):
    """
    Confronta l'AUC ottenuta con le etichette VERE con la distribuzione di
    AUC ottenuta rimescolando le etichette a caso molte volte.

    Parametri
    ---------
    pipe : una pipeline sklearn NON ancora fittata, con predict_proba e con
        iperparametri GIÀ FISSATI (es. quelli scelti come migliori dalla
        nested CV principale in run_analysis.py). Qui non si rifà nessuna
        ricerca di iperparametri.

    Ritorna
    -------
    real_auc : float — AUC ottenuta sui dati veri
    permuted_aucs : np.array — AUC ottenute sulle n_permutations etichette casuali
    p_value : float — frazione di permutazioni con AUC >= a quella reale
        (più è basso, più il risultato vero è difficile da ottenere per caso)
    """
    y_bin = (y == config.POSITIVE_CLASS).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    rng = np.random.RandomState(random_state)

    real_auc = _cv_auc(X, y_bin, pipe, n_folds, random_state)
    print(f"[permutation_test] AUC sui dati veri: {real_auc:.3f}")

    permuted_aucs = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_shuffled = y_bin.sample(frac=1.0, random_state=rng.randint(0, 1_000_000)).reset_index(drop=True)
        permuted_aucs[i] = _cv_auc(X, y_shuffled, pipe, n_folds, random_state=rng.randint(0, 1_000_000))
        if (i + 1) % 50 == 0:
            print(f"[permutation_test] {i+1}/{n_permutations} permutazioni completate")

    # p-value empirico: quante permutazioni hanno fatto MEGLIO o UGUALE ai dati veri
    p_value = (np.sum(permuted_aucs >= real_auc) + 1) / (n_permutations + 1)

    print(f"\n[permutation_test] AUC media con etichette casuali: "
          f"{permuted_aucs.mean():.3f} ± {permuted_aucs.std():.3f}")
    print(f"[permutation_test] p-value empirico: {p_value:.4f}")

    if p_value >= 0.05:
        print("[permutation_test] ATTENZIONE: il modello sui dati veri NON si distingue "
              "in modo significativo da un modello allenato su etichette a caso. "
              "Il segnale trovato dal modello va trattato con forte cautela: "
              "probabilmente non riflette il fenotipo tumorale.")
    else:
        print("[permutation_test] Il modello sui dati veri supera in modo significativo "
              "la distribuzione ottenuta per caso: il segnale sembra reale.")

    return real_auc, permuted_aucs, p_value


def plot_permutation_test(real_auc: float, permuted_aucs: np.ndarray, output_path):
    """Istogramma delle AUC ottenute per caso, con una linea sull'AUC reale."""
    plt.figure(figsize=(7, 5))
    plt.hist(permuted_aucs, bins=20, color="#8C8C8C", edgecolor="white",
              label="AUC con etichette casuali")
    plt.axvline(real_auc, color="#C44E52", linewidth=2,
                label=f"AUC dati veri = {real_auc:.3f}")
    plt.xlabel("AUC")
    plt.ylabel("Numero di permutazioni")
    plt.title("Test di permutazione: il modello batte il caso?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_permutation_test] salvato in {output_path}")


# ---------------------------------------------------------------------------
# CONTROLLO BATCH EFFECT
# ---------------------------------------------------------------------------
def batch_effect_diagnostic(X: pd.DataFrame, y: pd.Series, batch: pd.Series = None,
                             output_path=None):
    """
    Proietta i pazienti su 2 assi principali (PCA) e li colora per fenotipo
    e, se disponibile, per batch tecnico.

    Parametri
    ---------
    batch : pd.Series allineata all'indice di X, con l'etichetta di lotto
        tecnico (es. "run1", "run2"...). Se None, il grafico mostra solo la
        colorazione per fenotipo — utile comunque: se ADK e SCC si separano
        già visibilmente sulle prime 2 componenti, è un segnale (indiretto)
        che vale la pena approfondire da dove viene quella separazione.

    Se batch è fornito, calcola anche un test statistico (Kruskal-Wallis,
    l'equivalente non parametrico dell'ANOVA) su PC1 tra i gruppi di batch:
    un p-value molto basso è un segnale concreto di possibile batch effect.
    """
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    scores = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    has_batch = batch is not None and batch.notna().any()
    n_panels = 2 if has_batch else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    axes = [axes] if n_panels == 1 else list(axes)

    # Pannello 1: colorato per fenotipo
    ax = axes[0]
    for label in y.unique():
        mask = (y == label).values
        ax.scatter(scores[mask, 0], scores[mask, 1], label=str(label), alpha=0.7)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% varianza)")
    ax.set_title("Colorato per fenotipo (ADK/SCC)")
    ax.legend()

    kw_result = None
    if has_batch:
        batch_aligned = batch.reindex(y.index) if hasattr(batch, "reindex") else batch
        valid = batch_aligned.notna().values

        ax2 = axes[1]
        for b in pd.unique(batch_aligned.dropna()):
            mask = (batch_aligned == b).values & valid
            ax2.scatter(scores[mask, 0], scores[mask, 1], label=str(b), alpha=0.7)
        ax2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% varianza)")
        ax2.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% varianza)")
        ax2.set_title("Colorato per batch tecnico")
        ax2.legend()

        groups = [scores[(batch_aligned == b).values & valid, 0]
                  for b in pd.unique(batch_aligned.dropna())]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*groups)
            kw_result = (kw_stat, kw_p)
            print(f"[batch_effect_diagnostic] Kruskal-Wallis PC1 ~ batch: "
                  f"statistica={kw_stat:.3f}, p-value={kw_p:.4f}")
            if kw_p < 0.05:
                print("[batch_effect_diagnostic] ATTENZIONE: il batch spiega in modo "
                      "statisticamente significativo la posizione dei pazienti sulla "
                      "prima componente principale. Possibile batch effect: indaga "
                      "da dove viene questa differenza tecnica prima di fidarti dei "
                      "risultati del modello.")
            else:
                print("[batch_effect_diagnostic] Nessuna evidenza forte di batch effect "
                      "su PC1 con questo test.")
    else:
        print("[batch_effect_diagnostic] Nessuna variabile di batch fornita: "
              "controlla solo visivamente il pannello per fenotipo. Se non hai "
              "informazioni sul lotto tecnico dei tuoi campioni, ti conviene "
              "chiedere al laboratorio se i pazienti ADK e SCC sono stati "
              "processati in date/run diversi.")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[batch_effect_diagnostic] plot salvato in {output_path}")
    plt.close()

    return scores, var_exp, kw_result
