"""
Test di DeLong per il confronto tra due AUC correlate (stessi pazienti,
due modelli/sorgenti dati diverse).

Riferimento: DeLong, DeLong & Clarke-Pearson (1988); implementazione
efficiente O(n log n) da Sun & Xu (2014), "Fast Implementation of DeLong's
Algorithm for Comparing the Areas Under Correlated Receiver Operating
Characteristic Curves".

A differenza di un bootstrap, qui la varianza della differenza tra le due
AUC è derivata analiticamente dalla teoria delle U-statistics (la stessa
base teorica dell'AUC stessa), non stimata per ricampionamento: niente
seed, niente scelta del numero di iterazioni, risultato deterministico.

L'AUC misura quanto spesso il modello assegna uno score maggiore a un positivo 
rispetto a un negativo scelti casualmente.
"""

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank con gestione dei pari merito (ties), richiesto dall'algoritmo di DeLong."""
    J = np.argsort(x) # Returns the indices that would sort an array.
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """
    predictions_sorted_transposed : array (k_modelli, n_pazienti), con i
        pazienti ORDINATI in modo che i primi label_1_count siano la classe
        positiva (ADK) e i restanti la classe negativa (SCC).
    Ritorna: aucs (per ogni modello), delongcov (matrice di covarianza k x k)
    """
    m = label_1_count # num paz pos
    n = predictions_sorted_transposed.shape[1] - m # num paz neg
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float) # k modelli, m paz positivi
    ty = np.empty([k, n], dtype=float) # k modelli, n paz negativi
    tz = np.empty([k, m + n], dtype=float) # k modelli, pazienti totali
    for r in range(k): # risolvo pareggi di rank 
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n 
        # formula auc usando rank, descrive probab che score di positivo sia 
        # maggiore di negativo
    v01 = (tz[:, :m] - tx[:, :]) / n # quanto bene un positivo viene classificato
        # rispetto a tutti i negativi? misura quanto ogni positivo contribuisce all’AUC
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m # uguale, ma al contrario
    sx = np.cov(v01) # matrice cov dei contributi positivi, per due modelli ho quanto
        # varia il contributo (VAR) del modello A, del modello B e quanto sono
        # correlati i contributi
    sy = np.cov(v10) # matrice cov dei contributi negativi
    delongcov = sx / m + sy / n # matrice cov delle auc, divido per m e per n xk 
        # var prop a 1/num_dati
    return aucs, delongcov


def delong_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray):
    """
    Confronta AUC(proba_b) vs AUC(proba_a) sugli stessi pazienti (y_true
    condiviso). Ritorna un dizionario con AUC di entrambi, la differenza,
    il suo errore standard, un CI 95% in forma chiusa (normale asintotica,
    non percentile-bootstrap), z-score e p-value a due code.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)  # positivi (1) prima, negativi (0) dopo
    label_1_count = int(y_true.sum())

    preds_sorted = np.vstack([proba_a, proba_b])[:, order] # Stack arrays 
        # in sequence vertically (row wise).
    aucs, cov = _fast_delong(preds_sorted, label_1_count)

    auc_a, auc_b = aucs[0], aucs[1]
    diff = auc_b - auc_a
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se_diff = np.sqrt(max(var_diff, 0))  # max(...,0) per sicurezza numerica

    z = diff / se_diff if se_diff > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z))) # computes the cumulative 
        # distribution function (CDF) for a normal distribution
        # teoria asintotica di DeLong porta a trattare, approssimativamente, z 
        # come distribuito normale
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "auc_a": auc_a, "auc_b": auc_b, "diff": diff,
        "se_diff": se_diff, "ci_low": ci_low, "ci_high": ci_high,
        "z": z, "p_value": p_value,
    }