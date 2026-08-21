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
"""

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank con gestione dei pari merito (ties), richiesto dall'algoritmo di DeLong."""
    J = np.argsort(x)
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
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
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

    preds_sorted = np.vstack([proba_a, proba_b])[:, order]
    aucs, cov = _fast_delong(preds_sorted, label_1_count)

    auc_a, auc_b = aucs[0], aucs[1]
    diff = auc_b - auc_a
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se_diff = np.sqrt(max(var_diff, 0))  # max(...,0) per sicurezza numerica

    z = diff / se_diff if se_diff > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "auc_a": auc_a, "auc_b": auc_b, "diff": diff,
        "se_diff": se_diff, "ci_low": ci_low, "ci_high": ci_high,
        "z": z, "p_value": p_value,
    }