"""
Specification Curve Analysis (SCA) + robustezza delle feature attraverso le
scelte metodologiche ("multiverse analysis").

Perché: con n=54 e diversi gradi di libertà nella pipeline (sorgente dati,
criterio di selezione geni, shape separata o no, soglia di ridondanza),
riportare un risultato "migliore" nasconde quanto quel risultato dipenda
dalle scelte fatte. Qui enumeriamo esplicitamente le combinazioni
ragionevoli e guardiamo (a) come cambia la performance, (b) quali feature
emergono come importanti in modo consistente attraverso le specifiche.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import config
import data_utils

# ---------------------------------------------------------------------------
# SPAZIO DELLE SPECIFICHE — modifica/estendi liberamente
# ---------------------------------------------------------------------------
SPEC_GRID = {
    "data_source": ["radiomics", "genomics", "both"],
    "gene_selection_method": ["variance", "iqr_top_pct", "iqr_top_n"],
    "exclude_shape": [True, False],
    "redundancy_corr_threshold": [0.80, 0.90, 0.95],
}


def _fixed_pipe(C=0.01, l1_ratio=0.1):
    """
    Elastic Net con iperparametri fissi, uguali per ogni specifica.
    Niente grid search qui dentro apposta: vogliamo isolare l'effetto delle
    scelte di preprocessing sul risultato, non confonderlo con instabilità
    da tuning (run_analysis.py).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                    max_iter=5000, C=C, l1_ratio=l1_ratio,
                                    random_state=config.RANDOM_STATE)),
    ])


def _cv_auc_and_coefs(X, y_bin, pipe, n_folds=5, random_state=config.RANDOM_STATE):
    """AUC media in k-fold semplice + coefficienti medi (per il ranking feature)."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs, coef_frames = [], []
    for train_idx, test_idx in cv.split(X, y_bin):
        pipe.fit(X.iloc[train_idx], y_bin.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y_bin.iloc[test_idx], proba))
        coef_frames.append(pd.Series(pipe.named_steps["clf"].coef_.ravel(), index=X.columns))
            # series is one-dimensional labeled array. prende lo step "clf" della pipe (quindi 
            # il modello in se, no standardizzazione), ne prende i coeff imparati per ogni feature
            # e li rende 1d, poi crea la series, in modo che il fatto che sia labeled indichi che 
            # ogni coeff sia associato alla corrispondente feature.
    mean_coefs = pd.concat(coef_frames, axis=1).mean(axis=1)
        # concateno i coeff lungo l'asse 1, ovvero lungo le diverse fold, e poi faccio la media dei
        # valori sui diversi fold, in modo da avere un singolo valore di coeff ottenuto dalla media 
        # di n_folds valori, uno per fold.
    return float(np.mean(aucs)), float(np.std(aucs)), mean_coefs
        # restituisce auc non pooled, ma mediata sui n_folds fold.


def run_specification_curve(spec_grid=None, n_folds=5, top_n_features=15):
    """
    Ritorna:
    spec_df       -> una riga per ogni combinazione di specifiche, con AUC e n_features
    feature_votes -> Series feature -> numero di specifiche in cui è entrata nella top_n
    """
    spec_grid = spec_grid or SPEC_GRID
    keys = list(spec_grid.keys())
        # sono "data_source", "gene_selection_method", "exclude_shape", "redundancy_corr_threshold".
    combos = list(itertools.product(*spec_grid.values()))
        # per generare tutte le combinazioni possibili tra insiemi di opzioni differenti 
        # prende tutti i valori di spec grid (spec_grid.values()), li separa (*), ne fa prodotto
        # cartesiano creando tutte le combinazioni (itertools.product) e li trasforma in lista (list()).

    cache_raw = {}  # evita di ricaricare da disco la stessa sorgente più volte
        # cache, un dizionario temporaneo usato per evitare di fare la stessa operazione più volte.
    rows = []
    feature_votes = pd.Series(dtype=float)

    for combo in combos:
        spec = dict(zip(keys, combo)) # stessa dim perchè uso specifica combinazione (combo, not combos)
            # poi le trasforma in dict con key e valore
        print(f"[specification_curve] specifica: {spec}")

        if spec["data_source"] not in cache_raw:
            cache_raw[spec["data_source"]] = data_utils.load_data(source=spec["data_source"])
                # carico i dati in cache, così ogni volta che carico radiomics, genomics o both non devo
                # ripetere l'operazione di load, ma riuso quelli salvati in cache risparmiando tempo e
                # memoria
        X_raw, y = cache_raw[spec["data_source"]]

        X_reduced = data_utils.neutral_feature_reduction(
            X_raw,
            gene_selection_method=spec["gene_selection_method"],
            exclude_shape=spec["exclude_shape"],
            redundancy_corr_threshold=spec["redundancy_corr_threshold"],
        )

        y_bin = (y == config.POSITIVE_CLASS).astype(int)
        pipe = _fixed_pipe()
        auc_mean, auc_sd, mean_coefs = _cv_auc_and_coefs(X_reduced, y_bin, pipe, n_folds)

        top_feats = mean_coefs.abs().sort_values(ascending=False).head(top_n_features).index
        feature_votes = feature_votes.add(pd.Series(1.0, index=top_feats), fill_value=0)

        rows.append({**spec, "auc_mean": auc_mean, "auc_sd": auc_sd, "n_features": X_reduced.shape[1]})

    spec_df = pd.DataFrame(rows).sort_values("auc_mean").reset_index(drop=True)
    feature_votes = feature_votes.sort_values(ascending=False)

    print(f"\n[specification_curve] {len(spec_df)} specifiche testate | "
          f"AUC min={spec_df['auc_mean'].min():.3f} max={spec_df['auc_mean'].max():.3f}")

    return spec_df, feature_votes


# ---------------------------------------------------------------------------
# PLOT — curva di specificazione classica (pannello AUC + pannello scelte)
# ---------------------------------------------------------------------------
def plot_specification_curve(spec_df: pd.DataFrame, spec_keys: list, output_path):
    n_spec = len(spec_df)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(max(10, n_spec * 0.25), 8),
        gridspec_kw={"height_ratios": [2, 1.5]}, sharex=True
    )

    # pannello superiore: AUC ordinata, con barre di errore (sd tra i fold)
    x = np.arange(n_spec)
    ax_top.errorbar(x, spec_df["auc_mean"], yerr=spec_df["auc_sd"],
                     fmt="o", color="#4C72B0", ecolor="#A6C8FF", markersize=4, capsize=2)
    ax_top.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="AUC=0.5 (caso)")
    ax_top.set_ylabel("AUC")
    ax_top.set_title("Specification curve: AUC su tutte le combinazioni di preprocessing")
    ax_top.legend()

    # pannello inferiore: quale scelta è attiva per ogni colonna, per ogni parametro
    colors = plt.cm.tab10.colors
    y_positions = {}
    row_offset = 0
    for key in spec_keys:
        levels = sorted(spec_df[key].unique(), key=str)
        for level in levels:
            y_positions[(key, level)] = row_offset
            row_offset += 1
        row_offset += 0.5  # spazio tra gruppi di parametri

    for xi, (_, row) in enumerate(spec_df.iterrows()):
        for key in spec_keys:
            yi = y_positions[(key, row[key])]
            ax_bottom.scatter(xi, yi, color="#4C72B0", s=15)

    ax_bottom.set_yticks(list(y_positions.values()))
    ax_bottom.set_yticklabels([f"{k}={v}" for (k, v) in y_positions.keys()], fontsize=8)
    ax_bottom.set_xlabel("Specifiche ordinate per AUC crescente")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_specification_curve] salvato in {output_path}")


def plot_feature_votes(feature_votes: pd.Series, n_total_specs: int, output_path, top_n=25):
    top = (feature_votes / n_total_specs * 100).head(top_n).sort_values()
    plt.figure(figsize=(8, 0.3 * len(top) + 1))
    plt.barh(top.index, top.values, color="#55A868")
    plt.xlabel("% di specifiche in cui la feature è tra le più importanti")
    plt.title("Robustezza delle feature attraverso le scelte metodologiche")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_feature_votes] salvato in {output_path}")