"""
Specification Curve Analysis (SCA) + robustezza delle feature attraverso le
scelte metodologiche ("multiverse analysis").

Per ogni combinazione di preprocessing (data_source, selezione geni, shape,
soglia di ridondanza) e per due modelli (Elastic Net lineare, Random Forest)
calcoliamo:
- AUC media sui fold (con sd) e AUC pooled out-of-fold (più robusta con n
  piccolo: non è la media di 5 stime rumorose ma un'unica stima su tutti i
  54 pazienti concatenando le predizioni out-of-fold)
- statistiche complete sull'importanza di ogni feature attraverso i fold di
  quella specifica (non solo la media, che con oscillazioni di segno può
  nascondere una feature realmente importante ma incoerente in direzione)
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
from sklearn.ensemble import RandomForestClassifier

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

# i "due best" modelli: lineare interpretabile via coefficienti, e ad
# albero interpretabile via feature_importances_. Iperparametri FISSI per
# ogni specifica (niente grid search qui dentro, vedi nota nella docstring
# del modulo precedente: isoliamo l'effetto del preprocessing).
MODEL_TYPES = ["linear", "tree"]


def _build_pipe(model_type: str):
    if model_type == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=5000,
                C=0.1, l1_ratio=0.3, random_state=config.RANDOM_STATE
            )),
        ])
    elif model_type == "tree":
        return Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_leaf=3,
                random_state=config.RANDOM_STATE
            )),
        ])
    else:
        raise ValueError(f"model_type '{model_type}' non valido (usa 'linear' o 'tree')")


def _extract_importance(fitted_pipe, model_type: str, columns) -> pd.Series:
    clf = fitted_pipe.named_steps["clf"]
    if model_type == "linear":
        return pd.Series(clf.coef_.ravel(), index=columns)
    else:
        # feature_importances_ del Random Forest sono sempre >= 0 (non hanno segno,
        # misurano riduzione di impurità), a differenza dei coefficienti lineari
        return pd.Series(clf.feature_importances_, index=columns)


def _cv_eval(X: pd.DataFrame, y_bin: pd.Series, model_type: str,
             n_folds: int = 5, random_state: int = config.RANDOM_STATE):
    """
    k-fold CV semplice per UNA specifica + UN modello. Ritorna:
    - auc_mean, auc_sd: media/sd dell'AUC calcolata fold per fold (come prima)
    - auc_pooled: AUC su tutte le predizioni out-of-fold concatenate — con
      n=54 diviso in 5 fold (~11 pazienti a fold) la stima fold-level ha
      varianza enorme; la versione pooled è la stima più stabile da guardare
    - coef_matrix: DataFrame (feature x fold) con l'importanza per fold,
      base per le statistiche del punto 2
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs = []
    coef_frames = []
    oof_proba = np.full(len(X), np.nan)

    for train_idx, test_idx in cv.split(X, y_bin):
        pipe = _build_pipe(model_type)
        pipe.fit(X.iloc[train_idx], y_bin.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]

        aucs.append(roc_auc_score(y_bin.iloc[test_idx], proba))
        oof_proba[test_idx] = proba
        coef_frames.append(_extract_importance(pipe, model_type, X.columns))

    auc_pooled = roc_auc_score(y_bin, oof_proba)
    coef_matrix = pd.concat(coef_frames, axis=1)
    coef_matrix.columns = [f"fold_{i}" for i in range(coef_matrix.shape[1])]

    return float(np.mean(aucs)), float(np.std(aucs)), float(auc_pooled), coef_matrix


def _feature_stats(coef_matrix: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """
    Statistiche sull'importanza di ogni feature attraverso i fold di UNA
    specifica. La sola media può andare vicino a zero per una feature che
    oscilla di segno tra fold pur essendo sistematicamente "usata" dal
    modello (es. +0.8, -0.7, +0.9, -0.6, +0.75 -> media bassa ma
    fraction_nonzero=100%, std alta): mean_abs e fraction_nonzero
    catturano questo caso, mean_coefficient da solo no.
    """
    stats = pd.DataFrame({
        "mean_coefficient": coef_matrix.mean(axis=1),
        "std_coefficient": coef_matrix.std(axis=1),
        "mean_abs_coefficient": coef_matrix.abs().mean(axis=1),
        "fraction_nonzero": (coef_matrix.abs() > 1e-8).mean(axis=1),
    })
    if model_type == "linear":
        stats["fraction_positive"] = (coef_matrix > 1e-8).mean(axis=1)
        stats["fraction_negative"] = (coef_matrix < -1e-8).mean(axis=1)
    else:
        # Random Forest: le importances non hanno segno, quindi la
        # distinzione positive/negative non esiste per questo modello
        stats["fraction_positive"] = np.nan
        stats["fraction_negative"] = np.nan
    return stats


def run_specification_curve(spec_grid=None, model_types=None,
                             n_folds: int = 5, top_n_features: int = 15):
    """
    Ritorna
    -------
    spec_df : una riga per ogni combinazione (specifica x modello), con
        auc_mean_fold, auc_sd_fold, auc_pooled, n_features
    feature_votes : Series feature -> numero di combinazioni (specifica x
        modello, TUTTI i modelli insieme) in cui la feature è tra le top_n
    feature_votes_by_model : dict {model_type: Series}, stesso conteggio
        ma separato per modello (utile perché lineare e RF possono
        enfatizzare feature diverse per motivi strutturali, non è detto
        che "unire" i voti sia sempre la lettura giusta)
    feature_stats_long_df : DataFrame in formato lungo con TUTTE le
        statistiche del punto 2, per ogni specifica e ogni modello —
        salvalo, è la base per qualsiasi controllo più fine dopo
    """
    spec_grid = spec_grid or SPEC_GRID
    model_types = model_types or MODEL_TYPES
    keys = list(spec_grid.keys())
    combos = list(itertools.product(*spec_grid.values()))

    cache_raw = {}
    spec_rows = []
    feature_votes = pd.Series(dtype=float)
    feature_votes_by_model = {m: pd.Series(dtype=float) for m in model_types}
    feature_stats_long = []

    for combo in combos:
        spec = dict(zip(keys, combo))

        if spec["data_source"] not in cache_raw:
            cache_raw[spec["data_source"]] = data_utils.load_data(source=spec["data_source"])
        X_raw, y = cache_raw[spec["data_source"]]

        X_reduced = data_utils.neutral_feature_reduction(
            X_raw,
            gene_selection_method=spec["gene_selection_method"],
            exclude_shape=spec["exclude_shape"],
            redundancy_corr_threshold=spec["redundancy_corr_threshold"],
        )
        y_bin = (y == config.POSITIVE_CLASS).astype(int)

        for model_type in model_types:
            print(f"[specification_curve] {spec} | model={model_type}")
            auc_mean, auc_sd, auc_pooled, coef_matrix = _cv_eval(
                X_reduced, y_bin, model_type, n_folds
            )
            fstats = _feature_stats(coef_matrix, model_type)

            top_feats = fstats["mean_abs_coefficient"].sort_values(
                ascending=False
            ).head(top_n_features).index
            feature_votes = feature_votes.add(pd.Series(1.0, index=top_feats), fill_value=0)
            feature_votes_by_model[model_type] = feature_votes_by_model[model_type].add(
                pd.Series(1.0, index=top_feats), fill_value=0
            )

            fstats_named = fstats.reset_index().rename(columns={"index": "feature"})
            for k, v in spec.items():
                fstats_named[k] = v
            fstats_named["model_type"] = model_type
            feature_stats_long.append(fstats_named)

            spec_rows.append({
                **spec, "model_type": model_type,
                "auc_mean_fold": auc_mean, "auc_sd_fold": auc_sd,
                "auc_pooled": auc_pooled, "n_features": X_reduced.shape[1],
            })

    spec_df = pd.DataFrame(spec_rows).sort_values("auc_pooled").reset_index(drop=True)
    feature_votes = feature_votes.sort_values(ascending=False)
    for m in feature_votes_by_model:
        feature_votes_by_model[m] = feature_votes_by_model[m].sort_values(ascending=False)
    feature_stats_long_df = pd.concat(feature_stats_long, ignore_index=True)

    print(f"\n[specification_curve] {len(spec_df)} combinazioni (specifica x modello) testate | "
          f"AUC pooled min={spec_df['auc_pooled'].min():.3f} max={spec_df['auc_pooled'].max():.3f}")

    return spec_df, feature_votes, feature_votes_by_model, feature_stats_long_df


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
def plot_specification_curve(spec_df: pd.DataFrame, spec_keys: list, output_path):
    """
    spec_keys deve includere anche "model_type" se vuoi vederlo nel pannello
    inferiore insieme alle altre scelte di preprocessing.
    """
    spec_df = spec_df.sort_values("auc_pooled").reset_index(drop=True)
    n_spec = len(spec_df)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(max(10, n_spec * 0.22), 8),
        gridspec_kw={"height_ratios": [2, 1.8]}, sharex=True
    )
    x = np.arange(n_spec)

    model_colors = {"linear": "#4C72B0", "tree": "#C44E52"}
    for model_type, color in model_colors.items():
        mask = (spec_df["model_type"] == model_type).values
        ax_top.errorbar(x[mask], spec_df.loc[mask, "auc_mean_fold"],
                         yerr=spec_df.loc[mask, "auc_sd_fold"],
                         fmt="o", color=color, alpha=0.35, markersize=3, capsize=2,
                         label=f"{model_type}: media fold ± sd")
        ax_top.scatter(x[mask], spec_df.loc[mask, "auc_pooled"],
                        color=color, marker="D", s=20, label=f"{model_type}: pooled OOF")

    ax_top.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="AUC=0.5 (caso)")
    ax_top.set_ylabel("AUC")
    ax_top.set_title("Specification curve: AUC media-fold vs pooled OOF, per specifica e modello")
    ax_top.legend(fontsize=8, ncol=2)

    colors = plt.cm.tab10.colors
    y_positions = {}
    row_offset = 0
    for key in spec_keys:
        levels = sorted(spec_df[key].astype(str).unique(), key=str)
        for level in levels:
            y_positions[(key, level)] = row_offset
            row_offset += 1
        row_offset += 0.5

    for xi, (_, row) in enumerate(spec_df.iterrows()):
        for key in spec_keys:
            yi = y_positions[(key, str(row[key]))]
            ax_bottom.scatter(xi, yi, color="#4C72B0", s=15)

    ax_bottom.set_yticks(list(y_positions.values()))
    ax_bottom.set_yticklabels([f"{k}={v}" for (k, v) in y_positions.keys()], fontsize=8)
    ax_bottom.set_xlabel("Specifiche ordinate per AUC pooled crescente")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_specification_curve] salvato in {output_path}")


def plot_feature_votes(feature_votes: pd.Series, n_total_specs: int, output_path, top_n: int = 25):
    top = (feature_votes / n_total_specs * 100).head(top_n).sort_values()
    plt.figure(figsize=(8, 0.3 * len(top) + 1))
    plt.barh(top.index, top.values, color="#55A868")
    plt.xlabel("% di combinazioni in cui la feature è tra le più importanti")
    plt.title("Robustezza delle feature attraverso le scelte metodologiche")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_feature_votes] salvato in {output_path}")