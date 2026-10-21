"""
Specification curve v2 (descrittiva): confronta le performance di diversi
metodi di riduzione feature, sulle stesse sorgenti dati e con gli stessi modelli
della specification curve esistente. File NUOVO: specification_curve.py non
viene modificato; ne vengono riusati _cv_eval e (nel runner) il plot.

Assi della curva
----------------
- data_source    : radiomics / genomics / both
- reduction_mode : "legacy" = data_utils.neutral_feature_reduction (versione
                   precedente, rappresentante a varianza grezza);
                   "medoid" / "pca" = reduction_v2.BlockReducer
- budget         : "thr_0.90", "thr_0.50", "thr_0.25" = taglio per soglia di
                   correlazione; "k_small" / "k_large" = numero fisso di
                   feature per blocco (vedi BUDGETS). I tagli a k esistono solo
                   per medoid/pca (il legacy non li supporta).
- model_type     : linear (Elastic Net) / tree (Random Forest), come in
                   specification_curve.MODEL_TYPES

È una curva puramente descrittiva: nessun test di permutazione e nessuna
selezione della "specifica migliore". Come nella curva esistente, la riduzione
non usa la label e viene calcolata sull'intero dataset, mentre l'AUC viene da
una k-fold CV semplice con iperparametri scelti dentro i fold (LogisticRegressionCV).
"""

import itertools

import numpy as np
import pandas as pd

import config
import data_utils
import reduction_v2
import specification_curve as sc

# ---------------------------------------------------------------------------
# SPAZIO DELLE SPECIFICHE — modifica/estendi liberamente
# ---------------------------------------------------------------------------
SPEC_GRID_V2 = {
    "data_source": ["radiomics", "genomics", "both"],
    "reduction_mode": ["legacy", "medoid", "pca"],
    "budget": ["thr_0.90", "thr_0.50", "thr_0.25", "k_small", "k_large"],
}

# Definizione di ogni livello dell'asse "budget".
# "thr" = soglia di correlazione (per legacy e per i tagli a soglia di BlockReducer);
# le chiavi k_* valgono solo per BlockReducer (None = taglio a soglia).
BUDGETS = {
    "thr_0.90": dict(thr=0.90, k_texture=None, k_shape=None, k_gen=None),
    "thr_0.50": dict(thr=0.50, k_texture=None, k_shape=None, k_gen=None),
    "thr_0.25": dict(thr=0.25, k_texture=None, k_shape=None, k_gen=None),
    "k_small":  dict(thr=None, k_texture=4, k_shape=1, k_gen=5),
    "k_large":  dict(thr=None, k_texture=8, k_shape=2, k_gen=8),
}


def _is_valid_spec(spec: dict) -> bool:
    """Il metodo legacy non ha un taglio a k fisso."""
    return not (spec["reduction_mode"] == "legacy" and spec["budget"].startswith("k_"))


def _reduce(X_raw: pd.DataFrame, spec: dict) -> pd.DataFrame:
    b = BUDGETS[spec["budget"]]
    if spec["reduction_mode"] == "legacy":
        return data_utils.neutral_feature_reduction(
            X_raw, redundancy_corr_threshold=b["thr"], print_info=False
        )
    reducer = reduction_v2.BlockReducer(
        k_texture=b["k_texture"], k_shape=b["k_shape"], k_gen=b["k_gen"],
        mode=spec["reduction_mode"],
        corr_threshold=b["thr"] if b["thr"] is not None else 0.9,
        separate_shape=config.EXCLUDE_SHAPE_FROM_MAIN_REDUCTION,
        gen_prefilter_pct=config.GENE_IQR_TOP_PCT,
    )
    return reducer.fit(X_raw).transform(X_raw)


def run_specification_curve_v2(spec_grid=None, model_types=None, n_folds: int = 5,
                               n_repeats: int = 1, random_state: int = config.RANDOM_STATE,
                               verbose: bool = True, n_jobs_model: int = -1) -> pd.DataFrame:
    """
    Ritorna spec_df: una riga per (specifica x modello) con
        auc_mean_fold, auc_sd_fold   : media / sd dell'AUC fold per fold
        auc_pooled                   : AUC sulle predizioni out-of-fold concatenate
        auc_pooled_sd_repeats        : sd di auc_pooled tra le ripetizioni (NaN se n_repeats=1)
        n_features, n_features_rad, n_features_gen
        events_per_feature           : pazienti della classe minoritaria / n_features
        collapsed_model              : True se il modello non ha selezionato nessuna feature

    n_repeats : ripete la CV con seed diversi (random_state + r) e media i
        risultati. Con n=54 la stima da un singolo split è rumorosa: se le
        differenze tra specifiche sono piccole, n_repeats>1 le rende leggibili.
        Default 1 = stesso comportamento della curva esistente.
    """
    spec_grid = spec_grid or SPEC_GRID_V2
    model_types = model_types or sc.MODEL_TYPES
    keys = list(spec_grid.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*spec_grid.values())]
    combos = [s for s in combos if _is_valid_spec(s)]

    cache_raw, rows = {}, []
    for spec in combos:
        if spec["data_source"] not in cache_raw:
            cache_raw[spec["data_source"]] = data_utils.load_data(
                source=spec["data_source"], print_info=False)
        X_raw, y = cache_raw[spec["data_source"]]
        y_bin = (y == config.POSITIVE_CLASS).astype(int)
        n_min = int(min(y_bin.sum(), len(y_bin) - y_bin.sum()))

        X_red = _reduce(X_raw, spec)
        n_rad = sum(c.startswith("rad__") for c in X_red.columns)
        n_gen = sum(c.startswith("gen__") for c in X_red.columns)

        for model_type in model_types:
            if verbose:
                print(f"[specification_curve_v2] {spec} | model={model_type} "
                      f"| {X_red.shape[1]} feature")
            runs = [sc._cv_eval(X_red, y_bin, model_type, n_folds,
                                random_state=random_state + r, n_jobs_model=n_jobs_model)
                    for r in range(n_repeats)]
            pooled = [r[2] for r in runs]
            collapsed = any(bool(((r[3].abs() > 1e-8).mean(axis=1) == 0).all()) for r in runs)
            if collapsed:
                print(f"[specification_curve_v2] ATTENZIONE: modello collassato (nessuna "
                      f"feature selezionata) per {spec} model={model_type}: AUC non "
                      f"riflette segnale.")
            rows.append({
                **spec, "model_type": model_type,
                "auc_mean_fold": float(np.mean([r[0] for r in runs])),
                "auc_sd_fold": float(np.mean([r[1] for r in runs])),
                "auc_pooled": float(np.mean(pooled)),
                "auc_pooled_sd_repeats": float(np.std(pooled)) if n_repeats > 1 else np.nan,
                "n_features": X_red.shape[1], "n_features_rad": n_rad, "n_features_gen": n_gen,
                "events_per_feature": n_min / max(X_red.shape[1], 1),
                "collapsed_model": collapsed,
            })

    spec_df = pd.DataFrame(rows).sort_values("auc_pooled").reset_index(drop=True)
    print(f"\n[specification_curve_v2] {len(spec_df)} combinazioni (specifica x modello) | "
          f"AUC pooled min={spec_df['auc_pooled'].min():.3f} max={spec_df['auc_pooled'].max():.3f}\n")
    return spec_df


def summarize_by_axis(spec_df: pd.DataFrame, axes: list) -> pd.DataFrame:
    """
    Riepilogo descrittivo: per ogni livello di ogni asse (e per modello) mediana,
    min e max di auc_pooled e numero medio di feature. Serve a leggere la curva
    ("quanto cambia l'AUC passando da legacy a medoid?"), non a scegliere una specifica.
    """
    parts = []
    for ax in axes:
        if ax == "model_type":
            continue
        g = spec_df.groupby([ax, "model_type"]).agg(
            n_specs=("auc_pooled", "size"),
            auc_pooled_median=("auc_pooled", "median"),
            auc_pooled_min=("auc_pooled", "min"),
            auc_pooled_max=("auc_pooled", "max"),
            n_features_mean=("n_features", "mean"),
        ).reset_index().rename(columns={ax: "level"})
        g.insert(0, "axis", ax)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
