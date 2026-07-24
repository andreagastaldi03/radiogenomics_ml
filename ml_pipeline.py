"""
Pipeline di modellazione supervisionata con nested cross-validation.

Perché nested CV: con n=54 una singola divisione train/test dà stime di
performance molto instabili. La CV esterna stima la performance "onesta",
quella interna sceglie iperparametri/feature SENZA mai vedere il fold esterno
di test, evitando l'ottimismo da leakage.

Il modulo produce anche:
- un ranking di importanza delle feature aggregato su tutti i fold (stability)
- valori SHAP per l'interpretazione locale/globale del modello migliore
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)

import config

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] Libreria 'shap' non trovata: pip install shap per l'analisi di interpretabilità")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# DEFINIZIONE MODELLI E GRIGLIE DI IPERPARAMETRI
# ---------------------------------------------------------------------------
def get_model_grid():
    """
    Restituisce un dizionario {nome_modello: (pipeline, param_grid)}.

    Elastic Net è il modello principale consigliato: fa selezione automatica
    delle feature (L1) gestendo al contempo la collinearità (L2), ed è
    interpretabile tramite i coefficienti.

    Random Forest e SVM lineare sono inclusi come confronto / cross-check
    dell'importanza delle feature (via feature_importances_ o SHAP).
    """
    models = {}

    # --- Elastic Net (logistic regression con penalità elasticnet) ---
    en_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            random_state=config.RANDOM_STATE
        )),
    ])
    en_grid = {
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0],
        "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    models["elastic_net"] = (en_pipe, en_grid)

    # --- Random Forest ---
    rf_pipe = Pipeline([
        ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE)),
    ])
    rf_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    models["random_forest"] = (rf_pipe, rf_grid)

    # --- SVM lineare ---
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, random_state=config.RANDOM_STATE)),
    ])
    svm_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
    }
    models["svm_linear"] = (svm_pipe, svm_grid)

    # --- XGBoost (opzionale) ---
    if XGB_AVAILABLE:
        xgb_pipe = Pipeline([
            ("clf", XGBClassifier(
                eval_metric="logloss", random_state=config.RANDOM_STATE
            )),
        ])
        xgb_grid = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.01, 0.05, 0.1],
        }
        models["xgboost"] = (xgb_pipe, xgb_grid)

    return models


# ---------------------------------------------------------------------------
# NESTED CROSS VALIDATION
# ---------------------------------------------------------------------------
def nested_cv_evaluate(X: pd.DataFrame, y: pd.Series, model_name: str,
                        pipe, grid, n_outer=config.N_OUTER_FOLDS,
                        n_inner=config.N_INNER_FOLDS, random_state=config.RANDOM_STATE):
    """
    Esegue nested CV per un singolo modello e ritorna metriche per ciascun
    fold esterno + i migliori iperparametri scelti ad ogni fold.

    Ritorna un dict con liste di metriche (una entry per fold esterno) e la
    lista dei modelli fittati (utile per estrarre feature importance/SHAP).
    """
    y_bin = (y == config.POSITIVE_CLASS).astype(int)

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    results = {
        "auc": [], "balanced_accuracy": [], "f1": [],
        "best_params": [], "confusion_matrices": [],
        "fitted_models": [], "test_indices": []
    }

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_bin)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        results["auc"].append(roc_auc_score(y_test, y_proba))
        results["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred))
        results["f1"].append(f1_score(y_test, y_pred))
        results["best_params"].append(search.best_params_)
        results["confusion_matrices"].append(confusion_matrix(y_test, y_pred))
        results["fitted_models"].append(best_model)
        results["test_indices"].append(test_idx)

        print(f"[{model_name}] fold {fold_i+1}/{n_outer} | "
              f"AUC={results['auc'][-1]:.3f} | "
              f"BalAcc={results['balanced_accuracy'][-1]:.3f} | "
              f"best_params={search.best_params_}")

    print(f"\n[{model_name}] RISULTATI NESTED CV (media ± sd su {n_outer} fold esterni)")
    for metric in ["auc", "balanced_accuracy", "f1"]:
        vals = results[metric]
        print(f"  {metric}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    return results


def run_all_models(X: pd.DataFrame, y: pd.Series):
    """
    Esegue nested CV per tutti i modelli definiti in get_model_grid()
    e ritorna un dizionario {model_name: results}.
    """
    models = get_model_grid()
    all_results = {}
    for name, (pipe, grid) in models.items():
        print(f"\n{'='*70}\nModello: {name}\n{'='*70}")
        all_results[name] = nested_cv_evaluate(X, y, name, pipe, grid)
    return all_results


# ---------------------------------------------------------------------------
# STABILITY SELECTION (bootstrap) — quali feature emergono ripetutamente
# ---------------------------------------------------------------------------
def bootstrap_stability_selection(X: pd.DataFrame, y: pd.Series,
                                   n_bootstrap=config.N_BOOTSTRAP,
                                   threshold=config.STABILITY_SELECTION_THRESHOLD,
                                   random_state=config.RANDOM_STATE):
    """
    Rifitta un Elastic Net su n_bootstrap campioni bootstrap del dataset e
    conta quante volte ciascuna feature riceve un coefficiente non nullo.

    Con n=54 questo è più informativo di un singolo fit: ti dice quali
    feature sono "affidabilmente" rilevanti e non un artefatto del
    particolare campione osservato. Queste sono le feature da riportare
    come risultato principale, non i coefficienti di un singolo fit.
    """
    y_bin = (y == config.POSITIVE_CLASS).astype(int)
    n_samples = X.shape[0]
    selection_counts = pd.Series(0, index=X.columns)

    rng = np.random.RandomState(random_state)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            C=0.1, l1_ratio=0.5, random_state=random_state
        )),
    ])

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X.iloc[idx], y_bin.iloc[idx]

        if y_boot.nunique() < 2:
            continue  # bootstrap degenere, salta

        pipe.fit(X_boot, y_boot)
        coefs = pipe.named_steps["clf"].coef_.ravel()
        selected = X.columns[np.abs(coefs) > 1e-8]
        selection_counts[selected] += 1

    stability_freq = selection_counts / n_bootstrap
    stable_features = stability_freq[stability_freq >= threshold].sort_values(ascending=False)

    print(f"\n[stability_selection] {len(stable_features)} feature stabili "
          f"(selezionate in >={threshold*100:.0f}% dei {n_bootstrap} bootstrap)")
    print(stable_features)

    return stability_freq, stable_features


# ---------------------------------------------------------------------------
# SHAP — interpretabilità del modello (usare sul modello finale, fit su tutti i dati
# o su un modello rappresentativo tra i fold)
# ---------------------------------------------------------------------------
def shap_analysis(fitted_pipeline, X: pd.DataFrame, model_type="tree"):
    """
    Calcola valori SHAP per il modello fittato.

    model_type: "tree" per RandomForest/XGBoost, "linear" per Elastic Net/SVM lineare.
    Ritorna l'oggetto shap_values e l'explainer, utili per i plot successivi
    (shap.summary_plot, shap.dependence_plot, ecc. — da eseguire in notebook/script separato
    per la visualizzazione).
    """
    if not SHAP_AVAILABLE:
        raise ImportError("Installa 'shap' con: pip install shap")

    clf = fitted_pipeline.named_steps["clf"]

    # applica le trasformazioni precedenti (es. scaler) prima di passare a SHAP
    X_transformed = X.copy()
    for step_name, step in fitted_pipeline.steps[:-1]:
        X_transformed = pd.DataFrame(
            step.transform(X_transformed), columns=X.columns, index=X.index
        )

    if model_type == "tree":
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(clf, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        raise ValueError("model_type deve essere 'tree' o 'linear'")

    return explainer, shap_values, X_transformed
