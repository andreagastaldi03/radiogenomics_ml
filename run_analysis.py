"""
Entry point della pipeline ML.

Esegue in sequenza:
1. Caricamento dati (radiomica / genomica / entrambe, vedi config.DATA_SOURCE)
2. Riduzione feature "neutra" (varianza + ridondanza), indipendente dalla label
3. Nested CV per Elastic Net, Random Forest, SVM lineare, (XGBoost se disponibile)
4. Stability selection via bootstrap sull'Elastic Net
5. SHAP sul modello ad albero (se disponibile) per confronto dell'importanza
6. Salvataggio di tutte le tabelle di risultati in config.OUTPUT_DIR

Uso:
    python run_analysis.py
Modifica config.DATA_SOURCE per lanciare la variante radiomics / genomics / both.
"""

import json
import numpy as np
import pandas as pd

import config
import data_utils
import ml_pipeline


def summarize_results(all_results: dict) -> pd.DataFrame:
    """Tabella riassuntiva media±sd per ciascun modello, pronta da riportare in un report."""
    rows = []
    for model_name, res in all_results.items():
        rows.append({
            "model": model_name,
            "auc_mean": np.mean(res["auc"]),
            "auc_sd": np.std(res["auc"]),
            "balanced_accuracy_mean": np.mean(res["balanced_accuracy"]),
            "balanced_accuracy_sd": np.std(res["balanced_accuracy"]),
            "f1_mean": np.mean(res["f1"]),
            "f1_sd": np.std(res["f1"]),
        })
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False)


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Caricamento dati
    # ------------------------------------------------------------------
    X, y = data_utils.load_data(source=config.DATA_SOURCE)

    # ------------------------------------------------------------------
    # 2) Riduzione feature neutra (NON guarda la label)
    # ------------------------------------------------------------------
    X_reduced = data_utils.neutral_feature_reduction(X)
    X_reduced.to_csv(config.OUTPUT_DIR / "X_reduced_features.csv")

    # ------------------------------------------------------------------
    # 3) Nested CV su tutti i modelli
    # ------------------------------------------------------------------
    all_results = ml_pipeline.run_all_models(X_reduced, y)

    summary = summarize_results(all_results)
    summary.to_csv(config.OUTPUT_DIR / "model_comparison_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("RIEPILOGO CONFRONTO MODELLI (nested CV)")
    print("=" * 70)
    print(summary.to_string(index=False))

    # salva i best_params per fold di ogni modello (utile per capire la stabilità del tuning)
    for model_name, res in all_results.items():
        params_df = pd.DataFrame(res["best_params"])
        params_df.to_csv(config.OUTPUT_DIR / f"{model_name}_best_params_per_fold.csv", index=False)

    # ------------------------------------------------------------------
    # 4) Stability selection (bootstrap) su Elastic Net
    # ------------------------------------------------------------------
    stability_freq, stable_features = ml_pipeline.bootstrap_stability_selection(X_reduced, y)
    stability_freq.sort_values(ascending=False).to_csv(
        config.OUTPUT_DIR / "feature_stability_frequencies.csv",
        header=["selection_frequency"]
    )
    stable_features.to_csv(
        config.OUTPUT_DIR / "stable_features_final.csv",
        header=["selection_frequency"]
    )

    # ------------------------------------------------------------------
    # 5) SHAP sul miglior modello ad albero (se presente e se shap è installato)
    # ------------------------------------------------------------------
    if ml_pipeline.SHAP_AVAILABLE and "random_forest" in all_results:
        # usa il modello del fold con AUC più alta come rappresentativo
        rf_res = all_results["random_forest"]
        best_fold = int(np.argmax(rf_res["auc"]))
        best_rf_model = rf_res["fitted_models"][best_fold]

        explainer, shap_values, X_transformed = ml_pipeline.shap_analysis(
            best_rf_model, X_reduced, model_type="tree"
        )
        # per classificazione binaria TreeExplainer ritorna una lista [class0, class1]
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs_shap = pd.Series(
            np.abs(sv).mean(axis=0), index=X_reduced.columns
        ).sort_values(ascending=False)
        mean_abs_shap.to_csv(
            config.OUTPUT_DIR / "shap_feature_importance_random_forest.csv",
            header=["mean_abs_shap"]
        )
        print("\n[SHAP] Top 15 feature per importanza media |SHAP| (Random Forest):")
        print(mean_abs_shap.head(15))

    # ------------------------------------------------------------------
    # 6) Salva anche un JSON compatto con la configurazione usata (riproducibilità)
    # ------------------------------------------------------------------
    run_config = {
        "data_source": config.DATA_SOURCE,
        "n_patients": int(X.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_after_reduction": int(X_reduced.shape[1]),
        "variance_threshold": config.VARIANCE_THRESHOLD,
        "redundancy_corr_threshold": config.REDUNDANCY_CORR_THRESHOLD,
        "n_outer_folds": config.N_OUTER_FOLDS,
        "n_inner_folds": config.N_INNER_FOLDS,
        "n_bootstrap_stability": config.N_BOOTSTRAP,
    }
    with open(config.OUTPUT_DIR / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nTutti i risultati sono stati salvati in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
