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
    run_output_dir = config.OUTPUT_DIR / config.DATA_SOURCE
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Caricamento dati
    # ------------------------------------------------------------------
    X, y = data_utils.load_data(source=config.DATA_SOURCE)
    X.to_csv(run_output_dir / "X_features.csv")

    # ------------------------------------------------------------------
    # 2) Riduzione feature neutra (NON guarda la label)
    # ------------------------------------------------------------------
    X_reduced = data_utils.neutral_feature_reduction(X)
    X_reduced.to_csv(run_output_dir / "X_reduced_features.csv")

    # ------------------------------------------------------------------
    # 3) Nested CV su tutti i modelli
    # ------------------------------------------------------------------
    all_results = ml_pipeline.run_all_models(X_reduced, y)

    summary = summarize_results(all_results)
    summary.to_csv(run_output_dir / "model_comparison_summary.csv", index=False)
    pooled_summary = ml_pipeline.run_pooled_oof_analysis(all_results, X_reduced, y)

    print("\n" + "=" * 70)
    print("RIEPILOGO CONFRONTO MODELLI (nested CV)")
    print("=" * 70)
    print(summary.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("CONFRONTO FOLD-LEVEL vs POOLED OOF")
    print("=" * 70)
    print(pooled_summary.to_string(index=False))

    # salva i best_params per fold di ogni modello (utile per capire la stabilità del tuning)
    for model_name, res in all_results.items():
        params_df = pd.DataFrame(res["best_params"])
        params_df.to_csv(run_output_dir / f"{model_name}_best_params_per_fold.csv", index=False)

    # ------------------------------------------------------------------
    # 4) Stability selection (bootstrap) 
    # ------------------------------------------------------------------
    best_model_name = pooled_summary.iloc[0]["model"]
    best_params = ml_pipeline.majority_vote_params(all_results[best_model_name]["best_params"])

    stability_freq, stable_features = ml_pipeline.bootstrap_stability_selection(
        X_reduced, y, model_name=best_model_name, best_params=best_params
    )
    stability_freq.sort_values(ascending=False).to_csv(
        run_output_dir / f"feature_stability_frequencies_{best_model_name}.csv",
        header=["selection_frequency"]
    )
    stable_features.to_csv(
        run_output_dir / f"stable_features_final_{best_model_name}.csv",
        header=["selection_frequency"]
    )

    # ------------------------------------------------------------------
    # 5) SHAP sul modello con AUC migliore, calcolata out-of-fold e corredata di plot
    # ------------------------------------------------------------------
    if ml_pipeline.SHAP_AVAILABLE:
        best_model_name = pooled_summary.iloc[0]["model"]
        model_type = ml_pipeline.MODEL_TYPE_MAP.get(best_model_name)

        if model_type is None:
            print(f"\n[SHAP] modello migliore '{best_model_name}' non mappato in "
                  f"MODEL_TYPE_MAP: skip analisi SHAP")
        else:
            print(f"\n[SHAP] modello migliore secondo AUC (nested CV): "
                  f"{best_model_name} ({model_type})")

            best_results = all_results[best_model_name]
            shap_df, mean_abs_shap = ml_pipeline.out_of_fold_shap(
                best_results, X_reduced, model_type
            )

            # matrice completa pazienti x feature: riusabile anche come input
            # per lo studio di rete (es. correlazioni tra profili di importanza SHAP)
            shap_df.to_csv(run_output_dir / f"shap_values_{best_model_name}_out_of_fold.csv")
            mean_abs_shap.to_csv(
                run_output_dir / f"shap_feature_importance_{best_model_name}.csv",
                header=["mean_abs_shap"]
            )

            ml_pipeline.plot_shap_bar(
                mean_abs_shap, run_output_dir / f"shap_bar_{best_model_name}.png"
            )
            ml_pipeline.plot_shap_summary(
                shap_df, X_reduced, run_output_dir / f"shap_summary_{best_model_name}.png"
            )
            for feat in mean_abs_shap.head(3).index:
                safe_name = feat.replace("/", "_").replace(" ", "_")
                ml_pipeline.plot_shap_dependence(
                    shap_df, X_reduced, feat,
                    run_output_dir / f"shap_dependence_{safe_name}.png"
                )

            print("\n[SHAP] Top 15 feature per importanza media |SHAP| (out-of-fold):")
            print(mean_abs_shap.head(15))
            
            shap_sum = mean_abs_shap.sum()

            print(f"\n[SHAP] Somma totale valori shap medi = {shap_sum:.4f}")
            
            # Calcola la percentuale di contributo di ogni feature sul totale
            shap_percentage = (mean_abs_shap / mean_abs_shap.sum()) * 100

            print("\n[SHAP] Top 15 feature in percentuale di impatto:")
            print(shap_percentage.head(15).round(2).astype(str) + "%")

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
    with open(run_output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nTutti i risultati sono stati salvati in: {run_output_dir}")


if __name__ == "__main__":
    main()
