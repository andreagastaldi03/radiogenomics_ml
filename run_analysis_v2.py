"""
Entry point della pipeline con riduzione feature v2.

Per ogni specifica in config_reduction.SPECS:
  1. riduzione neutra v2 (reduction_v2.BlockReducer) sull'intero dataset
     (non usa la label);
  2. nested CV su tutti i modelli di ml_pipeline + metriche pooled OOF;
  3. solo per config_reduction.SPECS_FULL_ANALYSIS: stability selection e SHAP.
Alla fine salva la tabella con TUTTE le specifiche affiancate.

Uso:
    python run_analysis_v2.py
Il backend CPU/GPU dipende dalla versione di ml_pipeline.py presente (se usi
quella con gpu_utils.py si applica config.USE_GPU).
"""

import inspect
import json

import numpy as np
import pandas as pd

import config
import config_reduction as cr
import data_utils
import ml_pipeline
import reduction_v2

_SHAP_ACCEPTS_Y = "y" in inspect.signature(ml_pipeline.out_of_fold_shap).parameters


def summarize_results(all_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, res in all_results.items():
        rows.append({
            "model": model_name,
            "backend": res.get("backend", "cpu"),
            "auc_mean": np.mean(res["auc"]), "auc_sd": np.std(res["auc"]),
            "balanced_accuracy_mean": np.mean(res["balanced_accuracy"]),
            "balanced_accuracy_sd": np.std(res["balanced_accuracy"]),
            "f1_mean": np.mean(res["f1"]), "f1_sd": np.std(res["f1"]),
        })
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False)


def print_feature_budget(y: pd.Series, n_features: int):
    """Confronto tra feature disponibili e budget ~ classe minoritaria / 10."""
    n_pos = int((y == config.POSITIVE_CLASS).sum())
    n_min = min(n_pos, len(y) - n_pos)
    print(f"[budget] pazienti={len(y)} | classe minoritaria={n_min} | "
          f"budget ~{n_min / 10:.1f} feature | feature nel modello={n_features} "
          f"(~{n_min / max(n_features, 1):.1f} eventi per feature)")


def stability_and_shap(all_results, pooled_summary, X_red, y, out_dir):
    """Passi 4-5 di run_analysis.py, invariati nella logica."""
    best_model_name = pooled_summary.iloc[0]["model"]
    best_params = ml_pipeline.majority_vote_params(all_results[best_model_name]["best_params"])

    stability_freq, stable_features = ml_pipeline.bootstrap_stability_selection(
        X_red, y, model_name=best_model_name, best_params=best_params
    )
    stability_freq.sort_values(ascending=False).to_csv(
        out_dir / f"feature_stability_frequencies_{best_model_name}.csv",
        header=["selection_frequency"])
    stable_features.to_csv(
        out_dir / f"stable_features_final_{best_model_name}.csv",
        header=["selection_frequency"])

    if not ml_pipeline.SHAP_AVAILABLE:
        return
    model_type = ml_pipeline.MODEL_TYPE_MAP.get(best_model_name)
    if model_type is None:
        print(f"[SHAP] '{best_model_name}' non mappato in MODEL_TYPE_MAP: skip")
        return

    best_results = all_results[best_model_name]
    if _SHAP_ACCEPTS_Y:
        shap_df, mean_abs_shap = ml_pipeline.out_of_fold_shap(best_results, X_red, model_type, y=y)
    else:
        shap_df, mean_abs_shap = ml_pipeline.out_of_fold_shap(best_results, X_red, model_type)

    shap_df.to_csv(out_dir / f"shap_values_{best_model_name}_out_of_fold.csv")
    mean_abs_shap.to_csv(out_dir / f"shap_feature_importance_{best_model_name}.csv",
                         header=["mean_abs_shap"])
    ml_pipeline.plot_shap_bar(mean_abs_shap, out_dir / f"shap_bar_{best_model_name}.png")
    ml_pipeline.plot_shap_summary(shap_df, X_red, out_dir / f"shap_summary_{best_model_name}.png")
    for feat in mean_abs_shap.head(3).index:
        safe = feat.replace("/", "_").replace(" ", "_")
        ml_pipeline.plot_shap_dependence(shap_df, X_red, feat, out_dir / f"shap_dependence_{safe}.png")
    print("\n[SHAP] Top 15 feature per importanza media |SHAP| (out-of-fold):")
    print(mean_abs_shap.head(15))


def run_one_spec(spec_name: str, spec: dict, X: pd.DataFrame, y: pd.Series, base_dir):
    out_dir = base_dir / spec_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 70 + f"\nSPECIFICA: {spec_name}  {spec}\n" + "#" * 70)

    X_red, reducer = reduction_v2.reduce_features_v2(X, print_info=True, **spec)
    X_red.to_csv(out_dir / "X_reduced_features.csv")
    reducer.report_.to_csv(out_dir / "reduction_report.csv", index=False)
    reducer.feature_map_.to_csv(out_dir / "reduction_feature_map.csv", index=False)
    print_feature_budget(y, X_red.shape[1])

    all_results = ml_pipeline.run_all_models(X_red, y)
    summary = summarize_results(all_results)
    summary.to_csv(out_dir / "model_comparison_summary.csv", index=False)
    pooled = ml_pipeline.run_pooled_oof_analysis(all_results, X_red, y, output_dir=out_dir)

    for model_name, res in all_results.items():
        pd.DataFrame(res["best_params"]).to_csv(
            out_dir / f"{model_name}_best_params_per_fold.csv", index=False)

    print("\nRIEPILOGO (nested CV)\n" + summary.to_string(index=False))
    print("\nFOLD-LEVEL vs POOLED OOF\n" + pooled.to_string(index=False))

    if spec_name in cr.SPECS_FULL_ANALYSIS:
        stability_and_shap(all_results, pooled, X_red, y, out_dir)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "spec_name": spec_name, "spec": spec, "data_source": config.DATA_SOURCE,
            "n_patients": int(X.shape[0]), "n_features_raw": int(X.shape[1]),
            "n_features_after_reduction": int(X_red.shape[1]),
            "n_outer_folds": config.N_OUTER_FOLDS, "n_inner_folds": config.N_INNER_FOLDS,
            "backend_per_model": {n: r.get("backend", "cpu") for n, r in all_results.items()},
        }, f, indent=2, default=str)

    pooled = pooled.copy()
    pooled.insert(0, "spec", spec_name)
    pooled.insert(1, "n_features", X_red.shape[1])
    return pooled


def main():
    base_dir = cr.OUTPUT_ROOT / config.DATA_SOURCE
    base_dir.mkdir(parents=True, exist_ok=True)

    X, y = data_utils.load_data(source=config.DATA_SOURCE, print_info=True)

    tables = [run_one_spec(name, spec, X, y, base_dir) for name, spec in cr.SPECS.items()]
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(base_dir / "all_specs_comparison.csv", index=False)

    print("\n" + "=" * 70 + "\nCONFRONTO TRA TUTTE LE SPECIFICHE (pooled OOF)\n" + "=" * 70)
    print(combined[["spec", "n_features", "model", "pooled_oof_auc",
                    "pooled_oof_balanced_accuracy"]].to_string(index=False))
    print("\nNB: scegliere la specifica col miglior AUC e riportare solo quella è tuning "
          "sui dati di valutazione: riporta l'intera tabella.")
    print(f"\nRisultati salvati in: {base_dir}")


if __name__ == "__main__":
    main()