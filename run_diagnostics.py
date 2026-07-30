"""
Script di diagnostica —  verifica che i risultati del modello (in run_analysis.py)
non siano "falsi positivi" dovuti al caso o a un artefatto tecnico.

Vanno lanciati DOPO run_analysis.py, perché servono gli iperparametri del
modello migliore già trovati (guarda model_comparison_summary.csv e
elastic_net_best_params_per_fold.csv nella cartella outputs).

Uso:
    python run_diagnostics.py
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import config
import data_utils
import diagnostics


# ---------------------------------------------------------------------------
# CONFIGURA IL MODELLO "MIGLIORE" DA TESTARE
# ---------------------------------------------------------------------------
# Copia gli iperparametri del modello con AUC più alta trovato da
# run_analysis.py (li trovi in model_comparison_summary.csv e nel file
# <nome_modello>_best_params_per_fold.csv). Questo esempio usa elastic_net;
# se il tuo modello migliore è random_forest o svm_linear, sostituisci la
# pipeline sotto con quella corrispondente.
BEST_MODEL_PARAMS = {"C": 0.01, "l1_ratio": 0.1}  # <-- sostituisci con i tuoi valori reali


def build_best_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            C=BEST_MODEL_PARAMS["C"], l1_ratio=BEST_MODEL_PARAMS["l1_ratio"],
            random_state=config.RANDOM_STATE
        )),
    ])


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = data_utils.load_data(source=config.DATA_SOURCE)
    X_reduced = data_utils.neutral_feature_reduction(X)

    # ------------------------------------------------------------------
    # 1) TEST DI PERMUTAZIONE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST DI PERMUTAZIONE")
    print("=" * 70)
    pipe = build_best_pipeline()
    real_auc, permuted_aucs, p_value = diagnostics.permutation_test(X_reduced, y, pipe)

    diagnostics.plot_permutation_test(
        real_auc, permuted_aucs, config.OUTPUT_DIR / "permutation_test.png"
    )
    pd.Series(permuted_aucs, name="auc_permutato").to_csv(
        config.OUTPUT_DIR / "permutation_test_aucs.csv", index=False
    )
    with open(config.OUTPUT_DIR / "permutation_test_summary.txt", "w") as f:
        f.write(f"AUC dati veri: {real_auc:.4f}\n")
        f.write(f"AUC media permutata: {permuted_aucs.mean():.4f} ± {permuted_aucs.std():.4f}\n")
        f.write(f"p-value empirico: {p_value:.4f}\n")

    # ------------------------------------------------------------------
    # 2) CONTROLLO BATCH EFFECT
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONTROLLO BATCH EFFECT")
    print("=" * 70)
    batch = data_utils.load_batch_column(X_reduced.index)  # None se config.BATCH_COL non è impostato

    diagnostics.batch_effect_diagnostic(
        X_reduced, y, batch=batch,
        output_path=config.OUTPUT_DIR / "batch_effect_pca.png"
    )

    print(f"\nDiagnostica completata. File salvati in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
