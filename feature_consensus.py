"""
Consenso tra criteri di importanza indipendenti: stability selection
bootstrap (EN), SHAP out-of-fold (miglior modello nested CV), voti della
specification curve. Ogni criterio da solo può essere fragile (un solo
modello, un solo modo di ridurre le feature); le feature che emergono da
tutti e tre insieme sono quelle su cui vale la pena costruire
un'interpretazione biologica.
"""

import pandas as pd
import config


def _normalize_rank(series: pd.Series) -> pd.Series:
    """Rank normalizzato 0-1 (1 = più importante). Gestisce serie di 1 elemento."""
    if len(series) <= 1:
        return series * 0
    ranks = series.rank(method="average", ascending=True)
    return (ranks - 1) / (len(series) - 1)


def build_feature_consensus(stable_features_path, shap_importance_path,
                             spec_votes_path, output_path=None):
    stable = pd.read_csv(stable_features_path, index_col=0).iloc[:, 0]
    shap_imp = pd.read_csv(shap_importance_path, index_col=0).iloc[:, 0]
    spec_votes = pd.read_csv(spec_votes_path, index_col=0).iloc[:, 0]

    all_features = sorted(set(stable.index) | set(shap_imp.index) | set(spec_votes.index))

    df = pd.DataFrame(index=all_features)
    df["stability_selection_freq"] = stable.reindex(all_features).fillna(0)
    df["shap_mean_abs"] = shap_imp.reindex(all_features).fillna(0)
    df["spec_curve_votes"] = spec_votes.reindex(all_features).fillna(0)

    df["rank_stability"] = _normalize_rank(df["stability_selection_freq"])
    df["rank_shap"] = _normalize_rank(df["shap_mean_abs"])
    df["rank_spec_votes"] = _normalize_rank(df["spec_curve_votes"])

    df["consensus_score"] = df[["rank_stability", "rank_shap", "rank_spec_votes"]].mean(axis=1)
    df["n_criteria_present"] = (
        (df["stability_selection_freq"] > 0).astype(int)
        + (df["shap_mean_abs"] > 0).astype(int)
        + (df["spec_curve_votes"] > 0).astype(int)
    )

    df = df.sort_values("consensus_score", ascending=False)

    if output_path:
        df.to_csv(output_path)
        print(f"[feature_consensus] salvato in {output_path}")

    print("\n[feature_consensus] Top 15 feature per consenso:")
    print(df[["consensus_score", "n_criteria_present"]].head(15))
    return df


if __name__ == "__main__":
    # shap_feature_importance_*.csv ha nel nome il modello migliore trovato
    # da run_analysis.py, quindi lo cerchiamo dinamicamente invece di fissarlo
    shap_files = list(config.OUTPUT_DIR.glob("shap_feature_importance_*.csv"))
    if not shap_files:
        raise FileNotFoundError(
            "Nessun shap_feature_importance_*.csv trovato: esegui prima run_analysis.py"
        )

    build_feature_consensus(
        stable_features_path=config.OUTPUT_DIR / "stable_features_final.csv",
        shap_importance_path=shap_files[0],
        spec_votes_path=config.OUTPUT_DIR / "feature_votes_across_specs_linear.csv",  
        # o la versione _linear/_tree
        output_path=config.OUTPUT_DIR / "feature_consensus.csv",
    )