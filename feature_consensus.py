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
        # con una sola feature verrebbe divisione per 0 -> si risolve dando direttamente assegnando
        # 0, non significa che non sia importante, ma che non è possibile creare una classifica tra 
        # features con un solo elemento
        return series * 0
    ranks = series.rank(method="average", ascending=True)
        # prende la series e assegna a ogni elemento della series una posizione nel ranking, in
        # ordine crescente (al più basso la prima posizione, al più alto l'ultima) e risolvendo 
        # i pareggi con la posizione media (entrambi secondi diventa entrambi nella posizione 2.5)
    return (ranks - 1) / (len(series) - 1) # normalizzazione tra 0 e 1, al primo verrà asseganto 0
        # mentre all'ultimo 1.


def build_feature_consensus(stable_features_path, shap_importance_path,
                             spec_votes_path, output_path=None):
    stable = pd.read_csv(stable_features_path, index_col=0).iloc[:, 0]
    shap_imp = pd.read_csv(shap_importance_path, index_col=0).iloc[:, 0]
    spec_votes = pd.read_csv(spec_votes_path, index_col=0).iloc[:, 0]
    # crea delle Series dai file csv creati in precedenza, indicizzate con i nomi delle varie
    # features

    all_features = sorted(set(stable.index) | set(shap_imp.index) | set(spec_votes.index))
    # sorted crea ordine alfabetico tra features

    df = pd.DataFrame(index=all_features)
    df["stability_selection_freq"] = stable.reindex(all_features).fillna(0)
        # reindex conforms DataFrame to new index with optional filling logic -> NaN diventano 0
    df["shap_mean_abs"] = shap_imp.reindex(all_features).fillna(0)
    df["spec_curve_votes"] = spec_votes.reindex(all_features).fillna(0)

    df["rank_stability"] = _normalize_rank(df["stability_selection_freq"])
        # normalizza il rank tra 0 e 1 per ogni series -> per ogni colonna del df 
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