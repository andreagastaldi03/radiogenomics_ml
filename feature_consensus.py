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
import ml_pipeline


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
    run_output_dir = config.OUTPUT_DIR / config.DATA_SOURCE
    
    # shap_feature_importance_*.csv ha nel nome il modello migliore trovato
    # da run_analysis.py, quindi lo cerchiamo dinamicamente invece di fissarlo
    shap_files = list(run_output_dir.glob("shap_feature_importance_*.csv"))
    if not shap_files:
        raise FileNotFoundError(
            f"Nessun shap_feature_importance_*.csv trovato in {run_output_dir}: "
            f"esegui prima run_analysis.py con DATA_SOURCE='{config.DATA_SOURCE}'"
        )
    shap_path = shap_files[0]
    # il nome del modello migliore è incastonato nel filename, es.
    # "shap_feature_importance_random_forest.csv" -> "random_forest"
    best_model_name = shap_path.stem.replace("shap_feature_importance_", "")
    
    stable_files = list(run_output_dir.glob(f"stable_features_final_{best_model_name}.csv"))
    if not stable_files:
        raise FileNotFoundError(
            f"Nessun stable_features_final_{best_model_name}.csv trovato in {run_output_dir}: "
            f"controlla che la stability selection sia stata rifatta col modello migliore aggiornato."
        )
    stable_path = stable_files[0]
    
    # il file di voti della specification curve dipende dal TIPO di modello
    # (linear/tree), non dal nome specifico, e la spec curve è globale
    # (non nella sottocartella del data_source)
    model_type = ml_pipeline.MODEL_TYPE_MAP.get(best_model_name)
    if model_type is None:
        raise ValueError(
            f"'{best_model_name}' non è mappato in MODEL_TYPE_MAP: aggiungilo per poter "
            f"scegliere automaticamente il file di voti della specification curve."
        )
    spec_votes_path = config.OUTPUT_DIR / f"feature_votes_across_specs_{model_type}.csv"
    if not spec_votes_path.exists():
        raise FileNotFoundError(
            f"{spec_votes_path} non trovato: esegui prima run_specification_curve.py"
        )

    print(f"[feature_consensus] modello migliore: {best_model_name} ({model_type})")
    print(f"[feature_consensus] stability: {stable_path}")
    print(f"[feature_consensus] shap: {shap_path}")
    print(f"[feature_consensus] spec curve votes: {spec_votes_path}")

    build_feature_consensus(
        stable_features_path=stable_path,
        shap_importance_path=shap_path,
        spec_votes_path=spec_votes_path,
        output_path=run_output_dir / "feature_consensus.csv",
    )