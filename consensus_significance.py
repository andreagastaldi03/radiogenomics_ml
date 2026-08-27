"""
Correzione per test multipli sul consensus score (feature_consensus.py).

Il consensus score combina tre criteri indipendenti — stability selection
bootstrap, SHAP out-of-fold, voti della specification curve — proprio per
essere più difendibile di uno solo. Ma con centinaia di feature scremate da
tre criteri, guardare solo il ranking assoluto resta rischioso: anche se
nessuna feature avesse un legame reale col fenotipo, ci si aspetta comunque
che qualcuna ottenga un consenso alto tra i tre criteri semplicemente per
caso (tre criteri rumorosi che "concordano per sbaglio" su una feature
diversa ogni volta).

Questo modulo rifà l'intera pipeline di consenso (bootstrap stability
selection + SHAP out-of-fold + intera specification curve) su etichette
permutate, un piccolo numero di volte, e guarda il consensus_score più alto
ottenuto per caso ad ogni permutazione. Confrontare ogni feature reale con
questa distribuzione (del massimo, non della media) è l'approccio
max-statistic / Westfall-Young: controlla il tasso di falsi positivi
sull'intero esperimento (una feature qualunque emerge per caso), non solo
sulla singola feature.

Costoso di proposito tenuto economico: gli iperparametri sono sempre quelli
già scelti sui dati reali (nessun re-tuning dentro le permutazioni, stessa
logica già usata in diagnostics.py e specification_curve.py) e il numero di
permutazioni è basso — un controllo di plausibilità, non un test ad alta
risoluzione. Con m permutazioni il p-value minimo ottenibile è 1/(m+1).
"""

import ast
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

import config
import data_utils
import ml_pipeline
import specification_curve as sc
import feature_consensus


# ---------------------------------------------------------------------------
# CONTESTO "CONGELATO": tutto ciò che serve dai risultati già salvati da
# run_analysis.py / run_specification_curve.py, senza rifare nessun tuning.
# ---------------------------------------------------------------------------
def _parse_best_params(val):
    """
    specification_curve_results.csv salva la colonna best_params come
    stringa (repr di un dict) per le righe 'linear', NaN per le righe
    'tree' (RandomForest non viene tunato per-specifica, vedi
    specification_curve._cv_eval). Qui si fa il percorso inverso.
    """
    if pd.isna(val): # Detect missing values for an array-like object.
        return None
    return ast.literal_eval(val) # prende una stringa, converte in modo sicuro 
        # una stringa che rappresenta una struttura dati Python (come liste, 
        # dizionari o tuple) nel suo rispettivo oggetto Python. è sicura al 100% 
        # perché accetta solo stringhe contenenti stringhe, numeri, tuple, liste, 
        # dizionari, booleani e None. Se la stringa contiene comandi dannosi, 
        # si blocca ed estrae un errore.


def load_frozen_context(data_source: str = "both"):
    """
    Ricostruisce, dai file già salvati, tutto ciò che serve per rifare
    stability selection + SHAP + specification curve con iperparametri
    fissi (quelli scelti una volta sola sui dati reali):

    - best_model_name, best_params, model_type : il modello "migliore"
      secondo l'AUC pooled OOF
    - X_reduced, y : dati ridotti (neutri, indipendenti dalla label) e
      label originali, per la sorgente 'both' (serve avere sia le feature
      radiomiche sia quelle genomiche, essendo il consensus calcolato su
      un modello allenato su entrambe)
    - spec_fixed_params_dict : iperparametri congelati per ogni riga della
      specification curve reale (chiave = combo di preprocessing + modello),
      da riusare identici in tutte le permutazioni della spec curve
    """
    run_output_dir = config.OUTPUT_DIR / data_source

    pooled_summary = pd.read_csv(run_output_dir / "pooled_oof_model_comparison.csv")
    best_model_name = pooled_summary.sort_values(
        "pooled_oof_auc", ascending=False
    ).iloc[0]["model"]

    best_params_df = pd.read_csv(run_output_dir / f"{best_model_name}_best_params_per_fold.csv")
    best_params = ml_pipeline.majority_vote_params(best_params_df.to_dict(orient="records"))
        # converte un DataFrame di Pandas in una lista di dizionari, dove ogni dizionario 
        # corrisponde a una riga della tabella.

    model_type = ml_pipeline.MODEL_TYPE_MAP.get(best_model_name)
    if model_type is None:
        raise ValueError(f"'{best_model_name}' non è mappato in MODEL_TYPE_MAP.")

    X, y = data_utils.load_data(source=data_source, print_info=False)
    X_reduced = data_utils.neutral_feature_reduction(X, print_info=False)

    spec_results_path = config.OUTPUT_DIR / "specification_curve_results.csv"
    if not spec_results_path.exists():
        raise FileNotFoundError(
            f"{spec_results_path} non trovato: esegui prima run_specification_curve.py "
            f"(serve per congelare gli iperparametri della spec curve nelle permutazioni)."
        )
    spec_df = pd.read_csv(spec_results_path)
    spec_keys = list(sc.SPEC_GRID.keys())
    spec_fixed_params_dict = {}
    for _, row in spec_df.iterrows():
        key = tuple(row[k] for k in spec_keys) + (row["model_type"],)
        spec_fixed_params_dict[key] = _parse_best_params(row["best_params"])

    print(f"[load_frozen_context] modello migliore: {best_model_name} ({model_type}) | "
          f"iperparametri congelati: {best_params}")
    print(f"[load_frozen_context] {len(spec_fixed_params_dict)} combinazioni di specification "
          f"curve congelate da {spec_results_path.name}")

    return {
        "best_model_name": best_model_name, "best_params": best_params,
        "model_type": model_type, "X_reduced": X_reduced, "y": y,
        "spec_fixed_params_dict": spec_fixed_params_dict,
    }


# ---------------------------------------------------------------------------
# SHAP out-of-fold con iperparametri fissi: k-fold semplice (non nested,
# niente grid search), stessa filosofia di diagnostics._cv_auc — qui vogliamo
# solo "con la ricetta già scelta, quali feature emergono?", non ritarare i
# parametri sotto etichette permutate.
# ---------------------------------------------------------------------------
def _simple_outer_kfold_fit(X: pd.DataFrame, y_bin: pd.Series, model_name: str,
                            best_params: dict, n_folds: int, random_state: int):
    """
    Produce un dict {'fitted_models', 'test_indices'} compatibile con
    ml_pipeline.out_of_fold_shap, senza nested CV/tuning.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {"fitted_models": [], "test_indices": []}
    for train_idx, test_idx in cv.split(X, y_bin):
        pipe = ml_pipeline.build_pipeline_from_best_params(model_name, best_params)
        pipe.fit(X.iloc[train_idx], y_bin.iloc[train_idx])
        results["fitted_models"].append(pipe)
        results["test_indices"].append(test_idx)
    return results


# ---------------------------------------------------------------------------
# UNA SINGOLA (RI)COSTRUZIONE DEL CONSENSUS, con etichette eventualmente permutate
# ---------------------------------------------------------------------------
def _one_consensus_run(y_labels: pd.Series, ctx: dict, random_state: int,
                       n_bootstrap_stability: int, n_folds_shap: int,
                       n_jobs_model: int = -1, verbose: bool = False) -> pd.DataFrame:
    """
    Rifà i punti 1-3 di feature_consensus.py (stability selection, SHAP
    out-of-fold, voti spec curve) con le y fornite (reali o permutate) e
    iperparametri congelati da ctx, poi combina i tre criteri.

    Nota sulla permutazione: stability selection e SHAP condividono la
    stessa permutazione (la stessa y_labels passata a entrambe). I voti
    della specification curve, invece, permutano internamente ed
    indipendentemente le label per ciascuna data_source della griglia
    (radiomics/genomics/both hanno pazienti diversi, quindi non esiste
    un'unica permutazione condivisibile tra tutte e tre) — una
    semplificazione deliberata per riusare la funzione già esistente
    invece di duplicarne la logica.
    """
    X_reduced, best_model_name, best_params, model_type = (
        ctx["X_reduced"], ctx["best_model_name"], ctx["best_params"], ctx["model_type"]
    )

    stability_freq, _ = ml_pipeline.bootstrap_stability_selection(
        X_reduced, y_labels, model_name=best_model_name, best_params=best_params,
        n_bootstrap=n_bootstrap_stability, random_state=random_state
    )

    y_bin = (y_labels == config.POSITIVE_CLASS).astype(int)
    shap_results = _simple_outer_kfold_fit(
        X_reduced, y_bin, best_model_name, best_params,
        n_folds=n_folds_shap, random_state=random_state
    )
    _, mean_abs_shap = ml_pipeline.out_of_fold_shap(shap_results, X_reduced, model_type)

    permute_spec = not y_labels.equals(ctx["y"])
        # la variabile permute_spec assume il valore True se le etichette y_labels sono 
        # diverse da ctx["y"], altrimenti assume il valore False. Se y_lab è quella reale,
        # non c'è permutazione e viene passato false, quindi run_spec_curve funge normalmente,
        # altrimenti si applica la variante per etichette permutate (permutazione interna label)
    _, _, feature_votes_by_model, _ = sc.run_specification_curve(
        permute_labels=permute_spec, random_state=random_state,
        fixed_params_dict=ctx["spec_fixed_params_dict"],
        verbose=verbose, n_jobs_model=n_jobs_model
    )
    spec_votes = feature_votes_by_model[model_type]

    return feature_consensus._consensus_from_series(stability_freq, mean_abs_shap, spec_votes)


# ---------------------------------------------------------------------------
# TEST DI PERMUTAZIONE SUL CONSENSUS SCORE
# ---------------------------------------------------------------------------
def permutation_null_consensus(real_consensus_df: pd.DataFrame = None,
                               data_source: str = "both",
                               n_permutations: int = None,
                               n_bootstrap_stability: int = None,
                               n_folds_shap: int = None,
                               n_jobs: int = None,
                               random_state: int = config.RANDOM_STATE):
    """
    Parametri
    ---------
    real_consensus_df : il consensus reale già calcolato (feature_consensus.csv).
        Se None, viene ricalcolato da zero sui dati reali con la stessa
        funzione _one_consensus_run usata per le permutazioni (più lento ma
        garantisce la stessa identica pipeline per il confronto).
    n_permutations : default config.CONSENSUS_N_PERMUTATIONS (basso di
        proposito: ogni permutazione rifà bootstrap stability + SHAP +
        l'intera specification curve).
    n_bootstrap_stability : default config.N_BOOTSTRAP (stesso numero usato
        nella pipeline reale, per confrontabilità).
    n_folds_shap : default config.PERMUTATION_N_FOLDS.

    Ritorna
    -------
    result_df : il consensus reale con 2 colonne aggiunte,
        p_value_fwer (Westfall-Young: quante permutazioni hanno prodotto un
        consensus_score massimo >= a quello di QUESTA feature, sull'intero
        dataset) e significant_fwer (p_value_fwer < 0.05).
    null_max_scores : array (n_permutations,), il consensus_score più alto
        ottenuto per caso ad ogni permutazione — usa questa distribuzione
        come soglia, non il ranking assoluto.
    """
    n_permutations = n_permutations or config.CONSENSUS_N_PERMUTATIONS
    n_bootstrap_stability = n_bootstrap_stability or config.N_BOOTSTRAP
    n_folds_shap = n_folds_shap or config.PERMUTATION_N_FOLDS
    n_jobs = config.N_JOBS_CONSENSUS_PERMUTATIONS if n_jobs is None else n_jobs
    n_jobs_model = 1 if n_jobs != 1 else -1

    ctx = load_frozen_context(data_source=data_source)

    if real_consensus_df is None:
        print("[permutation_null_consensus] real_consensus_df non fornito: ricalcolo il "
              "consensus reale con la stessa pipeline (nessuna permutazione).")
        real_consensus_df = _one_consensus_run(
            ctx["y"], ctx, random_state=random_state,
            n_bootstrap_stability=n_bootstrap_stability, n_folds_shap=n_folds_shap,
            n_jobs_model=-1, verbose=False
        )

    n_specs = 1
    for vals in sc.SPEC_GRID.values():
        n_specs *= len(vals)
    n_specs *= len(sc.MODEL_TYPES)
    print(f"\n[permutation_null_consensus] {n_permutations} permutazioni, ognuna con: "
          f"{n_bootstrap_stability} bootstrap stability + {n_folds_shap}-fold SHAP + "
          f"l'intera specification curve (~{n_specs} combinazioni specifica x modello, "
          f"ognuna con n_folds fit). Può richiedere molto tempo.")

    rng = np.random.RandomState(random_state)
    perm_seeds = [rng.randint(0, 1_000_000) for _ in range(n_permutations)]

    def _run_one_permutation(perm_seed):
        prng = np.random.RandomState(perm_seed) # generatore casuale indipendente per specifica 
            # permutazione.
        y_perm = pd.Series(prng.permutation(ctx["y"].to_numpy()), index=ctx["y"].index)
        perm_consensus_df = _one_consensus_run(
            y_perm, ctx, random_state=perm_seed,
            n_bootstrap_stability=n_bootstrap_stability, n_folds_shap=n_folds_shap,
            n_jobs_model=n_jobs_model, verbose=False
        )
        return float(perm_consensus_df["consensus_score"].max())
            # Tu non vuoi sapere: "Quanto è alto mediamente il consensus delle feature sotto 
            # il caso?" ma vuoi sapere: "Quanto può diventare alto il consensus della feature 
            # più fortunata anche quando tutto è casuale?". Perché nella realtà tu non guardi 
            # una feature scelta prima, guardi centinaia di feature e poi dici: "Questa è 
            # quella con consensus più alto!". Quindi devi tenere conto del multiple testing 
            # implicito nella ricerca della feature migliore. Ecco perché il file parla di 
            # max-statistic / Westfall–Young.

    null_max_scores = np.array(Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_one_permutation)(seed) for seed in perm_seeds
    ))

    # Westfall-Young: confronta OGNI feature reale con la distribuzione del
    # MASSIMO (non con la propria distribuzione individuale) — controlla il
    # tasso di falsi positivi sull'intero esperimento, non sulla singola feature.
    m = len(null_max_scores)
    result_df = real_consensus_df.copy()
    result_df["p_value_fwer"] = result_df["consensus_score"].apply(
        lambda s: (int(np.sum(null_max_scores >= s)) + 1) / (m + 1)
    )
    result_df["significant_fwer"] = result_df["p_value_fwer"] < 0.05
        # Per ogni punteggio reale (s) nella colonna consensus_score, viene applicata una 
        # formula:np.sum(null_max_scores >= s): Conta quante volte nei dati simulati 
        # (ipotesi nulla) è stato ottenuto un punteggio massimo maggiore o uguale al punteggio
        # reale s. /(m + 1): Divide per il numero totale di simulazioni per trasformare il 
        # conteggio in una proporzione (probabilità).

    min_p = 1 / (m + 1)
    print(f"\n[permutation_null_consensus] consensus_score massimo nullo (permutato): "
          f"{null_max_scores.mean():.4f} ± {null_max_scores.std():.4f} "
          f"(range [{null_max_scores.min():.4f}, {null_max_scores.max():.4f}])")
    n_sig = int(result_df["significant_fwer"].sum())
    print(f"[permutation_null_consensus] {n_sig}/{len(result_df)} feature sopravvivono alla "
          f"soglia FWER (p<0.05) su {m} permutazioni | p-value minimo raggiungibile = "
          f"1/{m+1} = {min_p:.3f}")
    if min_p >= 0.05:
        print(f"[permutation_null_consensus] ATTENZIONE: con solo {m} permutazioni il p-value "
              f"minimo possibile ({min_p:.3f}) è già >= 0.05: NESSUNA feature può risultare "
              f"'significativa' con questa soglia, qualunque sia il suo consensus_score. Usa "
              f"comunque la distribuzione nulla come riferimento qualitativo (dove cade il "
              f"consensus_score reale rispetto al range ottenuto per caso?), e alza "
              f"config.CONSENSUS_N_PERMUTATIONS se ti serve un p-value formale.")

    return result_df, null_max_scores


if __name__ == "__main__":
    out_dir = config.OUTPUT_DIR / "both" / "consensus_significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    consensus_path = config.OUTPUT_DIR / "both" / "feature_consensus.csv"
    real_consensus_df = None
    if consensus_path.exists():
        real_consensus_df = pd.read_csv(consensus_path, index_col=0)
    else:
        print(f"[consensus_significance] {consensus_path} non trovato: verrà ricalcolato "
              f"da zero (esegui prima feature_consensus.py per evitare di rifare due volte "
              f"lo stesso lavoro).")

    result_df, null_max_scores = permutation_null_consensus(real_consensus_df=real_consensus_df)

    result_df.to_csv(out_dir / "feature_consensus_with_pvalues.csv")
    pd.Series(null_max_scores, name="consensus_score_max_permutato").to_csv(
        out_dir / "null_max_consensus_distribution.csv", index=False
    )

    print("\nTop 15 feature per consenso, con p-value FWER:")
    print(result_df[["consensus_score", "n_criteria_present", "p_value_fwer"]].head(15))
    print(f"\nRisultati salvati in: {out_dir}")