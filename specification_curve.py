"""
Specification Curve Analysis (SCA) + robustezza delle feature attraverso le
scelte metodologiche ("multiverse analysis").

Per ogni combinazione di preprocessing (data_source, selezione geni, shape,
soglia di ridondanza) e per due modelli (Elastic Net lineare, Random Forest)
calcoliamo:
- AUC media sui fold e AUC pooled out-of-fold
- statistiche complete sull'importanza di ogni feature attraverso i fold di
  quella specifica 
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import binomtest
from collections import Counter

import config
import data_utils

# ---------------------------------------------------------------------------
# SPAZIO DELLE SPECIFICHE — modifica/estendi liberamente
# ---------------------------------------------------------------------------
SPEC_GRID = {
    "data_source": ["radiomics", "genomics", "both"],
    "gene_selection_method": ["variance", "iqr_top_pct", "iqr_top_n"],
    "exclude_shape": [True, False],
    "redundancy_corr_threshold": [0.90, 0.95],
}

# Griglia ridotta usata SOLO dal test di significatività congiunto
# (joint_significance_test): la curva va rifatta N_PERMUTATIONS_SPEC_CURVE
# volte, quindi qui si tiene solo un asse di scelte "extra" (gene_selection)
# a 2 livelli invece di 3 e una sola soglia di ridondanza, mantenendo
# comunque tutte e tre le sorgenti dati (è la scelta più rilevante per
# l'interpretazione) ed entrambi i modelli.
REDUCED_SPEC_GRID = {
    "data_source": ["radiomics", "genomics", "both"],
    "gene_selection_method": ["variance", "iqr_top_pct"],
    "exclude_shape": [True, False],
    "redundancy_corr_threshold": [0.90],
}

# i "due best" modelli: lineare interpretabile via coefficienti, e ad
# albero interpretabile via feature_importances_. Iperparametri fissi per
# ogni specifica (niente grid search qui dentro).
MODEL_TYPES = ["linear", "tree"]


def _build_pipe(model_type: str, fixed_params: dict = None):
    if model_type == "linear":
        if fixed_params:
            # Usa iperparametri congelati (nessun tuning)
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="elasticnet", solver="saga", max_iter=5000,
                    C=fixed_params["C"], l1_ratio=fixed_params["l1_ratio"],
                    random_state=config.RANDOM_STATE
                )),
            ])
        else: 
            # tuning
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegressionCV(
                    penalty="elasticnet", solver="saga", max_iter=5000,
                    Cs=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
                    l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
                    cv=3, scoring="roc_auc", 
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,  # parallelizza le 6x5x3=90 combinazioni della grid search
                                # interna sui core disponibili — stesso risultato, più veloce
                )),
            ])
    elif model_type == "tree":
        return Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=3, min_samples_leaf=5,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,  # parallelizza la costruzione dei 200 alberi sui core disponibili
            )),
        ])
    else:
        raise ValueError(f"model_type '{model_type}' non valido (usa 'linear' o 'tree')")


def _extract_importance(fitted_pipe, model_type: str, columns) -> pd.Series:
    clf = fitted_pipe.named_steps["clf"] # nella pipe considera solo modello, escludi std scaler
    if model_type == "linear":
        return pd.Series(clf.coef_.ravel(), index=columns)
    else:
        # feature_importances_ del Random Forest sono sempre >= 0 (non hanno segno,
        # misurano riduzione di impurità), a differenza dei coefficienti lineari
        # questa proprietà fornisce una misura dell'importanza delle feature basata sulla 
        # riduzione di impurità degli alberi. In termini intuitivi, feature molto usata 
        # per fare split e split che migliorano molto la separazione
        return pd.Series(clf.feature_importances_, index=columns)


def _cv_eval(X: pd.DataFrame, y_bin: pd.Series, model_type: str,
             n_folds: int = 5, random_state: int = config.RANDOM_STATE,
             fixed_params: dict = None):
    """
    k-fold CV semplice per una specifica + un modello. Ritorna:
    - auc_mean, auc_sd: media/sd dell'AUC calcolata fold per fold
    - auc_pooled: AUC su tutte le predizioni out-of-fold concatenate 
    - coef_matrix: DataFrame (feature x fold) con l'importanza per fold,
      base per le statistiche del punto 2
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs = []
    coef_frames = []
    oof_proba = np.full(len(X), np.nan)
    
    fold_params = []

    for train_idx, test_idx in cv.split(X, y_bin):
        pipe = _build_pipe(model_type, fixed_params=fixed_params)
        pipe.fit(X.iloc[train_idx], y_bin.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        
        if model_type == "linear" and fixed_params is None:
            clf = pipe.named_steps["clf"]
            fold_params.append({
                "C": float(clf.C_[0]),
                "l1_ratio": float(clf.l1_ratio_[0])
            })

        aucs.append(roc_auc_score(y_bin.iloc[test_idx], proba))
        oof_proba[test_idx] = proba
        coef_frames.append(_extract_importance(pipe, model_type, X.columns))
        
    chosen_params = None
    if model_type == "linear":
        if fixed_params is None:
            param_tuples = [tuple(sorted(p.items())) for p in fold_params]
                # prende elem da fold_param, ogni elem è un dict. li ordina secondo il valore 
                # poi li rende una tupla (oggetto immutabile e "contabile")
            most_common, _ = Counter(param_tuples).most_common(1)[0]
                # conta quante volte compare ogni elemento, salva il più freq [0], siccome
                # most common restituisce una lista con il più freq e la freq, [0] oggetto e
                # [1] la freq (secondo elemento)
            chosen_params = dict(most_common)
                # ritrasformo in un comodo dict
        else:
            chosen_params = fixed_params

    auc_pooled = roc_auc_score(y_bin, oof_proba)
    coef_matrix = pd.concat(coef_frames, axis=1)
    coef_matrix.columns = [f"fold_{i}" for i in range(coef_matrix.shape[1])]

    return float(np.mean(aucs)), float(np.std(aucs)), float(auc_pooled), coef_matrix, chosen_params


def _feature_stats(coef_matrix: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """
    Statistiche sull'importanza di ogni feature attraverso i fold di una
    specifica. La sola media può andare vicino a zero per una feature che
    oscilla di segno tra fold pur essendo sistematicamente "usata" dal
    modello: mean_abs e fraction_nonzero catturano questo caso.
    """
    stats = pd.DataFrame({
        "mean_coefficient": coef_matrix.mean(axis=1),
        "std_coefficient": coef_matrix.std(axis=1),
        "mean_abs_coefficient": coef_matrix.abs().mean(axis=1),
        "fraction_nonzero": (coef_matrix.abs() > 1e-8).mean(axis=1),
    })
    if model_type == "linear":
        stats["fraction_positive"] = (coef_matrix > 1e-8).mean(axis=1)
        stats["fraction_negative"] = (coef_matrix < -1e-8).mean(axis=1)
    else:
        # Random Forest: le importances non hanno segno, quindi la
        # distinzione positive/negative non esiste per questo modello
        stats["fraction_positive"] = np.nan
        stats["fraction_negative"] = np.nan
    return stats


def run_specification_curve(spec_grid=None, model_types=None,
                             n_folds: int = 5, top_n_features: int = 15,
                             permute_labels: bool = False,
                             random_state: int = config.RANDOM_STATE,
                             fixed_params_dict: dict = None,
                             verbose: bool = True):
    """
    Parametri aggiuntivi
    --------------------
    permute_labels : se True, rimescola le etichette PRIMA di valutare
        qualunque specifica. Usato dal test di significatività congiunto
        (joint_significance_test) per costruire la distribuzione nulla:
        una singola permutazione per data_source viene fissata all'inizio
        di questa chiamata e riusata per tutte le combinazioni di quella
        sorgente, così la curva permutata è confrontabile con quella reale
        (stessa struttura, solo etichette senza informazione).
    random_state : seed per la permutazione delle etichette (ignorato se
        permute_labels=False).

    Ritorna
    -------
    spec_df : una riga per ogni combinazione (specifica x modello), con
        auc_mean_fold, auc_sd_fold, auc_pooled, n_features
    feature_votes : Series feature -> numero di combinazioni (specifica x
        modello, TUTTI i modelli insieme) in cui la feature è tra le top_n
    feature_votes_by_model : dict {model_type: Series}, stesso conteggio
        ma separato per modello (utile perché lineare e RF possono
        enfatizzare feature diverse per motivi strutturali, non è detto
        che "unire" i voti sia sempre la lettura giusta)
    feature_stats_long_df : DataFrame in formato lungo con TUTTE le
        statistiche del punto 2, per ogni specifica e ogni modello —
        salvalo, è la base per qualsiasi controllo più fine dopo
    """
    spec_grid = spec_grid or SPEC_GRID
    model_types = model_types or MODEL_TYPES
    keys = list(spec_grid.keys())
    combos = list(itertools.product(*spec_grid.values()))

    cache_raw = {}
    
    permuted_y_cache = {}  # data_source -> y_bin permutata 
    perm_rng = np.random.RandomState(random_state) if permute_labels else None

    spec_rows = []
    feature_votes = pd.Series(dtype=float)
    feature_votes_by_model = {m: pd.Series(dtype=float) for m in model_types}
    feature_stats_long = []

    for combo in combos:
        spec = dict(zip(keys, combo))

        if spec["data_source"] not in cache_raw:
            cache_raw[spec["data_source"]] = data_utils.load_data(source=spec["data_source"], 
                                                                  print_info = False)
        X_raw, y = cache_raw[spec["data_source"]]

        X_reduced = data_utils.neutral_feature_reduction(
            X_raw,
            gene_selection_method=spec["gene_selection_method"],
            exclude_shape=spec["exclude_shape"],
            redundancy_corr_threshold=spec["redundancy_corr_threshold"],
            print_info = False
        )
        y_bin = (y == config.POSITIVE_CLASS).astype(int)
        
        if permute_labels:
            if spec["data_source"] not in permuted_y_cache:
                # permuto i valori mantenendo l'indice/ordine dei pazienti invariato,
                # cosi' resta allineato riga-per-riga con X_reduced per qualunque 
                # spec di questa stessa data_source
                shuffled_values = perm_rng.permutation(y_bin.to_numpy()) # permuto gli y_bin
                permuted_y_cache[spec["data_source"]] = pd.Series(shuffled_values,
                                                                  index=y_bin.index)
                    # salvo la y_cache permutata con questa serie di valuri shuffled, ma 
                    # con gli stessi indici di y_bin, salvo solo l'ordine casuale delle y
            y_bin = permuted_y_cache[spec["data_source"]]


        for model_type in model_types:
            if verbose:
                print(f"[specification_curve] {spec} | model={model_type}")
            
            spec_key = tuple(spec.values()) + (model_type,)
            current_fixed_params = None
            if fixed_params_dict is not None and spec_key in fixed_params_dict:
                current_fixed_params = fixed_params_dict[spec_key]
            
            auc_mean, auc_sd, auc_pooled, coef_matrix, chosen_params = _cv_eval(
                X_reduced, y_bin, model_type, n_folds,
                random_state=config.RANDOM_STATE,
                fixed_params=current_fixed_params
            )
            fstats = _feature_stats(coef_matrix, model_type)
            
            collapsed = bool((fstats["fraction_nonzero"] == 0).all())
            if collapsed:
                print(f"[specification_curve] ATTENZIONE: modello collassato (nessuna feature "
                      f"selezionata in nessun fold) per spec={spec} model={model_type} — "
                      f"l'AUC per questa riga non riflette segnale, va scartata o rifatta con "
                      f"regolarizzazione più debole.")

            top_feats = fstats["mean_abs_coefficient"].sort_values(
                ascending=False
            ).head(top_n_features).index
            feature_votes = feature_votes.add(pd.Series(1.0, index=top_feats), fill_value=0)
            feature_votes_by_model[model_type] = feature_votes_by_model[model_type].add(
                pd.Series(1.0, index=top_feats), fill_value=0
            )

            fstats_named = fstats.reset_index().rename(columns={"index": "feature"})
            for k, v in spec.items():
                fstats_named[k] = v
            fstats_named["model_type"] = model_type
            feature_stats_long.append(fstats_named)

            spec_rows.append({
                **spec, "model_type": model_type,
                "auc_mean_fold": auc_mean, "auc_sd_fold": auc_sd,
                "auc_pooled": auc_pooled, "n_features": X_reduced.shape[1],
                "collapsed_model": collapsed,
                "best_params": chosen_params
            })

    spec_df = pd.DataFrame(spec_rows).sort_values("auc_pooled").reset_index(drop=True)
    feature_votes = feature_votes.sort_values(ascending=False)
    for m in feature_votes_by_model:
        feature_votes_by_model[m] = feature_votes_by_model[m].sort_values(ascending=False)
    feature_stats_long_df = pd.concat(feature_stats_long, ignore_index=True)

    print(f"\n[specification_curve] {len(spec_df)} combinazioni (specifica x modello) testate | "
          f"AUC pooled min={spec_df['auc_pooled'].min():.3f} max={spec_df['auc_pooled'].max():.3f}\n")

    return spec_df, feature_votes, feature_votes_by_model, feature_stats_long_df

# ---------------------------------------------------------------------------
# TEST DI SIGNIFICATIVITÀ CONGIUNTO (parte inferenziale della SCA)
#
# La curva descrittiva (run_specification_curve) risponde a "come cambia il
# risultato al variare delle scelte metodologiche?". Questa funzione risponde
# a una domanda diversa e più severa: "il pattern di risultati attraverso
# tutte le specifiche insieme è più forte di quanto ci si aspetterebbe per
# puro caso?" — permutando le etichette e rifacendo l'intera curva molte
# volte (Simonsohn, Simmons & Nelson 2020).
# ---------------------------------------------------------------------------
def _summarize_curve(auc_pooled: pd.Series, stat: str) -> float:
    """
    Statistica riassuntiva di una curva di specifiche (una per permutazione + una reale).
    """
    if stat == "median":
        return float(auc_pooled.median())
    elif stat == "mean":
        return float(auc_pooled.mean())
    else:
        raise ValueError(f"summary_stat '{stat}' non valida (usa 'median' o 'mean')")
        
        
def _empirical_p_with_ci(null_stats: np.ndarray, real_stat: float, confidence_level: float = 0.95):
    """
    P-value empirico da permutazione + intervallo di confidenza sulla sua incertezza.
 
    Il punto stimato usa la correzione standard (b+1)/(m+1) di Phipson & Smyth
    (2010) — lo standard per evitare p=0 con un numero finito di permutazioni. 
    Quello che un singolo numero non dice è quanto ci si può fidare di quella 
    stima con un m piccolo: il conteggio "quante permutazioni >= al reale" è 
    un conteggio binomiale, per cui possiamo chiedere a scipy l'intervallo di 
    confidenza esatto (Clopper-Pearson) sulla vera proporzione. 
    Con m piccolo questo intervallo sarà enorme — salire con N_PERMUTATIONS.
 
    Ritorna
    -------
    p_value : stima puntuale, formula (b+1)/(m+1)
    ci_low, ci_high : CI esatta (Clopper-Pearson) sulla proporzione b/m
    n_successes : quante permutazioni hanno eguagliato/superato il reale (b)
    """
    m = len(null_stats)
    b = int(np.sum(null_stats >= real_stat))
    p_value = (b + 1) / (m + 1)
    ci = binomtest(b, m, alternative="two-sided").proportion_ci(
        confidence_level=confidence_level, method="exact"
    )
    return p_value, ci.low, ci.high, b

 
def joint_significance_test(spec_grid=None, model_types=None, n_folds: int = 5,
                             n_permutations: int = None, summary_stat: str = None,
                             random_state: int = config.RANDOM_STATE):
    """
    Costruisce la distribuzione nulla della specification curve permutando
    le etichette e rifacendo l'INTERA curva n_permutations volte, poi
    confronta la statistica riassuntiva (mediana o media di auc_pooled) della
    curva reale con quella distribuzione.
 
    Per costo computazionale usa di default REDUCED_SPEC_GRID invece di
    SPEC_GRID: sia la curva reale sia tutte le curve permutate vengono
    valutate sulla stessa griglia ridotta, altrimenti il confronto tra
    statistica reale e nulla non sarebbe corretto.
 
    Ritorna
    -------
    real_spec_df : la curva reale (sulla griglia usata per il test, non
        necessariamente la stessa di run_specification_curve.py)
    real_stat : statistica riassuntiva osservata
    null_stats : array (n_permutations,) con la statistica per ogni curva permutata
    p_value : frazione di curve permutate con statistica >= a quella osservata
        (+1 correzione, come in diagnostics.permutation_test)
    """
    spec_grid = spec_grid or REDUCED_SPEC_GRID
    model_types = model_types or MODEL_TYPES
    n_permutations = n_permutations or config.N_PERMUTATIONS_SPEC_CURVE
    summary_stat = summary_stat or config.SPEC_CURVE_SUMMARY_STAT
 
    n_combos = len(list(itertools.product(*spec_grid.values()))) * len(model_types)
    print(f"\n[joint_significance_test] curva reale su griglia ridotta "
          f"({n_combos} combinazioni specifica x modello).")
    real_spec_df, _, _, _ = run_specification_curve(
        spec_grid=spec_grid, model_types=model_types, n_folds=n_folds, verbose=False
    )
    real_stat = _summarize_curve(real_spec_df["auc_pooled"], summary_stat)
    
    # Mappa i parametri scelti sui dati reali per tutte le configurazioni
    spec_keys = list(spec_grid.keys())
    fixed_params_dict = {}
    for _, row in real_spec_df.iterrows(): #  method that loops through rows. 
            # It yields a pair for each row: the row index and a Pandas Series with the row data.
        key = tuple(row[k] for k in spec_keys) + (row["model_type"],)
        fixed_params_dict[key] = row["best_params"]
        
    print(f"[joint_significance_test] statistica riassuntiva reale "
          f"({summary_stat} di auc_pooled) = {real_stat:.4f}")
 
    print(f"[joint_significance_test] {n_permutations} permutazioni x {n_combos} "
          f"combinazioni = {n_permutations * n_combos} fit totali.")
 
    rng = np.random.RandomState(random_state)
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_seed = rng.randint(0, 1000000)
        perm_spec_df, _, _, _ = run_specification_curve(
            spec_grid=spec_grid, model_types=model_types, n_folds=n_folds,
            permute_labels=True, random_state=perm_seed,
            fixed_params_dict=fixed_params_dict,
            verbose=False
        )
        null_stats[i] = _summarize_curve(perm_spec_df["auc_pooled"], summary_stat)
        if (i + 1) % 10 == 0:
            print(f"[joint_significance_test] {i+1}/{n_permutations} permutazioni completate")
 
    # p-value empirico (Phipson & Smyth) + CI esatta (Clopper-Pearson) sulla sua incertezza
    p_value, p_ci_low, p_ci_high, n_successes = _empirical_p_with_ci(null_stats, real_stat)
 
    print(f"\n[joint_significance_test] {summary_stat} nullo (permutato): "
          f"{null_stats.mean():.4f} ± {null_stats.std():.4f}")
    print(f"[joint_significance_test] {summary_stat} osservato: {real_stat:.4f}")
    print(f"[joint_significance_test] p-value empirico: {p_value:.4f} "
          f"({n_successes}/{n_permutations} permutazioni >= al reale) | "
          f"CI 95% esatta (Clopper-Pearson): [{p_ci_low:.4f}, {p_ci_high:.4f}]")
    
    if p_ci_high - p_ci_low > 0.15:
        print("[joint_significance_test] ATTENZIONE: il CI sul p-value è molto ampio — "
              "con questo numero di permutazioni la stima è poco precisa, considera di "
              "aumentare N_PERMUTATIONS_SPEC_CURVE prima di trarre conclusioni forti.")
 
    if p_value >= 0.05:
        print("[joint_significance_test] ATTENZIONE: il pattern di risultati attraverso tutte "
              "le specifiche non si distingue in modo significativo da quello ottenibile "
              "permutando le etichette. Il segnale complessivo va trattato con forte cautela.")
    else:
        print("[joint_significance_test] Il pattern di risultati attraverso le specifiche è "
              "più forte di quanto atteso per puro caso: evidenza congiunta di segnale reale, "
              "più robusta di un singolo permutation test su un solo modello.")
 
    return real_spec_df, real_stat, null_stats, p_value, p_ci_low, p_ci_high


def joint_significance_test_source_comparison(source_a: str = "genomics", source_b: str = "both",
                                                spec_grid=None, model_types=None, n_folds: int = 5,
                                                n_permutations: int = None, summary_stat: str = None,
                                                random_state: int = config.RANDOM_STATE):
    """
    Variante appaiata del test di significatività congiunto: non chiede "il
    segnale è più forte del caso?" (joint_significance_test) ma "SOURCE_B
    aggiunge sistematicamente segnale rispetto a SOURCE_A, attraverso tutte
    le combinazioni della griglia di specifiche, più di quanto ci si
    aspetterebbe permutando le etichette?". Generalizza il confronto singolo
    di compare_data_sources.py (un solo modello, un solo preprocessing) a
    tutta la griglia — risposta più robusta perché non dipende da una
    scelta arbitraria di specifica.
 
    Design appaiato (stessa logica di compare_data_sources.py):
    - stessi pazienti per le due sorgenti: dato che "both" è già
      un'intersezione radiomica/genomica, i suoi pazienti sono un
      sottoinsieme di quelli di "genomics" da sola — si usa quindi
      l'intersezione.
    - stessa permutazione delle etichette per le due sorgenti ad ogni
      iterazione, e stessa suddivisione in fold (stessi pazienti, stesso
      random_state -> fold identici): così la differenza isola l'effetto
      della sorgente dati, non il rumore di ricampionamento.
    - Iperparametri lineari congelati sulla curva reale e riusati identici
      in tutte le permutazioni.
 
    Statistica: per ogni combinazione (spec x modello),
    auc_pooled(source_b) - auc_pooled(source_a); la curva è la
    mediana/media di queste differenze attraverso le specifiche.
 
    Ritorna
    -------
    real_diff_df : DataFrame, una riga per specifica x modello, con
        auc_pooled_a, auc_pooled_b, diff = auc_pooled_b - auc_pooled_a
    real_stat, null_stats, p_value, p_ci_low, p_ci_high : come joint_significance_test
    """
    spec_grid = spec_grid or {k: v for k, v in REDUCED_SPEC_GRID.items() if k != "data_source"}
        # .items() prende tutte le coppie chiave (k) e valore (v), ricostruisce un dict escludendo
        # la chiave "data_source"
    model_types = model_types or MODEL_TYPES
    n_permutations = n_permutations or config.N_PERMUTATIONS_SPEC_CURVE
    summary_stat = summary_stat or config.SPEC_CURVE_SUMMARY_STAT
 
    # --- Allineamento pazienti tra le due sorgenti (confronto appaiato) ---
    X_raw_a, y_a = data_utils.load_data(source=source_a, print_info=False)
    X_raw_b, y_b = data_utils.load_data(source=source_b, print_info=False)
    common_idx = X_raw_a.index.intersection(X_raw_b.index)
    n_dropped = min(len(X_raw_a), len(X_raw_b)) - len(common_idx)
    if n_dropped > 0:
        print(f"\n[joint_significance_test_source_comparison] ATTENZIONE: {n_dropped} pazienti "
              f"presenti in una sola delle due sorgenti sono stati esclusi dal confronto appaiato.")
    X_raw_a = X_raw_a.loc[common_idx].sort_index()
    X_raw_b = X_raw_b.loc[common_idx].sort_index()
    y_a, y_b = y_a.loc[common_idx].sort_index(), y_b.loc[common_idx].sort_index()
    if not y_a.equals(y_b):
        raise RuntimeError("Le etichette non coincidono tra le due sorgenti per gli stessi pazienti: "
                            "controlla l'allineamento in data_utils.load_data.")
    y_bin_base = (y_a == config.POSITIVE_CLASS).astype(int)
 
    keys = list(spec_grid.keys())
    combos = list(itertools.product(*spec_grid.values())) 
        # itertools.product(...): funzione del modulo itertools che genera le combinazioni fra gli 
        # elementi degli iterabili passati
    n_combos = len(combos) * len(model_types)
 
    # Riduzione feature (indipendente dalla label): calcolata una sola volta per
    # ogni specifica e riusata identica per la curva reale e per tutte le
    # permutazioni (non dipende in alcun modo dalle etichette).
    reduction_cache_a, reduction_cache_b = {}, {}
    for combo in combos:
        spec = dict(zip(keys, combo))
        reduction_cache_a[combo] = data_utils.neutral_feature_reduction(
            X_raw_a, gene_selection_method=spec.get("gene_selection_method"),
            exclude_shape=spec.get("exclude_shape"),
            redundancy_corr_threshold=spec.get("redundancy_corr_threshold"),
            print_info=False)
        reduction_cache_b[combo] = data_utils.neutral_feature_reduction(
            X_raw_b, gene_selection_method=spec.get("gene_selection_method"),
            exclude_shape=spec.get("exclude_shape"),
            redundancy_corr_threshold=spec.get("redundancy_corr_threshold"),
            print_info=False)
 
    def _one_diff_curve(y_bin, fixed_params_dict=None):
        rows, params_out = [], {}
        for combo in combos:
            spec = dict(zip(keys, combo))
            X_a, X_b = reduction_cache_a[combo], reduction_cache_b[combo]
            for model_type in model_types:
                spec_key = combo + (model_type,)
                fp_a = fixed_params_dict.get((spec_key, "a")) if fixed_params_dict else None
                fp_b = fixed_params_dict.get((spec_key, "b")) if fixed_params_dict else None
                    # cerca un oggetto della forma (chiave, "a/b") nel dict dei param
                _, _, auc_pooled_a, _, chosen_a = _cv_eval(
                    X_a, y_bin, model_type, n_folds, random_state=config.RANDOM_STATE, fixed_params=fp_a)
                _, _, auc_pooled_b, _, chosen_b = _cv_eval(
                    X_b, y_bin, model_type, n_folds, random_state=config.RANDOM_STATE, fixed_params=fp_b)
                rows.append({**spec, "model_type": model_type,
                             "auc_pooled_a": auc_pooled_a, "auc_pooled_b": auc_pooled_b,
                             "diff": auc_pooled_b - auc_pooled_a})
                params_out[(spec_key, "a")] = chosen_a
                params_out[(spec_key, "b")] = chosen_b
        return pd.DataFrame(rows), params_out
 
    print(f"\n[joint_significance_test_source_comparison] curva reale ({source_b} - {source_a}) "
          f"su {n_combos} combinazioni specifica x modello.")
    real_diff_df, fixed_params_dict = _one_diff_curve(y_bin_base, fixed_params_dict=None)
    real_stat = _summarize_curve(real_diff_df["diff"], summary_stat)
    print(f"[joint_significance_test_source_comparison] {summary_stat} osservato della differenza "
          f"({source_b} - {source_a}) = {real_stat:+.4f}")
    print(f"[joint_significance_test_source_comparison] {n_permutations} permutazioni x {n_combos} "
          f"combinazioni x 2 sorgenti.")
 
    rng = np.random.RandomState(random_state)
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_seed = rng.randint(0, 1000000)
        prng = np.random.RandomState(perm_seed)
        y_bin_perm = pd.Series(prng.permutation(y_bin_base.to_numpy()), index=y_bin_base.index)
        perm_diff_df, _ = _one_diff_curve(y_bin_perm, fixed_params_dict=fixed_params_dict)
        null_stats[i] = _summarize_curve(perm_diff_df["diff"], summary_stat)
        if (i + 1) % 10 == 0:
            print(f"[joint_significance_test_source_comparison] {i+1}/{n_permutations} permutazioni completate")
 
    p_value, p_ci_low, p_ci_high, n_successes = _empirical_p_with_ci(null_stats, real_stat)
 
    print(f"\n[joint_significance_test_source_comparison] {summary_stat} nullo (permutato): "
          f"{null_stats.mean():+.4f} ± {null_stats.std():.4f}")
    print(f"[joint_significance_test_source_comparison] {summary_stat} osservato: {real_stat:+.4f}")
    print(f"[joint_significance_test_source_comparison] p-value empirico: {p_value:.4f} "
          f"({n_successes}/{n_permutations} permutazioni >= al reale) | "
          f"CI 95% esatta: [{p_ci_low:.4f}, {p_ci_high:.4f}]")
 
    if p_value >= 0.05:
        print(f"[joint_significance_test_source_comparison] ATTENZIONE: non c'è evidenza congiunta "
              f"(attraverso le specifiche testate) che {source_b} aggiunga segnale rispetto a "
              f"{source_a}. Coerente con quanto già visto in compare_data_sources.py, ma qui su "
              f"più scelte di preprocessing insieme.")
    else:
        print(f"[joint_significance_test_source_comparison] Evidenza congiunta che {source_b} "
              f"aggiunge segnale reale rispetto a {source_a}, robusta attraverso le scelte di "
              f"preprocessing testate — più forte di un singolo confronto DeLong/bootstrap.")
 
    return real_diff_df, real_stat, null_stats, p_value, p_ci_low, p_ci_high
 
# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
def plot_joint_significance_test(null_stats: np.ndarray, real_stat: float, summary_stat: str,
                                 output_path, xlabel: str = None, title: str = None):
    """Istogramma della distribuzione nulla della statistica riassuntiva, con la statistica osservata."""
    plt.figure(figsize=(7, 5))
    plt.hist(null_stats, bins=20, color="#8C8C8C", edgecolor="white",
              label="curve con etichette permutate")
    plt.axvline(real_stat, color="#C44E52", linewidth=2,
                label=f"{summary_stat} osservato = {real_stat:.3f}")
    plt.xlabel(xlabel or f"{summary_stat} di auc_pooled attraverso le specifiche")
    plt.ylabel("Numero di permutazioni")
    plt.title(title or "Test di significatività congiunto sulla specification curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_joint_significance_test] salvato in {output_path}")
    
    
def plot_specification_curve(spec_df: pd.DataFrame, spec_keys: list, output_path):
    """
    spec_keys deve includere anche "model_type" se vuoi vederlo nel pannello
    inferiore insieme alle altre scelte di preprocessing.
    """
    spec_df = spec_df.sort_values("auc_pooled").reset_index(drop=True)
    n_spec = len(spec_df)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(max(10, n_spec * 0.22), 8),
        gridspec_kw={"height_ratios": [2, 1.8]}, sharex=True
    )
    x = np.arange(n_spec)

    model_colors = {"linear": "#4C72B0", "tree": "#C44E52"}
    for model_type, color in model_colors.items():
        mask = (spec_df["model_type"] == model_type).values
        ax_top.errorbar(x[mask], spec_df.loc[mask, "auc_mean_fold"],
                         yerr=spec_df.loc[mask, "auc_sd_fold"],
                         fmt="o", color=color, alpha=0.35, markersize=3, capsize=2,
                         label=f"{model_type}: media fold ± sd")
        ax_top.scatter(x[mask], spec_df.loc[mask, "auc_pooled"],
                        color=color, marker="D", s=20, label=f"{model_type}: pooled OOF")

    ax_top.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="AUC=0.5 (caso)")
    ax_top.set_ylabel("AUC")
    ax_top.set_title("Specification curve: AUC media-fold vs pooled OOF, per specifica e modello")
    ax_top.legend(fontsize=8, ncol=2)

    colors = plt.cm.tab10.colors
    y_positions = {}
    row_offset = 0
    for key in spec_keys:
        levels = sorted(spec_df[key].astype(str).unique(), key=str)
        for level in levels:
            y_positions[(key, level)] = row_offset
            row_offset += 1
        row_offset += 0.5

    for xi, (_, row) in enumerate(spec_df.iterrows()):
        for key in spec_keys:
            yi = y_positions[(key, str(row[key]))]
            ax_bottom.scatter(xi, yi, color="#4C72B0", s=15)

    ax_bottom.set_yticks(list(y_positions.values()))
    ax_bottom.set_yticklabels([f"{k}={v}" for (k, v) in y_positions.keys()], fontsize=8)
    ax_bottom.set_xlabel("Specifiche ordinate per AUC pooled crescente")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_specification_curve] salvato in {output_path}")


def plot_feature_votes(feature_votes: pd.Series, n_total_specs: int, output_path, top_n: int = 25):
    top = (feature_votes / n_total_specs * 100).head(top_n).sort_values()
    plt.figure(figsize=(8, 0.3 * len(top) + 1))
    plt.barh(top.index, top.values, color="#55A868")
    plt.xlabel("% di combinazioni in cui la feature è tra le più importanti")
    plt.title("Robustezza delle feature attraverso le scelte metodologiche")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_feature_votes] salvato in {output_path}")