"""
Test di permutazione sulle specifiche delle specification curve v2 e di rete.

Due test, due domande diverse
-----------------------------
1) label_permutation_maxstat — "c'è segnale?"
   Si permutano le etichette dei pazienti (la STESSA permutazione per tutte le
   specifiche e i modelli, così si preserva la dipendenza tra di loro) e si
   rifà la stessa valutazione della curva (specification_curve._cv_eval,
   compreso il tuning interno di LogisticRegressionCV: nessun iperparametro
   congelato sulle etichette vere). Per ogni specifica si ottiene un p-value
   "grezzo" e due p-value corretti per l'aver provato molte specifiche
   (Westfall-Young maxT, single-step), a livello di famiglia (stessa
   data_source) e globale. Le statistiche vengono standardizzate per
   specifica con media/sd della distribuzione nulla, perché modelli diversi
   (lineare, albero) hanno nulli con media e varianza diverse.
   La famiglia conta: la radiomica da sola va giudicata contro le altre
   specifiche radiomiche, non contro la genomica (segnale molto più forte).

2) block_permutation_added_value — "la radiomica aggiunge qualcosa ai geni?"
   Si tengono fissi geni ed etichette e si permutano solo le RIGHE del blocco
   radiomico (ogni paziente riceve il vettore radiomico di un altro paziente,
   preservando la struttura interna della radiomica). La statistica è
   AUC(geni + radiomica) - AUC(solo geni). L'ipotesi nulla è che la radiomica
   non porti informazione né sulle etichette né sui geni. Differisce dal
   test congiunto della specification curve esistente, che permuta le
   etichette per entrambe le sorgenti e quindi azzera anche il segnale dei geni.

Parallelismo: sulle permutazioni (joblib, processi separati); dentro ogni
valutazione n_jobs_model=1 per non annidare il parallelismo.
Risoluzione minima del p-value: 1/(n_perm+1).
"""

import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
import config_network_selection as cn
import network_feature_selection as nfs
import reduction_v2
import specification_curve as sc
import specification_curve_v2 as sc2


# ---------------------------------------------------------------------------
# COSTRUZIONE DELLE SPECIFICHE (riduzioni label-free, calcolate una volta sola)
# ---------------------------------------------------------------------------
def _split_cols(X):
    rad = [c for c in X.columns if c.startswith("rad__")]
    gen = [c for c in X.columns if c.startswith("gen__")]
    return rad, gen


def build_specs_v2(X: pd.DataFrame) -> dict:
    """Specifiche di specification_curve_v2 sugli stessi pazienti di X (source='both')."""
    rad, gen = _split_cols(X)
    cols = {"radiomics": rad, "genomics": gen, "both": rad + gen}
    specs = {}
    for combo in itertools.product(*sc2.SPEC_GRID_V2.values()):
        spec = dict(zip(sc2.SPEC_GRID_V2.keys(), combo))
        # Accoppia i nomi dei parametri (le chiavi della griglia) con i valori della combo corrente
        if not sc2._is_valid_spec(spec):
            continue
        name = f"{spec['data_source']}|{spec['reduction_mode']}|{spec['budget']}"
        specs[name] = {"X": sc2._reduce(X[cols[spec["data_source"]]], spec),
                       "family": spec["data_source"]}
    return specs


def build_net_blocks(X: pd.DataFrame):
    """
    Blocchi della curva di rete: (X_genes, rad_sets) con rad_sets[nome] = matrice
    delle sole feature radiomiche scelte (label-free). Stessa logica di
    specification_curve_net.run_specification_curve_net.
    """
    rad, gen = _split_cols(X)
    X_genes = reduction_v2.BlockReducer(
        k_texture=None, k_shape=None, gen_prefilter_pct=config.GENE_IQR_TOP_PCT,
        **cn.GENE_REDUCTION).fit(X[gen]).transform(X[gen])

    nets = {}
    for name, path in cn.NETWORK_GRAPHML.items():
        G = nfs.load_network(path)
        scores = nfs.radiomic_node_scores(G)
        scores = scores.drop(index=[n for n in scores.index if n not in X.columns])
        nets[name] = (G, scores)

    rad_sets = {}
    for k in cn.K_RAD_LEVELS:
        kt, ks = sc2_k_split(k)
        for mode in ("medoid", "pca"):
            red = reduction_v2.BlockReducer(
                k_texture=kt, k_shape=ks, k_gen=None, mode=mode,
                separate_shape=config.EXCLUDE_SHAPE_FROM_MAIN_REDUCTION).fit(X[rad])
            rad_sets[f"v2_{mode}|-|k{k}"] = red.transform(X[rad])
        for net_name, (G, scores) in nets.items():
            for sel in cn.NET_SELECTORS:
                feats = nfs.select_radiomic_features(G, scores, sel, k, cn.REDUNDANCY_MAX_ABS_R)
                if feats:
                    rad_sets[f"{sel}|{net_name}|k{k}"] = X[feats]
    return X_genes, rad_sets


def sc2_k_split(k):
    """k radiomiche -> (k_texture, k_shape) per BlockReducer (come in specification_curve_net)."""
    if not config.EXCLUDE_SHAPE_FROM_MAIN_REDUCTION:
        return k, None
    ks = 1 if k < 8 else 2
    return k - ks, ks


def specs_from_net_blocks(X_genes, rad_sets) -> dict:
    """Specifiche (radiomics, both) + riferimento genomics, per il test sulle etichette."""
    specs = {"genomics|none|-": {"X": X_genes, "family": "genomics"}}
    for name, Xr in rad_sets.items():
        specs[f"radiomics|{name}"] = {"X": Xr, "family": "radiomics"}
        specs[f"both|{name}"] = {"X": pd.concat([Xr, X_genes], axis=1), "family": "both"}
    return specs


# ---------------------------------------------------------------------------
# VALUTAZIONE
# ---------------------------------------------------------------------------
def _auc(X, y_bin, model_type, n_folds, seed):
    """Stessa statistica della curva: AUC pooled OOF con tuning interno ai fold."""
    return sc._cv_eval(X.reset_index(drop=True), y_bin, model_type, n_folds,
                       random_state=seed, n_jobs_model=1)[2]


def _eval_label_perm(Xs, y_vals, model_types, n_folds, cv_seed, perm_seed):
    """Una permutazione delle etichette: AUC di tutte le (specifica, modello)."""
    yp = pd.Series(np.random.RandomState(perm_seed).permutation(y_vals))
    return {(name, mt): _auc(X, yp, mt, n_folds, cv_seed)
            for name, X in Xs.items() for mt in model_types}


def _eval_block_perm(rad_sets, X_genes, y_bin, model_types, n_folds, cv_seed, perm_seed):
    """Una permutazione delle righe del blocco radiomico: AUC(both) per ogni (set, modello)."""
    perm = np.random.RandomState(perm_seed).permutation(len(y_bin))
    G = X_genes.reset_index(drop=True)
    out = {}
    for name, Xr in rad_sets.items():
        Xp = pd.concat([Xr.reset_index(drop=True).iloc[perm].reset_index(drop=True), G], axis=1)
        # riordina le righe del blocco radiomico secondo l'ordine casuale perm. 
        # .reset_index(drop=True): reimposta l'indice a 0 ... N-1. Senza questo secondo reset, pd.concat 
        # cercherebbe di riallineare i dati usando gli indici originali, annullando di fatto la permutazione
        # .pd.concat([..., G], axis=1): unisce il blocco radiomico permutato con il blocco genomico intatto G.
        for mt in model_types:
            out[(name, mt)] = _auc(Xp, y_bin, mt, n_folds, cv_seed)
    return out


def _maxT_table(obs: pd.Series, null: pd.DataFrame, families: dict, label: str) -> pd.DataFrame:
    """
    obs  : statistica osservata per chiave (specifica, modello)
    null : n_perm x chiavi, stessa statistica sotto H0
    families : {chiave: nome famiglia}
    p_raw: p-value empirico unilaterale (1 + #null >= obs) / (n_perm + 1)
    p_adj_*: Westfall-Young maxT sulle statistiche standardizzate (z) per chiave.
    """
    n_perm = len(null)
    mu, sd = null.mean(), null.std(ddof=1).replace(0, np.nan)
    z_null = ((null - mu) / sd).fillna(0.0)
    z_obs = ((obs - mu) / sd).fillna(0.0)
    fam = pd.Series(families).reindex(null.columns)

    glob_max = z_null.max(axis=1)
    fam_max = {f: z_null.loc[:, fam == f].max(axis=1) for f in fam.unique()}

    rows = []
    for key in null.columns:
        rows.append({
            "spec": key[0], "model_type": key[1], "family": fam[key],
            label: obs[key],
            "null_mean": mu[key], "null_p95": float(null[key].quantile(.95)),
            "z_obs": z_obs[key],
            "p_raw": (1 + (null[key] >= obs[key]).sum()) / (n_perm + 1),
            "p_adj_family": (1 + (fam_max[fam[key]] >= z_obs[key]).sum()) / (n_perm + 1),
            "p_adj_global": (1 + (glob_max >= z_obs[key]).sum()) / (n_perm + 1),
        })
    return pd.DataFrame(rows).sort_values(label, ascending=False).reset_index(drop=True)


def _estimate(t_real, n_perm, n_jobs, tag):
    import os
    cores = os.cpu_count() or 1
    workers = cores if n_jobs in (-1, None) else n_jobs
    print(f"[{tag}] passaggio reale: {t_real:.0f}s | stima {n_perm} permutazioni su "
          f"~{workers} processi: ~{t_real * n_perm / 60:.0f} min "
          f"(la parte reale era già parallela sui core: ordine di grandezza)")


# ---------------------------------------------------------------------------
# TEST 1: etichette permutate + maxT su tutte le specifiche
# ---------------------------------------------------------------------------
def label_permutation_maxstat(specs: dict, y: pd.Series, model_types=None, n_perm: int = 100,
                              n_folds: int = 5, seed: int = config.RANDOM_STATE,
                              n_jobs: int = -1, dry_run: bool = False):
    """Ritorna (results_df, null_df). results_df ha una riga per (specifica, modello)."""
    model_types = model_types or sc.MODEL_TYPES
    y_bin = (y == config.POSITIVE_CLASS).astype(int).reset_index(drop=True)
    Xs = {k: v["X"] for k, v in specs.items()}
    for k, X in Xs.items():
        assert len(X) == len(y_bin), f"specifica {k}: numero di pazienti diverso da y"
    families = {(k, mt): v["family"] for k, v in specs.items() for mt in model_types}
    keys = list(itertools.product(Xs.keys(), model_types))

    t0 = time.time()
    obs_vals = Parallel(n_jobs=n_jobs)(
        delayed(_auc)(Xs[name], y_bin, mt, n_folds, seed) for name, mt in keys)
    obs = pd.Series(dict(zip(keys, obs_vals)))
    _estimate(time.time() - t0, n_perm, n_jobs, "label_permutation")
    if dry_run:
        return None, None

    perm_seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_perm)
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_eval_label_perm)(Xs, y_bin.to_numpy(), model_types, n_folds, seed, int(ps))
        for ps in perm_seeds)
    null = pd.DataFrame(res)[keys]
    results = _maxT_table(obs, null, families, "auc_real")
    return results, null


# ---------------------------------------------------------------------------
# TEST 2: valore aggiunto della radiomica ai geni (blocco radiomico permutato)
# ---------------------------------------------------------------------------
def block_permutation_added_value(rad_sets: dict, X_genes: pd.DataFrame, y: pd.Series,
                                  model_types=None, n_perm: int = 100, n_folds: int = 5,
                                  seed: int = config.RANDOM_STATE, n_jobs: int = -1,
                                  dry_run: bool = False):
    """
    Statistica: AUC(geni + radiomica del set) - AUC(solo geni), stesso protocollo di CV.
    Ritorna (results_df, null_df). results_df include auc_genes, auc_both e la differenza.
    """
    model_types = model_types or sc.MODEL_TYPES
    y_bin = (y == config.POSITIVE_CLASS).astype(int).reset_index(drop=True)
    G = X_genes.reset_index(drop=True)
    keys = list(itertools.product(rad_sets.keys(), model_types))

    t0 = time.time()
    auc_genes = {mt: _auc(G, y_bin, mt, n_folds, seed) for mt in model_types}
    both_vals = Parallel(n_jobs=n_jobs)(
        delayed(_auc)(pd.concat([rad_sets[n].reset_index(drop=True), G], axis=1),
                      y_bin, mt, n_folds, seed) for n, mt in keys)
    auc_both = pd.Series(dict(zip(keys, both_vals)))
    obs = pd.Series({k: auc_both[k] - auc_genes[k[1]] for k in keys})
    _estimate(time.time() - t0, n_perm, n_jobs, "block_permutation")
    if dry_run:
        return None, None

    perm_seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_perm)
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_eval_block_perm)(rad_sets, G, y_bin, model_types, n_folds, seed, int(ps))
        for ps in perm_seeds)
    null_both = pd.DataFrame(res)[keys]
    null = null_both.sub(pd.Series({k: auc_genes[k[1]] for k in keys}), axis=1)

    results = _maxT_table(obs, null, {k: "added_value" for k in keys}, "auc_diff_real")
    results["auc_genes"] = [auc_genes[mt] for mt in results["model_type"]]
    results["auc_both"] = [auc_both[(s, m)] for s, m in zip(results["spec"], results["model_type"])]
    return results, null


# ---------------------------------------------------------------------------
# PLOT: curva reale con fascia nulla per specifica
# ---------------------------------------------------------------------------
def plot_label_permutation(results: pd.DataFrame, null: pd.DataFrame, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = list(results["family"].unique())
    fig, axes = plt.subplots(len(fams), 1, figsize=(max(10, 0.22 * len(results) / len(fams) + 6),
                                                  3.6 * len(fams)), squeeze=False)
    colors = {"linear": "#4C72B0", "tree": "#C44E52"}
    for ax, fam in zip(axes[:, 0], fams):
        d = results[results["family"] == fam].sort_values("auc_real").reset_index(drop=True)
        for i, r in d.iterrows():
            col = null[(r["spec"], r["model_type"])]
            ax.vlines(i, col.quantile(.05), col.quantile(.95), color="#BBBBBB", lw=3, zorder=1)
            sig = r["p_adj_family"] < 0.05
            ax.scatter(i, r["auc_real"], marker="D", s=28, zorder=3,
                       facecolor=colors[r["model_type"]] if sig else "white",
                       edgecolor=colors[r["model_type"]])
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_ylabel("AUC pooled OOF")
        ax.set_title(f"famiglia: {fam} — fascia grigia = 5°-95° percentile nullo (etichette permutate); "
                     f"pieno = p corretto (famiglia) < 0.05", fontsize=9)
        ax.set_xticks([])
    axes[-1, 0].set_xlabel("Specifiche ordinate per AUC reale")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
