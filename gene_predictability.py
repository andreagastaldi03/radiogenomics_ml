"""
Prevedibilità dei geni dalla radiomica, CONDIZIONATA al fenotipo.

Domanda: le feature radiomiche contengono informazione sull'espressione genica
oltre a quella già spiegata dal fenotipo (ADK/SCC)? Se radiomica e geni si
muovono insieme solo perché entrambi cambiano tra ADK e SCC, il legame è
mediato dal fenotipo; se resta dopo averlo tolto, è un legame intrinseco.

Bersagli (tutti label-free, trasformati con rank-inverse-normal per robustezza)
  gene_PC       : le componenti geniche già usate (BlockReducer, k_gen/mode di
                  config_network_selection.GENE_REDUCTION)
  gene_module   : primo componente principale di ogni community gene-gene
                  (Louvain) della rete neutral unified, con >= 3 geni
  exploratory   : EXPLORATORY_GENES (TPI1, PKM, ...): scelti DOPO aver visto
                  le reti, quindi esplorativi

Predittori radiomici (la stessa matrice viene permutata per tutti)
  A_ridge_pool     : ridge su tutte le radiomiche dei nodi della rete neutral
                     (riduzione label-free, nessuna selezione per legame con i geni)
  B_infold_topK    : stesso pool, ma le K feature più correlate al bersaglio
                     scelte SOLO sul training di ogni fold
  C_v2_pca_kK / C_v2_medoid_kK : BlockReducer sulle 107 radiomiche
  V_log_volume     : solo log(volume tumorale), riferimento/confondente
NON si usano i selettori di rete basati sul legame radiomica-gene
(gene_strength, community_rep): sarebbero circolari rispetto a questo test.

Statistica (fuori campione, CV stratificata per classe, ripetuta)
  dR2_cond : 1 - SSE(covariate + radiomica) / SSE(covariate), dove le
             covariate sono la classe (+ log volume con covariates="class_volume").
             Il modello ridge è addestrato sul RESIDUO dopo le covariate.
             0 = nessun guadagno, <0 = peggio della sola classe.
  dR2_unc  : 1 - SSE(radiomica) / SSE(media), senza condizionare al fenotipo
             (include il legame mediato dal fenotipo).
  r2_covariates_only : quanto del bersaglio spiegano già le sole covariate.

Null
  cond : righe radiomiche permutate DENTRO ciascuna classe (conserva
         l'associazione radiomica-fenotipo, rompe quella radiomica-geni a parità
         di fenotipo). Stessa permutazione per tutti i predittori e bersagli.
  unc  : permutazione globale delle righe radiomiche.
Correzione per test multipli: maxT di Westfall-Young (permutation_specs._maxT_table)
con famiglia = predittore (corregge per il numero di bersagli) e globale.
Folds fissi (dipendono solo dalla classe): identici per osservato e permutazioni.
"""

import time

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import config
import config_network_selection as cn
import network_feature_selection as nfs
import permutation_specs as ps
import reduction_v2

EXPLORATORY_GENES = ["TPI1", "PKM", "ALDOA", "NFKBIA"]
RIDGE_ALPHAS = np.logspace(-1, 4, 11)
VOLUME_CANDIDATES = ["rad__original_shape_MeshVolume", "rad__original_shape_VoxelVolume"]


# ---------------------------------------------------------------------------
# COSTRUZIONE BERSAGLI E PREDITTORI (label-free)
# ---------------------------------------------------------------------------
def _rank_int(v):
    return norm.ppf((rankdata(v) - 0.5) / len(v))
# calcola solo la "classifica" (il rango) di ciascun valore - rankdata
# converte questa classifica in una probabilità cumulativa/ valore percentuale
# funzione Percent Point Function (l'inversa della distribuzione normale). 
# prende quelle percentuali e le mappa sulle coordinate di una curva a campana perfetta. 
# il paziente che era esattamente a metà classifica riceverà il valore 0, ecc


def build_targets(X: pd.DataFrame, net_path=None):
    """Ritorna (T DataFrame pazienti x bersagli, info DataFrame)."""
    gen = [c for c in X.columns if c.startswith("gen__")]
    cols, info = {}, []

    red = reduction_v2.BlockReducer(k_texture=None, k_shape=None,
                                    gen_prefilter_pct=config.GENE_IQR_TOP_PCT,
                                    **cn.GENE_REDUCTION).fit(X[gen])
    Xg = red.transform(X[gen])
    for _, r in red.report_.iterrows():
        cols[r["feature_out"]] = Xg[r["feature_out"]].to_numpy()
        info.append({"target": r["feature_out"], "family": "gene_PC", "n_genes": r["n_members"],
                     "members": r["members"]})

    G = nfs.load_network(net_path or cn.NETWORK_GRAPHML["unified"])
    genes = [n for n in G if n.startswith("gen__") and n in X.columns]
    comms = nfs.stable_louvain(G.subgraph(genes), seed=config.RANDOM_STATE)
    for i, comm in enumerate(comms, start=1):
        if len(comm) < 3:
            continue
        members = sorted(comm)
        M = X[members].to_numpy(float)
        sd = M.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (M - M.mean(axis=0)) / sd
        pc = Z @ np.linalg.svd(Z, full_matrices=False)[2][0]
        if np.sum(Z.T @ pc) < 0:      # segno: media dei loading positiva
            pc = -pc
        name = f"module{i}_n{len(members)}"
        cols[name] = pc
        info.append({"target": name, "family": "gene_module", "n_genes": len(members),
                     "members": ";".join(m.replace("gen__", "") for m in members)})

    for g in EXPLORATORY_GENES:
        c = f"gen__{g}"
        if c in X.columns:
            cols[g] = X[c].to_numpy(float)
            info.append({"target": g, "family": "exploratory", "n_genes": 1, "members": g})

    T = pd.DataFrame({k: _rank_int(v) for k, v in cols.items()}, index=X.index)
    # _rank_int rimappa ciascuna colonna sulla stessa scala standard (media 0 e varianza 1 
    # distribuita normalmente), rendendo i bersagli direttamente comparabili tra loro
    return T, pd.DataFrame(info)


def log_volume(X: pd.DataFrame):
    for c in VOLUME_CANDIDATES:
        if c in X.columns and (X[c] > 0).all():
            return np.log(X[c].to_numpy(float)), c
    return None, None


def build_predictors(X: pd.DataFrame, k_topk: int = 5, k_v2: int = 5):
    """
    Ritorna preds: {nome: (matrice n x p, tipo, k)}; tipo 'ridge' o 'topk'.
    """
    rad = [c for c in X.columns if c.startswith("rad__")]
    G = nfs.load_network(cn.NETWORK_GRAPHML["unified"])
    pool = [n for n in G if n.startswith("rad__") and n in X.columns]
    Mpool = X[pool].to_numpy(float)
    preds = {"A_ridge_pool": (Mpool, "ridge", None),
             f"B_infold_top{k_topk}": (Mpool, "topk", k_topk)}
    kt, ks = ps.sc2_k_split(k_v2)
    for mode in ("pca", "medoid"):
        red = reduction_v2.BlockReducer(k_texture=kt, k_shape=ks, k_gen=None, mode=mode,
                                        separate_shape=config.EXCLUDE_SHAPE_FROM_MAIN_REDUCTION
                                        ).fit(X[rad])
        preds[f"C_v2_{mode}_k{k_v2}"] = (red.transform(X[rad]).to_numpy(float), "ridge", None)
    vol, vname = log_volume(X)
    if vol is not None:
        preds["V_log_volume"] = (vol[:, None], "ridge", None)
    else:
        print("[gene_predictability] volume tumorale non trovato: predittore V e covariata esclusi")
    return preds, vname


# ---------------------------------------------------------------------------
# VALUTAZIONE
# ---------------------------------------------------------------------------
def make_folds(cls, n_repeats, seed, n_splits=5):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros(len(cls)), cls))


def _cv_stats(preds, Tv, Zd, folds, idx, conditional):
    """Ritorna (sse_base[m], {pred: sse_model[m]}) sommando su tutti i fold."""
    m = Tv.shape[1] # numero di bersagli genici da prevedere (colonne di Tv)
    sse_base = np.zeros(m) # somma dei quadrati degli errori commessi dal modello di riferimento
    # su tutti i pazienti di test
    sse_model = {p: np.zeros(m) for p in preds} # accumula gli errori per ciascun predittore radiomico
    for tr, te in folds: # indici dei pazienti del set di training (tr) e al test (te) per il fold corrente
        if conditional: # minimi quadrati ordinari (np.linalg.lstsq)
            beta = np.linalg.lstsq(Zd[tr], Tv[tr], rcond=None)[0]
            base_tr, base_te = Zd[tr] @ beta, Zd[te] @ beta
        else: # modello base predice semplicemente la media mu dei geni calcolata sul training fold
            mu = Tv[tr].mean(axis=0)
            base_tr, base_te = np.tile(mu, (len(tr), 1)), np.tile(mu, (len(te), 1))
        res_tr = Tv[tr] - base_tr
        # sottrae la predizione base dal valore reale dei geni del training set. 
        # è ciò che resta ancora da spiegare dopo aver già tenuto conto del fenotipo/volume. 
        sse_base += ((Tv[te] - base_te) ** 2).sum(axis=0)
        for name, (M, kind, k) in preds.items():
            Mp = M if idx is None else M[idx]
            sc = StandardScaler().fit(Mp[tr])
            Xtr, Xte = sc.transform(Mp[tr]), sc.transform(Mp[te])
            if kind == "ridge":
                pr = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(Xtr, res_tr).predict(Xte)
            else:  # selezione dei top-k SOLO sul training del fold
                pr = np.empty((len(te), m))
                for j in range(m):
                    score = np.abs(Xtr.T @ (res_tr[:, j] - res_tr[:, j].mean()))
                    # calcola la covarianza/correlazione tra ciascuna feature radiomica in Xtr e il residuo genico
                    sel = np.argsort(-score)[:k]
                    # prime k feature con correlazione assoluta più alta
                    pr[:, j] = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xtr[:, sel], res_tr[:, j]).predict(Xte[:, sel])
            sse_model[name] += ((Tv[te] - base_te - pr) ** 2).sum(axis=0)
    return sse_base, sse_model
# stima complessiva per il paziente di test è data da: predizione base + predizione radiomica sui residui (base_te + pr).   
# La differenza rispetto al bersaglio reale Tv[te] viene elevata al quadrato e sommata su tutti i soggetti di test fold 
# dopo fold. Alla fine, sse_base e sse_model conterranno le metriche out-of-fold utilizzate nella funzione successiva 
# per ricavare direttamente: Delta R^2 = 1 - SSE-modello radiomico / SSE-modello base


def _stat_dict(preds, tnames, Tv, Zd, folds, idx, conditional):
    sse_base, sse_model = _cv_stats(preds, Tv, Zd, folds, idx, conditional)
    return {(t, p): 1.0 - sse_model[p][j] / sse_base[j]
            for p in preds for j, t in enumerate(tnames)}, sse_base


def _one_perm(preds, tnames, Tv, Zd, folds, cls, seed):
    rng = np.random.RandomState(seed)
    idx_in = np.arange(len(cls))
    # Crea un array di indici da 0 a N-1 (dove N è il numero dei pazienti)
    for c in np.unique(cls):                      # permutazione DENTRO ciascuna classe
    # Itera sulle classi del fenotipo (es. 0 per ADK e 1 per SCC)
        members = np.where(cls == c)[0]
        idx_in[members] = rng.permutation(members)
        # Rimescola casualmente gli indici dei pazienti, ma soltanto all'interno dello stesso fenotipo
        # spezza il legame tra la radiomica e i geni dello specifico paziente a parità di fenotipo
    idx_gl = rng.permutation(len(cls))             # permutazione globale
    cond, _ = _stat_dict(preds, tnames, Tv, Zd, folds, idx_in, True)
    # cond (ipotesi nulla condizionata), passa la permutazione vincolata alla classe
    # cosa ci si può aspettare se la radiomica non avesse alcun legame intrinseco con 
    # i geni oltre a quello mediato dal fenotipo
    unc, _ = _stat_dict(preds, tnames, Tv, np.ones((len(cls), 1)), folds, idx_gl, False)
    return cond, unc


def run_gene_predictability(X: pd.DataFrame, y: pd.Series, covariates: str = "class",
                            n_perm: int = 2000, n_repeats: int = 3, k_topk: int = 5,
                            seed: int = config.RANDOM_STATE, n_jobs: int = -1,
                            dry_run: bool = False):
    """
    covariates : "class" (default) oppure "class_volume" (classe + log volume tumorale).
    Ritorna (results_df, null_cond_df, null_unc_df, targets_info_df).
    """
    cls = (y == config.POSITIVE_CLASS).astype(int).to_numpy()
    T, tinfo = build_targets(X)
    preds, vname = build_predictors(X, k_topk=k_topk)
    tnames, Tv = list(T.columns), T.to_numpy(float)

    Z = [np.ones(len(cls)), cls.astype(float)]
    if covariates == "class_volume":
        vol, _ = log_volume(X)
        if vol is None:
            raise SystemExit("covariates='class_volume' ma il volume tumorale non è disponibile")
        Z.append((vol - vol.mean()) / vol.std())
    elif covariates != "class":
        raise ValueError("covariates deve essere 'class' o 'class_volume'")
    Zd = np.column_stack(Z)

    folds = make_folds(cls, n_repeats, seed)
    t0 = time.time()
    obs_cond, sse_cov = _stat_dict(preds, tnames, Tv, Zd, folds, None, True)
    obs_unc, sse_mean = _stat_dict(preds, tnames, Tv, np.ones((len(cls), 1)), folds, None, False)
    t_obs = time.time() - t0
    print(f"[gene_predictability] {len(tnames)} bersagli x {len(preds)} predittori | "
          f"covariate: {covariates} (volume: {vname}) | passaggio osservato: {t_obs:.1f}s | "
          f"stima {n_perm} permutazioni: ~{t_obs * n_perm / 60:.0f} min su un core "
          f"(divisi per il numero di processi)")
    if dry_run:
        return None, None, None, tinfo

    seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_perm)
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_perm)(preds, tnames, Tv, Zd, folds, cls, int(s)) for s in seeds)
    keys = list(obs_cond.keys())
    null_cond = pd.DataFrame([o[0] for o in out])[keys]
    null_unc = pd.DataFrame([o[1] for o in out])[keys]

    fam = {k: k[1] for k in keys}
    cond = ps._maxT_table(pd.Series(obs_cond), null_cond, fam, "dR2_cond")
    unc = ps._maxT_table(pd.Series(obs_unc), null_unc, fam, "dR2_unc")

    res = cond.rename(columns={"spec": "target", "model_type": "predictor",
                               "null_p95": "dR2_cond_null_p95", "p_raw": "p_raw_cond",
                               "p_adj_family": "p_adj_cond_within_predictor",
                               "p_adj_global": "p_adj_cond_global"}).drop(columns=["family", "null_mean", "z_obs"])
    u = unc.rename(columns={"spec": "target", "model_type": "predictor", "p_raw": "p_raw_unc",
                            "p_adj_family": "p_adj_unc_within_predictor"})[
        ["target", "predictor", "dR2_unc", "p_raw_unc", "p_adj_unc_within_predictor"]]
    res = res.merge(u, on=["target", "predictor"])
    r2_cov = {t: 1.0 - sse_cov[j] / sse_mean[j] for j, t in enumerate(tnames)}
    res["r2_covariates_only"] = res["target"].map(r2_cov)
    res = res.merge(tinfo[["target", "family", "n_genes"]], on="target")
    res = res.sort_values("dR2_cond", ascending=False).reset_index(drop=True)
    return res, null_cond, null_unc, tinfo


def plot_gene_predictability(res: pd.DataFrame, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    preds = list(res["predictor"].unique())
    fig, axes = plt.subplots(len(preds), 1, figsize=(10, 2.6 * len(preds)), squeeze=False, sharex=True)
    order = res.groupby("target")["dR2_cond"].max().sort_values(ascending=False).index.tolist()
    for ax, p in zip(axes[:, 0], preds):
        d = res[res["predictor"] == p].set_index("target").reindex(order)
        x = np.arange(len(d))
        sig = d["p_adj_cond_within_predictor"] < 0.05
        ax.bar(x, d["dR2_cond"], color=np.where(sig, "#C44E52", "#8C8C8C"))
        ax.scatter(x, d["dR2_cond_null_p95"], marker="_", s=250, color="black", zorder=3)
        ax.axhline(0, color="black", lw=.8)
        ax.set_ylabel("dR2 cond.", fontsize=8)
        ax.set_title(f"{p}  (rosso: p corretto < 0.05; trattino nero: 95° percentile nullo)", fontsize=9)
    axes[-1, 0].set_xticks(np.arange(len(order)))
    axes[-1, 0].set_xticklabels(order, rotation=60, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
