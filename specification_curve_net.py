"""
Specification curve per la selezione di feature radiomiche guidata dalla rete.
File NUOVO: riusa specification_curve._cv_eval, reduction_v2 e
network_feature_selection senza modificarli.

Assi:
  data_source  : radiomics | both  (+ riferimento "genomics" con sole feature geniche)
  rad_selector : v2_medoid, v2_pca       (riferimenti senza rete, su tutte le feature radiomiche)
                 gene_strength, betweenness, eigenvector, community_rep  (da rete neutral)
                 random_net_nodes        (controllo: k nodi casuali dallo stesso pool della rete)
  network      : separate | unified | "-"  (quale grafo neutral; "-" se non usato)
  k_rad        : budget di feature radiomiche (k3, k5, k8)
  model_type   : linear | tree

Tutte le specifiche usano gli stessi pazienti (load_data(source="both") una
volta sola, poi si separano le colonne) e, con "both", lo stesso blocco
genomico (config_network_selection.GENE_REDUCTION): le differenze dipendono
solo dalla parte radiomica.

Il controllo random_net_nodes restituisce UNA riga per (sorgente, k, modello)
con la MEDIANA di auc_pooled su N_RANDOM_DRAWS estrazioni e il 5°-95° percentile
in auc_pooled_p05 / auc_pooled_p95: un selettore di rete è interessante solo se
sta chiaramente sopra questa fascia.
"""

import numpy as np
import pandas as pd

import config
import config_network_selection as cn
import data_utils
import network_feature_selection as nfs
import reduction_v2
import specification_curve as sc


def _evaluate(X_red, y_bin, model_types, n_folds, n_repeats, seed, n_jobs_model):
    out = {}
    for mt in model_types:
        runs = [sc._cv_eval(X_red, y_bin, mt, n_folds, random_state=seed + r,
                            n_jobs_model=n_jobs_model) for r in range(n_repeats)]
        out[mt] = {
            "auc_mean_fold": float(np.mean([r[0] for r in runs])),
            "auc_sd_fold": float(np.mean([r[1] for r in runs])),
            "auc_pooled": float(np.mean([r[2] for r in runs])),
            "collapsed_model": any(bool(((r[3].abs() > 1e-8).mean(axis=1) == 0).all()) for r in runs),
        }
    return out


def _k_split(k):
    """k radiomiche -> (k_texture, k_shape) per BlockReducer."""
    if not config.EXCLUDE_SHAPE_FROM_MAIN_REDUCTION:
        return k, None
    ks = 1 if k < 8 else 2
    return k - ks, ks


def run_specification_curve_net(model_types=None, n_folds=None, n_repeats=None,
                                n_random_draws=None, random_state=config.RANDOM_STATE,
                                verbose=True, n_jobs_model=-1):
    """Ritorna (spec_df, selected_df, random_draws_df)."""
    model_types = model_types or sc.MODEL_TYPES
    n_folds = n_folds or cn.N_FOLDS
    n_repeats = n_repeats or cn.N_REPEATS
    n_random_draws = n_random_draws or cn.N_RANDOM_DRAWS

    X, y = data_utils.load_data(source="both", print_info=False)
    y_bin = (y == config.POSITIVE_CLASS).astype(int)
    n_min = int(min(y_bin.sum(), len(y_bin) - y_bin.sum()))
    rad_cols = [c for c in X.columns if c.startswith("rad__")]
    gen_cols = [c for c in X.columns if c.startswith("gen__")]

    # blocco genomico fisso (label-free), calcolato una sola volta
    gene_red = reduction_v2.BlockReducer(
        k_texture=None, k_shape=None, gen_prefilter_pct=config.GENE_IQR_TOP_PCT,
        **cn.GENE_REDUCTION).fit(X[gen_cols])
    X_genes = gene_red.transform(X[gen_cols])

    # reti neutral e punteggi dei nodi radiomici
    nets = {}
    for name, path in cn.NETWORK_GRAPHML.items():
        G = nfs.load_network(path)
        scores = nfs.radiomic_node_scores(G)
        missing = [n for n in scores.index if n not in X.columns]
        if missing:
            print(f"[net] rete '{name}': {len(missing)} nodi radiomici non presenti in X "
                  f"(esclusi): {missing[:3]}...")
            scores = scores.drop(index=missing)
        nets[name] = (G, scores)
    pool = list(nets[next(iter(nets))][1].index)   # pool di candidati dei controlli casuali

    rows, selected_rows, draw_rows = [], [], []

    def add_rows(spec, X_rad, X_full, metrics, extra=None):
        for mt, m in metrics.items():
            rows.append({**spec, "model_type": mt, **m, **(extra or {}),
                         "n_features": X_full.shape[1],
                         "n_features_rad": X_rad.shape[1],
                         "n_features_gen": X_full.shape[1] - X_rad.shape[1],
                         "events_per_feature": n_min / max(X_full.shape[1], 1)})

    def run_set(spec, X_rad):
        sources = {"radiomics": X_rad, "both": pd.concat([X_rad, X_genes], axis=1)}
        for src, Xf in sources.items():
            if verbose:
                print(f"[net-curve] {src} | {spec['rad_selector']} | {spec['network']} | "
                      f"{spec['k_rad']} | {Xf.shape[1]} feature")
            m = _evaluate(Xf, y_bin, model_types, n_folds, n_repeats, random_state, n_jobs_model)
            add_rows({"data_source": src, **spec}, X_rad, Xf, m)

    # --- riferimento: sole feature geniche --------------------------------
    m = _evaluate(X_genes, y_bin, model_types, n_folds, n_repeats, random_state, n_jobs_model)
    add_rows({"data_source": "genomics", "rad_selector": "none", "network": "-", "k_rad": "-"},
             X_genes.iloc[:, :0], X_genes, m)

    for k in cn.K_RAD_LEVELS:
        kt, ks = _k_split(k)
        label = f"k{k}"

        # --- riferimenti senza rete ----------------------------------------
        for mode in ("medoid", "pca"):
            red = reduction_v2.BlockReducer(k_texture=kt, k_shape=ks, k_gen=None, mode=mode,
                                            separate_shape=config.EXCLUDE_SHAPE_FROM_MAIN_REDUCTION
                                            ).fit(X[rad_cols])
            run_set({"rad_selector": f"v2_{mode}", "network": "-", "k_rad": label},
                    red.transform(X[rad_cols]))

        # --- selettori di rete -----------------------------------------------
        for net_name, (G, scores) in nets.items():
            for sel in cn.NET_SELECTORS:
                feats = nfs.select_radiomic_features(G, scores, sel, k, cn.REDUNDANCY_MAX_ABS_R)
                for rank, f in enumerate(feats, start=1):
                    selected_rows.append({"network": net_name, "rad_selector": sel, "k_rad": label,
                                          "rank": rank, "feature": f, **scores.loc[f].to_dict()})
                if not feats:
                    print(f"[net] {net_name}/{sel}/{label}: nessun candidato, specifica saltata")
                    continue
                if len(feats) < k:
                    print(f"[net] {net_name}/{sel}/{label}: solo {len(feats)} feature disponibili")
                run_set({"rad_selector": sel, "network": net_name, "k_rad": label}, X[feats])

        # --- controllo casuale: stessi candidati, scelta a caso -------------
        draws = {src: {mt: [] for mt in model_types} for src in ("radiomics", "both")}
        for d in range(n_random_draws):
            rng = np.random.RandomState(random_state + 1000 * k + d)
            feats = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
            for src, Xf in (("radiomics", X[feats]), ("both", pd.concat([X[feats], X_genes], axis=1))):
                m = _evaluate(Xf, y_bin, model_types, n_folds, n_repeats, random_state, n_jobs_model)
                for mt, v in m.items():
                    draws[src][mt].append(v)
                    draw_rows.append({"data_source": src, "k_rad": label, "model_type": mt,
                                      "draw": d, "features": ";".join(feats), **v})
        for src, per_model in draws.items():
            Xr = X[pool[:min(k, len(pool))]]
            Xf = Xr if src == "radiomics" else pd.concat([Xr, X_genes], axis=1)
            for mt, vs in per_model.items():
                auc = np.array([v["auc_pooled"] for v in vs])
                add_rows({"data_source": src, "rad_selector": "random_net_nodes",
                          "network": "-", "k_rad": label}, Xr, Xf, {mt: {
                              "auc_mean_fold": float(np.median([v["auc_mean_fold"] for v in vs])),
                              "auc_sd_fold": float(np.median([v["auc_sd_fold"] for v in vs])),
                              "auc_pooled": float(np.median(auc)),
                              "collapsed_model": False}},
                         extra={"auc_pooled_p05": float(np.quantile(auc, .05)),
                                "auc_pooled_p95": float(np.quantile(auc, .95)),
                                "n_draws": n_random_draws})

    spec_df = pd.DataFrame(rows).sort_values("auc_pooled").reset_index(drop=True)
    return spec_df, pd.DataFrame(selected_rows), pd.DataFrame(draw_rows)
