"""
Il ponte esplicito tra i due studi (ML supervisionato e studio di rete):
rende quantitativo quello che finora era solo un'osservazione qualitativa
("TPI1 è hub sia nel modello ML sia nella rete").

Due analisi:

1) CORRELAZIONE IMPORTANZA-ML <-> CENTRALITÀ DI RETE (ml_importance_vs_centrality)
   Sulla rete "stable" (nodi = feature ML-rilevanti): la centralità di un
   nodo dipende da quanto quella feature era importante nel modello ML?
   Spearman tra centralità di rete (grado pesato, betweenness, eigenvector)
   e le componenti del consenso ML (stability_selection_freq, shap_mean_abs,
   spec_curve_votes, consensus_score aggregato).

2) LA SELEZIONE ML È "CASUALE" RISPETTO ALLA STRUTTURA GENERALE DEL DATASET?
   (stable_vs_neutral_overlap)
   Sulla rete "neutral" (nodi = tutte le feature dopo sola riduzione,
   nessun filtro legato alla label): le feature che il modello ML ha
   giudicato "stabili" hanno grado più alto, nello stesso identico grafo,
   di quelle che ha scartato? Mann-Whitney U (non parametrico, coerente
   con il resto del progetto) — risponde in un solo p-value alla domanda
   "la selezione ML riflette la struttura generale del dataset, o è
   indipendente da essa?". Più il Jaccard tra i set di nodi/archi delle
   due reti, come misura descrittiva di sovrapposizione strutturale.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use("Agg")  # backend non interattivo, per salvare plot da script
import matplotlib.pyplot as plt
import networkx as nx

import config
import radiogenomics as rg


# ---------------------------------------------------------------------------
# UTILITY DI CARICAMENTO
# ---------------------------------------------------------------------------
def _load_consensus(data_source: str = "both") -> pd.DataFrame:
    path = config.OUTPUT_DIR / data_source / "feature_consensus.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non trovato: esegui prima feature_consensus.py con "
            f"DATA_SOURCE='{data_source}'."
        )
    return pd.read_csv(path, index_col=0)


def _load_network_node_stats(feature_set: str, fdr_mode: str = "unified") -> pd.DataFrame:
    path = config.OUTPUT_DIR / "network" / feature_set / fdr_mode / "network_node_stats.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non trovato: esegui prima network_analysis.py "
            f"(feature_set='{feature_set}', fdr_mode='{fdr_mode}')."
        )
    return pd.read_csv(path)


def _load_network_graph(feature_set: str, fdr_mode: str = "unified") -> nx.Graph:
    path = config.OUTPUT_DIR / "network" / feature_set / fdr_mode / "network.graphml"
    if not path.exists():
        raise FileNotFoundError(f"{path} non trovato.")
    return nx.read_graphml(path)


# ---------------------------------------------------------------------------
# 1) CORRELAZIONE IMPORTANZA ML <-> CENTRALITÀ DI RETE
# ---------------------------------------------------------------------------
CENTRALITY_COLS = ["degree_weighted", "betweenness", "eigenvector_centrality"]
IMPORTANCE_COLS = ["consensus_score", "stability_selection_freq", "shap_mean_abs",
                   "spec_curve_votes"]


def ml_importance_vs_centrality(feature_set: str = "stable", fdr_mode: str = "unified",
                                 data_source: str = "both"):
    """
    Correla ogni misura di centralità di rete con ogni misura di importanza
    ML, sulle feature in comune tra la rete scelta e feature_consensus.csv.

    Ritorna (result, merged): result è la tabella lunga (una riga per
    combinazione centralità x importanza, ordinata per |rho| decrescente);
    merged è la tabella wide usata per calcolarla, utile per i plot.
    """
    stats_df = _load_network_node_stats(feature_set, fdr_mode).set_index("feature")
    consensus = _load_consensus(data_source)

    # consensus_score/n_criteria_present sono già attaccati come attributi
    # nodo per la rete "stable" (via network_analysis.build_graph), ma le
    # componenti "grezze" del consenso no: le prendiamo da
    # feature_consensus.csv, che le ha tutte, evitando di duplicare colonne
    merged = stats_df.drop(columns=[c for c in ("consensus_score", "n_criteria_present")
                                     if c in stats_df.columns]).join(consensus, how="inner")

    n_common = len(merged)
    n_network_only = len(stats_df) - n_common
    print(f"[ml_importance_vs_centrality] {n_common} feature in comune tra la rete "
          f"'{feature_set}/{fdr_mode}' ({len(stats_df)} nodi totali) e feature_consensus.csv.")
    if n_network_only > 0:
        print(f"[ml_importance_vs_centrality] ATTENZIONE: {n_network_only} nodi della rete "
              f"non hanno un corrispondente in feature_consensus.csv (feature non valutate "
              f"dal modello ML in quella run) — esclusi dal calcolo.")

    rows = []
    for cent_col in CENTRALITY_COLS:
        if cent_col not in merged.columns:
            continue
        for imp_col in IMPORTANCE_COLS:
            if imp_col not in merged.columns:
                continue
            sub = merged[[cent_col, imp_col]].dropna()
            if len(sub) < 4:
                continue
            rho, p = spearmanr(sub[cent_col], sub[imp_col])
            rows.append({"centrality_metric": cent_col, "importance_metric": imp_col,
                         "rho": rho, "p_value": p, "n": len(sub)})

    result = pd.DataFrame(rows).sort_values("rho", key=lambda s: s.abs(), ascending=False)
    result["q_value"] = rg._benjamini_hochberg(result["p_value"].to_numpy())
    print("\n[ml_importance_vs_centrality] correlazioni centralità di rete <-> importanza ML "
          "(q_value = p corretto per i 12 test):")
    print(result.to_string(index=False))
    n_sig = int((result["q_value"] < 0.05).sum())
    print(f"[ml_importance_vs_centrality] {n_sig}/{len(result)} combinazioni restano "
          f"significative dopo correzione FDR (q<0.05).")

    return result, merged


def plot_importance_vs_centrality(merged: pd.DataFrame, output_path,
                                   centrality_col: str = "degree_weighted",
                                   importance_col: str = "consensus_score",
                                   n_labels: int = 40):
    """
    Scatter + legenda separata: con qualche decina di feature, i nomi
    scritti vicino ai punti si sovrappongono nelle zone dense dello
    scatter (stesso problema già risolto in network_analysis.plot_network).
    Ogni punto è marcato con un numero piccolo; la legenda numero->nome sta
    in un pannello a fianco, mai sovrapposta ai dati. Se le feature sono
    più di n_labels, mostra solo le n_labels più estreme su
    centrality_col (le più informative da identificare singolarmente) e
    lascia gli altri punti senza numero.
    """
    sub = merged[[centrality_col, importance_col]].dropna().copy()
 
    # ordina per centralità: assegna i numeri partendo dai punti più
    # estremi (più interessanti da identificare), il resto resta senza
    # etichetta se n_labels è più piccolo del numero di feature disponibili
    order = sub[centrality_col].sort_values(ascending=False).index
    labeled = list(order[:n_labels])
    rank_of = {feat: i + 1 for i, feat in enumerate(labeled)}
 
    rho, p = spearmanr(sub[centrality_col], sub[importance_col])
 
    fig, (ax_sc, ax_legend) = plt.subplots(
        1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [3, 1.1]}
    )
 
    ax_sc.scatter(sub[importance_col], sub[centrality_col], color="#4C72B0",
                  alpha=0.85, s=60, zorder=2)
    for feat in labeled:
        row = sub.loc[feat]
        ax_sc.annotate(str(rank_of[feat]), (row[importance_col], row[centrality_col]),
                        fontsize=8, fontweight="bold", color="white",
                        ha="center", va="center", zorder=3)
    ax_sc.set_xlabel(importance_col)
    ax_sc.set_ylabel(centrality_col)
    ax_sc.set_title(f"Importanza ML vs centralità di rete\n"
                     f"Spearman rho={rho:.3f}, p={p:.4f}, n={len(sub)}")
 
    ax_legend.axis("off")
    legend_lines = [f"{rank_of[feat]:>2}.  {feat.split('__', 1)[-1]}" for feat in labeled]
    ax_legend.text(0, 1, "\n".join(legend_lines), fontsize=8.5, va="top", ha="left",
                    family="monospace", transform=ax_legend.transAxes)
    n_unlabeled = len(sub) - len(labeled)
    title = f"Top {len(labeled)} per\n{centrality_col}"
    if n_unlabeled > 0:
        title += f"\n(+{n_unlabeled} punti senza numero)"
    ax_legend.set_title(title, fontsize=9.5, loc="left")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_importance_vs_centrality] salvato in {output_path}")


# ---------------------------------------------------------------------------
# 2) STABLE vs NEUTRAL: la selezione ML è casuale rispetto alla rete generale?
# ---------------------------------------------------------------------------
def stable_vs_neutral_overlap(fdr_mode: str = "unified", data_source: str = "both"):
    """
    Confronta la rete "stable" (nodi = feature ML-rilevanti) con la rete
    "neutral" (nodi = tutte le feature dopo sola riduzione, nessun filtro
    legato alla label), entrambe con la stessa fdr_mode.

    Ritorna (summary, neutral_stats_flagged): summary è un dizionario con
    Jaccard nodi/archi e il risultato del Mann-Whitney; neutral_stats_flagged
    è la tabella dei nodi della rete neutral con una colonna booleana
    'is_ml_stable', utile per il plot.
    """
    G_stable = _load_network_graph("stable", fdr_mode)
    G_neutral = _load_network_graph("neutral", fdr_mode)

    nodes_stable, nodes_neutral = set(G_stable.nodes()), set(G_neutral.nodes())
    jaccard_nodes = (len(nodes_stable & nodes_neutral) / len(nodes_stable | nodes_neutral)
                     if (nodes_stable | nodes_neutral) else np.nan)
        # sta facendo intersezione (&) diviso unione (/)

    common_nodes = nodes_stable & nodes_neutral
    edges_stable = {tuple(sorted(e)) for e in G_stable.edges()
                     if e[0] in common_nodes and e[1] in common_nodes}
    edges_neutral = {tuple(sorted(e)) for e in G_neutral.edges()
                      if e[0] in common_nodes and e[1] in common_nodes}
    union_edges = edges_stable | edges_neutral
    jaccard_edges = len(edges_stable & edges_neutral) / len(union_edges) if union_edges else np.nan
        # jaccard rappresenta "elementi condivisi" / "elementi presenti complessivamente almeno in uno dei due insiemi".

    print(f"[stable_vs_neutral_overlap] nodi: stable={len(nodes_stable)}, "
          f"neutral={len(nodes_neutral)}, comuni={len(common_nodes)} | "
          f"Jaccard nodi={jaccard_nodes:.3f}")
    if union_edges:
        print(f"[stable_vs_neutral_overlap] archi (ristretti ai nodi comuni): "
              f"stable={len(edges_stable)}, neutral={len(edges_neutral)}, "
              f"Jaccard archi={jaccard_edges:.3f}")
    else:
        print("[stable_vs_neutral_overlap] nessun arco tra nodi comuni in nessuna delle due reti.")

    # confronto di centralità dentro la rete neutral: le feature "stabili"
    # per il ML hanno grado più alto delle altre, nello stesso grafo?
    consensus = _load_consensus(data_source)
    neutral_stats = _load_network_node_stats("neutral", fdr_mode).set_index("feature")
    neutral_stats = neutral_stats.drop(
        columns=[c for c in ("consensus_score", "n_criteria_present") if c in neutral_stats.columns]
    ).join(consensus[["n_criteria_present"]], how="left")
    neutral_stats["n_criteria_present"] = neutral_stats["n_criteria_present"].fillna(0)
    neutral_stats["is_ml_stable"] = neutral_stats["n_criteria_present"] >= config.RADIOGENOMICS_MIN_CRITERIA

    deg_stable = neutral_stats.loc[neutral_stats["is_ml_stable"], "degree_weighted"] 
        # La sintassi di base è df.loc[righe, colonne]. Scegli solo le righe con True come riga, ne prendi il valore colonna.
    deg_other = neutral_stats.loc[~neutral_stats["is_ml_stable"], "degree_weighted"]

    mw_result = None
    if len(deg_stable) >= 3 and len(deg_other) >= 3:
        u_stat, p_value = mannwhitneyu(deg_stable, deg_other, alternative="two-sided")
            # The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution underlying sample x 
            # is the same as the distribution underlying sample y. It is often used as a test of difference in location 
            # between distributions. Param alternative{‘two-sided’, ‘less’, ‘greater’}, defines the alternative hypothesis.
            # Default is ‘two-sided’. Let SX(u) and SY(u) be the survival functions of the distributions underlying x and y,
            # respectively. ‘two-sided’: the distributions are not equal
            # i valori di un gruppo tendono a essere sistematicamente più alti o più bassi dei valori dell'altro?
            # È un test non parametrico per confrontare due gruppi indipendenti.
            # H0 (hp di base): degree stable e degree other provengono dalla stessa distribuzione; contro: H1: le 
            # due distribuzioni differiscono. Li ordina e guarda i rank, il test quantifica quanto sarebbe improbabile 
            # ottenere una separazione dei rank così forte se i due gruppi fossero realmente uguali.
            # restituisce due cose: u_stat, è la statistica U del test, è principalmente un valore intermedio utilizzato 
            # per costruire l'inferenza. p_value, è la parte inferenziale.
            # La U può essere interpretata come il numero di confronti a coppie in cui un gruppo tende a precedere l'altro. In
            # generale U va da 0 a n_a * n_b, dove n_a/n_b sono il numero di osservazioni nel gruppo A/B. Se tutti i confronti
            # sono vinti da B allora sarà prossimo a 0, e viceversa.
            # U descrive quanto i rank dei due gruppi sono separati.
            # p-value dice che una separazione almeno così forte sarebbe poco compatibile con H0.
            # mediane ti dicono la direzione e una misura semplice della posizione dei due gruppi.

        mw_result = {"u_statistic": float(u_stat), "p_value": float(p_value),
                     "median_degree_stable": float(deg_stable.median()),
                     "median_degree_other": float(deg_other.median()),
                     "n_stable": int(len(deg_stable)), "n_other": int(len(deg_other))}
        print(f"\n[stable_vs_neutral_overlap] grado pesato nella rete NEUTRAL: "
              f"feature ML-stabili (n={len(deg_stable)}) mediana={deg_stable.median():.2f} | "
              f"feature ML-scartate (n={len(deg_other)}) mediana={deg_other.median():.2f}")
        print(f"[stable_vs_neutral_overlap] Mann-Whitney U={u_stat:.1f}, p={p_value:.4f}")
        if p_value < 0.05:
            direction = "più" if deg_stable.median() > deg_other.median() else "meno"
            print(f"[stable_vs_neutral_overlap] Le feature giudicate stabili dal modello ML "
                  f"sono significativamente {direction} centrali nella rete generale: "
                  f"la selezione ML non è indipendente dalla struttura di rete.")
        else:
            print("[stable_vs_neutral_overlap] Nessuna differenza significativa: sulla base di "
                  "questo campione, la centralità nella rete generale non distingue le feature "
                  "giudicate stabili dal ML dalle altre.")
    else:
        print("\n[stable_vs_neutral_overlap] ATTENZIONE: troppo pochi nodi in uno dei due "
              "gruppi (stabili/non stabili) nella rete neutral per un test attendibile "
              f"(n_stabili={len(deg_stable)}, n_altri={len(deg_other)}).")

    summary = {
        "jaccard_nodes": jaccard_nodes, "jaccard_edges": jaccard_edges,
        "n_nodes_stable": len(nodes_stable), "n_nodes_neutral": len(nodes_neutral),
        "n_common_nodes": len(common_nodes), "mannwhitney": mw_result,
    }
    return summary, neutral_stats


def plot_stable_vs_neutral_degree(neutral_stats_flagged: pd.DataFrame, output_path):
    """
    Strip plot del grado pesato nella rete neutral, separato per feature
    ML-stabili vs ML-scartate — la versione visiva del test Mann-Whitney.
    """
    groups = [
        ("ML-stabili", neutral_stats_flagged.loc[neutral_stats_flagged["is_ml_stable"], "degree_weighted"]),
        ("ML-scartate", neutral_stats_flagged.loc[~neutral_stats_flagged["is_ml_stable"], "degree_weighted"]),
    ]
    if any(len(g) == 0 for _, g in groups):
        print("[plot_stable_vs_neutral_degree] un gruppo è vuoto: nessun plot generato.")
        return

    plt.figure(figsize=(6, 6))
    positions = [0, 1]
    for pos, (label, values) in zip(positions, groups):
        jitter = np.random.RandomState(config.RANDOM_STATE).uniform(-0.08, 0.08, size=len(values))
        plt.scatter(np.full(len(values), pos) + jitter, values, alpha=0.7, s=40,
                    color="#C44E52" if label == "ML-stabili" else "#4C72B0")
        plt.scatter([pos], [values.median()], marker="_", s=400, color="black", linewidths=2)

    plt.xticks(positions, [f"{label}\n(n={len(g)})" for label, g in groups])
    plt.ylabel("Grado pesato nella rete 'neutral'")
    plt.title("Centralità nella rete generale: feature ML-stabili vs scartate\n"
              "(barra nera = mediana)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_stable_vs_neutral_degree] salvato in {output_path}")


if __name__ == "__main__":
    out_dir = config.OUTPUT_DIR / "network" / "ml_bridge"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("1) IMPORTANZA ML <-> CENTRALITÀ DI RETE")
    print("=" * 70)
    for feature_set in ("stable", "neutral"): 
        for fdr_mode in ("unified", "separate"):
            print(f"\nfeature_set='{feature_set}', fdr_mode='{fdr_mode}'")
            corr_table, merged = ml_importance_vs_centrality(feature_set=feature_set, fdr_mode=fdr_mode)
            corr_table.to_csv(out_dir / f"ml_importance_vs_centrality_{feature_set}_{fdr_mode}.csv",
                              index=False)
            merged.to_csv(out_dir / f"ml_importance_vs_centrality_merged_table_{feature_set}_{fdr_mode}.csv")
            plot_importance_vs_centrality(merged, out_dir / f"importance_vs_degree_{feature_set}_{fdr_mode}.png")

    print("\n" + "=" * 70)
    print("2) RETE 'stable' vs RETE 'neutral'")
    print("=" * 70)
    summary, neutral_stats_flagged = stable_vs_neutral_overlap()
    neutral_stats_flagged.to_csv(out_dir / "neutral_network_stats_with_ml_flag.csv")
    plot_stable_vs_neutral_degree(neutral_stats_flagged, out_dir / "stable_vs_neutral_degree.png")

    with open(out_dir / "stable_vs_neutral_summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"\nTutti i risultati sono stati salvati in: {out_dir}")