"""
Specification curve per lo studio di rete (feature_set="neutral" di
network_analysis.py): il grafo regge al variare delle scelte di riduzione
neutra delle feature (metodo di selezione geni, soglia di ridondanza,
trattamento separato delle feature di forma), o è un artefatto di una
combinazione particolare?

Stessa domanda di specification_curve.py sul lato ML, ma qui non c'è una
metrica di performance condivisa (niente label, niente AUC): bisogna
scegliere una metrica diversa, e la scelta non è ovvia.

Perchè non usare conteggi grezzi (n_nodi, n_archi, densità) come metrica
------------------------------------------------------------------------
Una soglia di ridondanza più permissiva (es. 0.95 invece di 0.90) tiene
semplicemente più feature: quasi meccanicamente, più coppie testate e
spesso più archi sopravvivono a FDR — non perché la struttura di
correlazione sia "migliore", ma perché il metodo ha più materiale grezzo
su cui lavorare. Confrontare conteggi grezzi tra specifiche con un numero
di nodi diverso è come confrontare l'AUC di due modelli allenati su task
diversi: il numero non è comparabile.

La metrica usata qui: modularity z-score rispetto a un modello nullo
---------------------------------------------------------------------
network_diagnostics.null_model_comparison confronta già la modularità
osservata con quella di grafi Erdős–Rényi casuali a parità di nodi e
archi. Il risultato (uno z-score, o il p-value empirico) è già relativo
alla dimensione del grafo di quella specifica: gioca lo stesso ruolo che
l'AUC giocava nella spec curve ML — una misura di "segnale oltre il
rumore", comparabile tra specifiche di dimensione diversa.

Metrica secondaria monitorata (non "più alta è meglio", ma da tracciare):
domain_assortativity — la conclusione "radiomica e genomica formano
community miste" regge in tutte le specifiche o dipende da una scelta
particolare?

Sottoprodotto (non è "la curva", ma nello spirito di feature_consensus.py):
per ogni nodo/arco, in quante specifiche compare nel grafo finale — un
conteggio di robustezza analogo ai "voti" della spec curve ML.

Limite ereditato: la riduzione per ridondanza fa clustering e sceglie 
un rappresentante per gruppo di feature correlate;
quel rappresentante può cambiare nome da una soglia all'altra. Il
conteggio "quante specifiche confermano questo nodo" eredita quindi la
stessa imprecisione già presente in modo implicito in
feature_votes_across_specs lato ML.

Costo: a differenza della spec curve ML, qui non c'è nessun fit di
modello — solo correlazioni e metriche di grafo. L'intera griglia (12
combinazioni di default) gira in pochi minuti anche in sequenziale;
niente REDUCED_SPEC_GRID, niente parallelizzazione necessaria.
"""

import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data_utils
import network_analysis as na
import network_diagnostics as nd


# ---------------------------------------------------------------------------
# GRIGLIA DI SPECIFICHE — stesse 3 dimensioni già usate in specification_curve.py
# lato ML (gene_selection_method, exclude_shape, redundancy_corr_threshold),
# per restare confrontabile con quella. variance_threshold non è qui incluso
# per lo stesso motivo per cui non lo è in SPEC_GRID lato ML: neutral_feature_
# reduction non lo espone come parametro di override (resta fisso su
# config.VARIANCE_THRESHOLD). Aggiungibile in futuro sovrascrivendo
# temporaneamente config.VARIANCE_THRESHOLD, se servisse.
# ---------------------------------------------------------------------------
NETWORK_SPEC_GRID = {
    "gene_selection_method": ["variance", "iqr_top_pct", "iqr_top_n"],
    "redundancy_corr_threshold": [0.90, 0.95],
    "exclude_shape": [True, False],
}


# Nodo di cui tracciare la posizione in classifica attraverso le specifiche.
# Estendibile a una lista se in futuro interessa tracciarne più di uno.
TRACKED_NODE = "gen__TPI1"


# ---------------------------------------------------------------------------
# Rank percentile di un nodo su una colonna di statistiche di rete
# ---------------------------------------------------------------------------
def _percentile_rank(stats_df: pd.DataFrame, node: str, column: str) -> float:
    """
    1 - (rank-1)/(n_nodi-1): 1.0 = nodo più centrale della rete, valori
    vicini a 0 = nodo periferico. NaN se il nodo non è presente (filtrato
    dalla riduzione feature di questa specifica, o isolato/rimosso dal
    grafo). Comparabile tra reti di dimensione diversa, a differenza del
    rank grezzo.
    """
    if node not in stats_df.index or column not in stats_df.columns:
        return np.nan
    n = len(stats_df)
    if n <= 1:
        return np.nan
    ranks = stats_df[column].rank(method="min", ascending=False)  # 1 = valore più alto
    return 1 - (ranks.loc[node] - 1) / (n - 1)


# ---------------------------------------------------------------------------
# UNA SINGOLA SPECIFICA
# ---------------------------------------------------------------------------
def _run_one_spec(X_raw: pd.DataFrame, spec: dict, data_source: str,
                   fdr_mode: str, fdr_alpha: float, method: str,
                   n_null: int, n_assort_perm: int, 
                   random_state: int, print_info: bool = False):
    """
    Riduce le feature con questa combinazione di parametri, costruisce il
    grafo (rad-rad, gen-gen, rad-gen con correzione FDR), e calcola le
    metriche di specifica. Ritorna (row_dict, G) — G serve al chiamante per
    accumulare i conteggi di consenso su nodi/archi.
    """
    X_reduced = data_utils.neutral_feature_reduction(
        X_raw,
        gene_selection_method=spec.get("gene_selection_method"),
        exclude_shape=spec.get("exclude_shape"),
        redundancy_corr_threshold=spec.get("redundancy_corr_threshold"),
        print_info=print_info,
    )
    rad_df = X_reduced[[c for c in X_reduced.columns if c.startswith("rad__")]]
    gene_df = X_reduced[[c for c in X_reduced.columns if c.startswith("gen__")]]

    row = {**spec, "n_rad_features": rad_df.shape[1], "n_gene_features": gene_df.shape[1]}
    empty_metrics = {
        "n_nodes": np.nan, "n_edges": np.nan, "density": np.nan,
        "modularity_observed": np.nan, "modularity_null_mean": np.nan,
        "modularity_null_std": np.nan, "modularity_z": np.nan,
        "modularity_p_value": np.nan,
        "domain_assortativity": np.nan, "assortativity_z": np.nan,
        "assortativity_p_value": np.nan,
        "tpi1_degree_percentile": np.nan, "tpi1_betweenness_percentile": np.nan,
    }

    if rad_df.shape[1] < 2 or gene_df.shape[1] < 2:
        print(f"[_run_one_spec] {spec}: troppo poche feature sopravvissute "
              f"({rad_df.shape[1]} rad, {gene_df.shape[1]} gen) per costruire una rete "
              f"significativa — riga segnata come NaN.")
        row.update(empty_metrics)
        return row, None

    edge_long_df = na.build_edge_list(rad_df, gene_df, method=method, fdr_mode=fdr_mode,
                                       fdr_alpha=fdr_alpha, print_info=print_info)
    G = na.build_graph(edge_long_df, fdr_alpha=fdr_alpha)

    if G.number_of_edges() == 0:
        row.update({**empty_metrics, "n_nodes": 0, "n_edges": 0})
        return row, G

    # --- modularità vs modello nullo Erdos-Rényi ---
    obs_mod, null_mod, p_value = nd.null_model_comparison(G, n_null=n_null,
                                                           random_state=random_state)
    z = ((obs_mod - null_mod.mean()) / null_mod.std()) if null_mod.std() > 0 else np.nan

    # --- assortatività vs modello nullo per permutazione di dominio ---
    obs_assort, _, assort_p, assort_z = nd.domain_assortativity_permutation_test(
        G, n_permutations=n_assort_perm, random_state=random_state
    )

    # --- rank percentile di TPI1 o del nodo tracciato ---
    tpi1_degree_pct, tpi1_betw_pct = np.nan, np.nan
    try:
        stats_df = na.compute_network_stats(G).set_index("feature")
        tpi1_degree_pct = _percentile_rank(stats_df, TRACKED_NODE, "degree_weighted")
        tpi1_betw_pct = _percentile_rank(stats_df, TRACKED_NODE, "betweenness")
    except Exception as e:
        print(f"[_run_one_spec] ATTENZIONE: calcolo del rank di {TRACKED_NODE} fallito per "
              f"{spec} ({type(e).__name__}: {e}) — colonne lasciate NaN per questa specifica.")
    
    try:
        assort = nx.attribute_assortativity_coefficient(G, "domain")
    except Exception:
        # un solo dominio presente (tutti rad o tutti gen sopravvissuti): non
        # ha senso un coefficiente di assortatività per dominio in quel caso
        assort = np.nan

    row.update({
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "modularity_observed": obs_mod, "modularity_null_mean": null_mod.mean(),
        "modularity_null_std": null_mod.std(), "modularity_z": z_mod,
        "modularity_p_value": p_value,
        "domain_assortativity": obs_assort, "assortativity_z": assort_z,
        "assortativity_p_value": assort_p,
        "tpi1_degree_percentile": tpi1_degree_pct,
        "tpi1_betweenness_percentile": tpi1_betw_pct,
    })
    return row, G



# ---------------------------------------------------------------------------
# L'INTERA CURVA
# ---------------------------------------------------------------------------
def run_network_specification_curve(spec_grid: dict = None, data_source: str = "both",
                                     fdr_mode: str = None, fdr_alpha: float = None,
                                     method: str = None,
                                     n_null: int = None, n_assort_perm: int = None,
                                     random_state: int = config.RANDOM_STATE,
                                     print_info: bool = False):
    """
    Ritorna
    -------
    spec_df : una riga per combinazione, con le metriche di specifica
        (vedi _run_one_spec). Ordinare per 'modularity_z' per il plot.
    node_votes : Series (indice = nome feature) con la FRAZIONE di
        specifiche in cui quella feature compare come nodo non isolato nel
        grafo finale — soggetto al limite sui rappresentanti di cluster
        descritto nel docstring del modulo.
    edge_votes : Series (indice = tupla (feature_1, feature_2), ordinata
        alfabeticamente per coerenza) con la frazione di specifiche in cui
        quella coppia risulta un arco significativo.
    """
    spec_grid = spec_grid or NETWORK_SPEC_GRID
    fdr_mode = fdr_mode or config.NETWORK_FDR_MODE
    fdr_alpha = config.NETWORK_FDR_ALPHA if fdr_alpha is None else fdr_alpha
    method = method or config.RADIOGENOMICS_CORR_METHOD
    n_null = n_null or config.NETWORK_SPEC_CURVE_N_NULL
    n_assort_perm = n_assort_perm or config.NETWORK_ASSORTATIVITY_N_PERM

    X_raw, _ = data_utils.load_data(source=data_source, print_info=False)

    keys = list(spec_grid.keys())
    combos = list(itertools.product(*spec_grid.values()))
    print(f"[run_network_specification_curve] {len(combos)} combinazioni "
          f"({' x '.join(f'{k}={len(v)}' for k, v in spec_grid.items())}), "
          f"fdr_mode='{fdr_mode}', n_null={n_null}, n_assort_perm={n_assort_perm} | "
          f"traccio il rank di '{TRACKED_NODE}'")

    rows = []
    node_counts, edge_counts = {}, {}
    n_valid = 0

    for i, combo in enumerate(combos):
        spec = dict(zip(keys, combo))
        row, G = _run_one_spec(X_raw, spec, data_source=data_source, fdr_mode=fdr_mode,
                                fdr_alpha=fdr_alpha, method=method, n_null=n_null,
                                n_assort_perm=n_assort_perm,
                                random_state=random_state, print_info=print_info)
        rows.append(row)

        if G is not None and G.number_of_edges() > 0:
            n_valid += 1
            for node in G.nodes():
                node_counts[node] = node_counts.get(node, 0) + 1
            for u, v in G.edges():
                pair = tuple(sorted((u, v)))
                edge_counts[pair] = edge_counts.get(pair, 0) + 1

        if not pd.isna(row.get("modularity_z", np.nan)):
            print(f"[run_network_specification_curve] {i+1}/{len(combos)} | {spec} | "
                  f"nodi={row['n_nodes']}, archi={row['n_edges']}, "
                  f"modularity_z={row['modularity_z']:.2f}, "
                  f"assortativity_z={row['assortativity_z']:.2f}, "
                  f"{TRACKED_NODE}_degree_pct={row['tpi1_degree_percentile']}")
        else:
            print(f"[run_network_specification_curve] {i+1}/{len(combos)} | {spec} | "
                  f"rete vuota/non valida")


    spec_df = pd.DataFrame(rows)

    denom = n_valid if n_valid > 0 else len(combos)
    node_votes = pd.Series(node_counts, name="n_specs_present").sort_values(ascending=False) / denom
    edge_votes = pd.Series(edge_counts, name="n_specs_present").sort_values(ascending=False) / denom
    edge_votes.index = pd.MultiIndex.from_tuples(edge_votes.index, names=["feature_1", "feature_2"])

    n_valid_rows = spec_df["modularity_z"].notna().sum()
    n_tpi1_present = spec_df["tpi1_degree_percentile"].notna().sum()
    print(f"\n[run_network_specification_curve] {n_valid_rows}/{len(spec_df)} specifiche con "
          f"rete non vuota | modularity_z: mediana={spec_df['modularity_z'].median():.2f} | "
          f"assortativity_z: mediana={spec_df['assortativity_z'].median():.2f} | "
          f"{TRACKED_NODE} presente e non isolato in {n_tpi1_present}/{len(spec_df)} specifiche")
    if n_valid_rows < len(spec_df):
        print(f"[run_network_specification_curve] ATTENZIONE: {len(spec_df) - n_valid_rows} "
              f"specifiche non hanno prodotto una rete valida — vedi le righe NaN in spec_df.")
    if n_tpi1_present < n_valid_rows:
        print(f"[run_network_specification_curve] NOTA: {TRACKED_NODE} non è sopravvissuto "
              f"(o è rimasto isolato) in {n_valid_rows - n_tpi1_present} specifiche su "
              f"{n_valid_rows} con rete valida — trattato come NaN, non ignorato.")

    return spec_df, node_votes, edge_votes


# ---------------------------------------------------------------------------
# PLOT — stessa grammatica visiva della spec curve ML: pannello superiore con
# la metrica ordinata, pannello inferiore con la matrice a puntini delle
# scelte di preprocessing corrispondenti a ciascun punto.
# ---------------------------------------------------------------------------
def plot_network_specification_curve(spec_df: pd.DataFrame, spec_keys: list,
                                      output_path, metric: str = "modularity_z",
                                      ylabel: str = None, title: str = None,
                                      show_null_reference: bool = True):
    """
    show_null_reference : traccia le linee z=0 / z=1.96, sensate per
        metriche "z-score" (modularity_z, assortativity_z). Per metriche
        già in scala 0-1 come i percentili di rango di TPI1, passare False
        — non c'è un "atteso sotto il nullo" univoco da disegnare lì.
    """
    plot_df = spec_df.dropna(subset=[metric]).sort_values(metric).reset_index(drop=True)
    if len(plot_df) == 0:
        print(f"[plot_network_specification_curve] nessuna specifica valida per '{metric}'.")
        return
    n = len(plot_df)
 
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(max(9, 0.35 * n + 4), 8),
        gridspec_kw={"height_ratios": [2, 1.4]}, sharex=True
    )
 
    ax_top.scatter(range(n), plot_df[metric], color="#4C72B0", zorder=3)
    if show_null_reference:
        ax_top.axhline(0, color="gray", linestyle="--", linewidth=1,
                        label="atteso sotto il nullo (z=0)")
        ax_top.axhline(1.96, color="#C44E52", linestyle=":", linewidth=1,
                        label="z=1.96 (~p=0.05 a due code)")
        ax_top.legend(fontsize=8, loc="upper left")
    ax_top.set_ylabel(ylabel or metric)
    ax_top.set_title(title or f"Network specification curve: {metric}")
 
    row_labels = []
    for key in spec_keys:
        for val in sorted(plot_df[key].unique(), key=str):
            row_labels.append((key, val))
 
    for r, (key, val) in enumerate(row_labels):
        mask = plot_df[key] == val
        ax_bottom.scatter(np.where(mask)[0], [r] * mask.sum(), color="#4C72B0", s=18)
    ax_bottom.set_yticks(range(len(row_labels)))
    ax_bottom.set_yticklabels([f"{k}={v}" for k, v in row_labels], fontsize=8)
    ax_bottom.set_xlabel(f"Specifiche ordinate per {metric} crescente")
    ax_bottom.set_ylim(-0.5, len(row_labels) - 0.5)
    ax_bottom.invert_yaxis()
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_network_specification_curve] salvato in {output_path}")



if __name__ == "__main__":
    out_dir = config.OUTPUT_DIR / "network" / "specification_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
 
    spec_df, node_votes, edge_votes = run_network_specification_curve()
 
    spec_df.to_csv(out_dir / "network_spec_curve_results.csv", index=False)
    node_votes.to_csv(out_dir / "node_votes_across_specs.csv", header=["frac_specs_present"])
    edge_votes.to_csv(out_dir / "edge_votes_across_specs.csv", header=["frac_specs_present"])
 
    plot_network_specification_curve(
        spec_df, spec_keys=list(NETWORK_SPEC_GRID.keys()),
        output_path=out_dir / "network_spec_curve_modularity.png",
        metric="modularity_z",
        ylabel="Modularity z-score\n(vs. Erdős–Rényi a parità di nodi/archi)",
        title="Network specification curve: la struttura a community regge?",
        show_null_reference=True,
    )
    plot_network_specification_curve(
        spec_df, spec_keys=list(NETWORK_SPEC_GRID.keys()),
        output_path=out_dir / "network_spec_curve_assortativity.png",
        metric="assortativity_z",
        ylabel="Assortativity z-score\n(vs. permutazione delle etichette di dominio)",
        title="Network specification curve: la segregazione per dominio regge?",
        show_null_reference=True,
    )
    plot_network_specification_curve(
        spec_df, spec_keys=list(NETWORK_SPEC_GRID.keys()),
        output_path=out_dir / f"network_spec_curve_{TRACKED_NODE}_degree.png",
        metric="tpi1_degree_percentile",
        ylabel=f"Percentile di rango — grado pesato di {TRACKED_NODE}\n(1.0 = nodo più centrale)",
        title=f"Network specification curve: quanto resta centrale {TRACKED_NODE}?",
        show_null_reference=False,
    )
    plot_network_specification_curve(
        spec_df, spec_keys=list(NETWORK_SPEC_GRID.keys()),
        output_path=out_dir / f"network_spec_curve_{TRACKED_NODE}_betweenness.png",
        metric="tpi1_betweenness_percentile",
        ylabel=f"Percentile di rango — betweenness di {TRACKED_NODE}\n(1.0 = nodo più centrale)",
        title=f"Network specification curve: {TRACKED_NODE} resta un ponte tra community?",
        show_null_reference=False,
    )
 
    print("\nTop 15 nodi per frazione di specifiche in cui compaiono (non isolati):")
    print(node_votes.head(15))
    print("\nTop 15 archi per frazione di specifiche in cui risultano significativi:")
    print(edge_votes.head(15))
 
    print(f"\nTutti i risultati sono stati salvati in: {out_dir}")
