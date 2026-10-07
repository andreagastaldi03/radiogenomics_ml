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
                   random_state: int, print_info: bool = False,
                   graphs_dir = None):
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
        "tpi1_degree_weighted": np.nan, "tpi1_betweenness": np.nan,
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
    
    if graphs_dir is not None:
        graph_path = graphs_dir / f"{_spec_id(spec, fdr_mode)}.graphml"
        nx.write_graphml(G, graph_path)

    # --- modularità vs modello nullo Erdos-Rényi ---
    obs_mod, null_mod, p_value = nd.null_model_comparison(G, n_null=n_null,
                                                           random_state=random_state)
    z_mod = ((obs_mod - null_mod.mean()) / null_mod.std()) if null_mod.std() > 0 else np.nan

    # --- assortatività vs modello nullo per permutazione di dominio ---
    obs_assort, _, assort_p, assort_z = nd.domain_assortativity_permutation_test(
        G, n_permutations=n_assort_perm, random_state=random_state
    )

    # --- rank percentile di TPI1 o del nodo tracciato, più valori grezzi ---
    tpi1_degree_pct, tpi1_betw_pct = np.nan, np.nan
    tpi1_degree_weighted, tpi1_betweenness = np.nan, np.nan
    try:
        stats_df = na.compute_network_stats(G).set_index("feature")
        tpi1_degree_pct = _percentile_rank(stats_df, TRACKED_NODE, "degree_weighted")
        tpi1_betw_pct = _percentile_rank(stats_df, TRACKED_NODE, "betweenness")
        if TRACKED_NODE in stats_df.index:
            tpi1_degree_weighted = float(stats_df.loc[TRACKED_NODE, "degree_weighted"])
            tpi1_betweenness = float(stats_df.loc[TRACKED_NODE, "betweenness"])
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
        "tpi1_degree_weighted": tpi1_degree_weighted,
        "tpi1_betweenness": tpi1_betweenness,
    })
    return row, G


def _spec_id(spec:dict, fdr_mode: str = None) -> str:
    """Nome file sicuro che codifica la combinazione di specifica (+ fdr_mode)."""
    parts = [f"{k}-{str(v).replace(' ', '')}" for k, v in spec.items()]
    if fdr_mode is not None:
        parts.append(f"fdr-{fdr_mode}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# L'INTERA CURVA
# ---------------------------------------------------------------------------
def run_network_specification_curve(spec_grid: dict = None, data_source: str = "both",
                                     fdr_mode: str = None, fdr_alpha: float = None,
                                     method: str = None,
                                     n_null: int = None, n_assort_perm: int = None,
                                     random_state: int = config.RANDOM_STATE,
                                     print_info: bool = False, graphs_dir=None):
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
        if graphs_dir is not None:
        graphs_dir.mkdir(parents=True, exist_ok=True)

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
                                random_state=random_state, print_info=print_info,
                                graphs_dir=graphs_dir)
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
# GRIGLIA x DUE MODALITÀ FDR: la stessa griglia di preprocessing, lanciata
# una volta con fdr_mode="unified" e una con fdr_mode="separate", per
# verificare se l'effetto di "separate" sul ruolo di TPI1 è coerente attraverso 
# le scelte di preprocessing, o dipende anche da quelle.
# ---------------------------------------------------------------------------
def run_network_specification_curve_fdr_comparison(spec_grid: dict = None, 
                                                   fdr_modes: tuple = ("unified", "separate"),
                                                   data_source: str = "both", 
                                                   fdr_alpha: float = None, method: str = None,
                                                   n_null: int = None, n_assort_perm: int = None,
                                                   random_state: int = config.RANDOM_STATE, 
                                                   print_info: bool = False, out_dir = None):
    """
    Richiama run_network_specification_curve una volta per ciascun
    fdr_mode (nessuna duplicazione di logica), tagga ogni risultato con la
    colonna 'fdr_mode' e concatena. Se out_dir è fornito, salva anche il
    grafo (graphml) di OGNI combinazione preprocessing x fdr_mode in
    out_dir/graphs/<fdr_mode>/.
 
    Ritorna
    -------
    combined : spec_df di tutte le combinazioni, per ENTRAMBE le modalità
        FDR (n_combinazioni x len(fdr_modes) righe), con colonna 'fdr_mode'.
    votes : dict {fdr_mode: (node_votes, edge_votes)} — tenuti separati
        per modalità, perché i "voti" hanno significato solo all'interno
        della stessa soglia di correzione.
    """
    spec_grid = spec_grid or NETWORK_SPEC_GRID
    frames = []
    votes = {}
 
    for fdr_mode in fdr_modes:
        graphs_dir = (out_dir / "graphs" / fdr_mode) if out_dir is not None else None
        print(f"\n{'=' * 70}\nFDR MODE = '{fdr_mode}'\n{'=' * 70}")
        spec_df, node_votes, edge_votes = run_network_specification_curve(
            spec_grid=spec_grid, data_source=data_source, fdr_mode=fdr_mode,
            fdr_alpha=fdr_alpha, method=method, n_null=n_null, n_assort_perm=n_assort_perm,
            random_state=random_state, print_info=print_info, graphs_dir=graphs_dir,
        )
        spec_df = spec_df.copy()
        spec_df["fdr_mode"] = fdr_mode
        frames.append(spec_df)
        votes[fdr_mode] = (node_votes, edge_votes)
 
    combined = pd.concat(frames, ignore_index=True)
 
    valid = combined.dropna(subset=["tpi1_betweenness"])
    if len(valid) > 0:
        summary = valid.groupby("fdr_mode")[["n_edges", "tpi1_degree_weighted",
                                              "tpi1_betweenness",
                                              "tpi1_degree_percentile",
                                              "tpi1_betweenness_percentile"]].median()
        print(f"\n[run_network_specification_curve_fdr_comparison] mediane per fdr_mode "
              f"(su {len(valid)} specifiche valide totali):")
        print(summary.to_string())
 
    return combined, votes


# ---------------------------------------------------------------------------
# PLOT per il confronto unified vs separate
# ---------------------------------------------------------------------------
def plot_fdr_mode_comparison_tpi1(combined: pd.DataFrame, spec_keys: list, output_path):
    """
    Confronto APPAIATO: la stessa combinazione di preprocessing collegata
    da una linea tra il suo risultato con fdr_mode='unified' e con
    fdr_mode='separate'. Risponde direttamente a "passare a 'separate' fa
    crescere il ruolo di TPI1, a parità di tutto il resto?" — più diretto
    del semplice confronto di medie, perché isola l'effetto della sola
    scelta FDR da quello delle altre scelte di preprocessing.
    """
    spec_keys = list(spec_keys)
    pivot_deg = combined.pivot_table(index=spec_keys, columns="fdr_mode",
                                      values="tpi1_degree_percentile")
    pivot_bet = combined.pivot_table(index=spec_keys, columns="fdr_mode",
                                      values="tpi1_betweenness_percentile")
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    for ax, pivot, label in zip(
        axes, [pivot_deg, pivot_bet],
        [f"grado pesato di {TRACKED_NODE}\n(percentile di rango)",
         f"betweenness di {TRACKED_NODE}\n(percentile di rango)"]
    ):
        pivot = pivot.dropna()
        if pivot.empty or "unified" not in pivot.columns or "separate" not in pivot.columns:
            ax.set_title(f"{label}\n(dati insufficienti: servono entrambe le modalità)")
            ax.axis("off")
            continue
        for _, row in pivot.iterrows():
            color = "#2E7D32" if row["separate"] > row["unified"] else "#B71C1C"
            ax.plot(["unified", "separate"], [row["unified"], row["separate"]],
                     color=color, alpha=0.55, marker="o", markersize=7)
        n_up = int((pivot["separate"] > pivot["unified"]).sum())
        n_down = int((pivot["separate"] < pivot["unified"]).sum())
        n_tot = len(pivot)
        ax.set_ylabel(label)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{n_up}/{n_tot} specifiche: cresce con 'separate' (verde)\n"
                     f"{n_down}/{n_tot}: cala (rosso)")
 
    plt.suptitle(f"{TRACKED_NODE}: confronto appaiato unified vs separate, "
                 f"per ciascuna combinazione di preprocessing")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_fdr_mode_comparison_tpi1] salvato in {output_path}")
 
 
def plot_edges_vs_tpi1_role(combined: pd.DataFrame, output_path):
    """
    Grado pesato e betweenness GREZZI (non percentile) di TPI1 in funzione
    del numero di archi sopravvissuti nella rete, colorati per fdr_mode —
    con una retta di tendenza (OLS, solo aiuto visivo, non un test) per
    ciascuna modalità. Risponde a "al calare/crescere della densità della
    rete, il ruolo di TPI1 si rafforza o si diluisce?"
    """
    valid = combined.dropna(subset=["n_edges", "tpi1_degree_weighted", "tpi1_betweenness"])
    if valid.empty:
        print("[plot_edges_vs_tpi1_role] nessuna specifica valida (TPI1 sempre assente/isolato).")
        return
 
    fig, (ax_deg, ax_betw) = plt.subplots(1, 2, figsize=(13, 5.5))
    markers = {"unified": "o", "separate": "^"}
    colors = {"unified": "#4C72B0", "separate": "#C44E52"}
 
    for fdr_mode, sub in valid.groupby("fdr_mode"):
        for ax, col in ((ax_deg, "tpi1_degree_weighted"), (ax_betw, "tpi1_betweenness")):
            ax.scatter(sub["n_edges"], sub[col], label=fdr_mode,
                       marker=markers.get(fdr_mode, "o"), color=colors.get(fdr_mode, "gray"),
                       s=60, alpha=0.8)
            if len(sub) >= 3:
                coeffs = np.polyfit(sub["n_edges"], sub[col], 1)
                xs = np.linspace(sub["n_edges"].min(), sub["n_edges"].max(), 50)
                ax.plot(xs, np.polyval(coeffs, xs), color=colors.get(fdr_mode, "gray"),
                        linestyle="--", alpha=0.6, linewidth=1.5)
 
    ax_deg.set_xlabel("Numero di archi sopravvissuti nella rete")
    ax_deg.set_ylabel(f"Grado pesato di {TRACKED_NODE} (valore grezzo)")
    ax_deg.set_title("Grado pesato vs densità della rete")
    ax_deg.legend(title="correzione FDR", fontsize=8)
 
    ax_betw.set_xlabel("Numero di archi sopravvissuti nella rete")
    ax_betw.set_ylabel(f"Betweenness di {TRACKED_NODE} (valore grezzo)")
    ax_betw.set_title("Betweenness vs densità della rete")
    ax_betw.legend(title="correzione FDR", fontsize=8)
 
    plt.suptitle(f"{TRACKED_NODE}: ruolo strutturale in funzione della densità della rete")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_edges_vs_tpi1_role] salvato in {output_path}")


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
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    combined, votes = run_network_specification_curve_fdr_comparison(out_dir=out_dir)
    
    combined.to_csv(out_dir / "network_spec_curve_fdr_comparison_results.csv", index=False)
    for fdr_mode, (node_votes, edge_votes) in votes.items():
        node_votes.to_csv(out_dir / f"node_votes_across_specs_{fdr_mode}.csv",
                          header=["frac_specs_present"])
        edge_votes.to_csv(out_dir / f"edge_votes_across_specs_{fdr_mode}.csv",
                          header=["frac_specs_present"])
 
    """
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
    """
    
    spec_keys = list(NETWORK_SPEC_GRID.keys())
 
    # un plot "a specification curve" per fdr_mode, sulle metriche già esistenti
    for fdr_mode in combined["fdr_mode"].unique():
        sub = combined[combined["fdr_mode"] == fdr_mode]
        plot_network_specification_curve(
            sub, spec_keys=spec_keys,
            output_path=plots_dir / f"spec_curve_modularity_{fdr_mode}.png",
            metric="modularity_z",
            ylabel="Modularity z-score\n(vs. Erdős–Rényi a parità di nodi/archi)",
            title=f"Network specification curve ({fdr_mode}): la struttura a community regge?",
            show_null_reference=True,
        )
        plot_network_specification_curve(
            sub, spec_keys=spec_keys,
            output_path=plots_dir / f"spec_curve_assortativity_{fdr_mode}.png",
            metric="assortativity_z",
            ylabel="Assortativity z-score\n(vs. permutazione delle etichette di dominio)",
            title=f"Network specification curve: la segregazione per dominio regge?",
            show_null_reference=True,
        )
        plot_network_specification_curve(
            sub, spec_keys=spec_keys,
            output_path=plots_dir / f"spec_curve_{TRACKED_NODE}_betweenness_{fdr_mode}.png",
            metric="tpi1_betweenness_percentile",
            ylabel=f"Percentile di rango — betweenness di {TRACKED_NODE}\n(1.0 = nodo più centrale)",
            title=f"Network specification curve ({fdr_mode}): {TRACKED_NODE} resta un ponte?",
            show_null_reference=False,
        )
        plot_network_specification_curve(
            sub, spec_keys=spec_keys,
            output_path=plots_dir / f"spec_curve_{TRACKED_NODE}_degree_weighted_{fdr_mode}.png",
            metric="tpi1_degree_percentile",
            ylabel=f"Percentile di rango — grado pesato di {TRACKED_NODE}\n(1.0 = nodo più centrale)",
            title=f"Network specification curve: quanto resta centrale {TRACKED_NODE}?",
            show_null_reference=False,
        )
 
    # i due plot nuovi, pensati apposta per verificare l'affermazione sulle slide
    plot_fdr_mode_comparison_tpi1(combined, spec_keys,
                                   plots_dir / f"{TRACKED_NODE}_unified_vs_separate.png")
    plot_edges_vs_tpi1_role(combined, plots_dir / f"{TRACKED_NODE}_role_vs_density.png")
 
    print(f"\nTutti i risultati sono stati salvati in: {out_dir}")
    print(f"  grafi (graphml) in: {out_dir / 'graphs'}")
    print(f"  plot in: {plots_dir}")

