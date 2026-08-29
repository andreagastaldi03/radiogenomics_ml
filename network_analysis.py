"""
Studio di rete: relazioni tra le feature (radiomica x radiomica, gene x
gene, radiomica x gene) — completa run_analysis.py/radiogenomics.py
estendendo la logica già usata per il singolo blocco incrociato rad x gen
a tutte e tre le combinazioni possibili, rappresentate come un unico
grafo pesato (nodi = feature, archi = correlazioni significative).

Motivazione: la classificazione binaria ADK/SCC è un collo di bottiglia.
Qui si abbandona del tutto la label per guardare direttamente come si
organizza la struttura di correlazione tra le feature stesse, indipendente
dal fenotipo tumorale. La label rientra solo alla fine, come attributo dei
nodi per l'interpretazione (es. "gli hub della rete coincidono con le
feature più discriminative nello studio ML?"), mai per decidere quali
archi tenere.

Riusa da radiogenomics.py:
- load_stable_feature_sets / _load_raw_values: stessa selezione di feature
  "stabili" (stability selection + SHAP + voti spec curve), indispensabile
  con n=54 pazienti per non costruire una rete su centinaia di feature/geni
  grezzi con potenza statistica quasi nulla per arco;
- _benjamini_hochberg: stessa correzione per test multipli.

Dati: config.RADIOMICS_PATH punta già esclusivamente a out_CTinvivo_roiOrig.csv, 
quindi data_utils.load_data(source="both") non richiede alcuna modifica per 
restringere la sorgente radiomica.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")  # backend non interattivo, per salvare plot da script
import matplotlib.pyplot as plt

import config
import radiogenomics as rg  # riusa load_stable_feature_sets, _load_raw_values, _benjamini_hochberg


# ---------------------------------------------------------------------------
# 0) SCELTA DEL SET DI FEATURE SU CUI COSTRUIRE LA RETE
# ---------------------------------------------------------------------------
def load_feature_set(feature_set: str = "neutral", data_source: str = "both",
                      min_criteria: int = None):
    """
    Due studi complementari, con scopi diversi:
 
    feature_set="stable" -> nodi = feature "stabili" del modello ML
        (stability selection + SHAP + voti spec curve), via
        radiogenomics.load_stable_feature_sets(). Gli ARCHI restano
        indipendenti dalla label (correlazione feature-feature, mai
        feature-label), ma la scelta dei nodi dipende dalla label: solo
        feature già dimostrate rilevanti per ADK/SCC possono comparire.
        Risponde a "come si relazionano tra loro le feature che il modello
        ha trovato importanti?" — un ponte diretto con lo studio ML, non
        uno studio radiogenomico generale.
 
    feature_set="neutral" -> nodi = tutte le feature sopravvissute alla sola
        riduzione neutra (varianza + ridondanza), lette direttamente da
        X_reduced_features.csv — la stessa riduzione già usata come input
        del modello ML in run_analysis.py, non un nuovo criterio scelto ad
        hoc per l'occasione. Nessuna dipendenza dalla label, né negli archi
        né nella scelta dei nodi: è lo studio radiogenomico generale.
 
    Ritorna rad_df, gene_df, consensus (consensus è None per "neutral": non
    esiste un consensus ML da cui leggere n_criteria_present per feature che
    il modello potrebbe non aver mai visto come importanti).
    """
    if feature_set == "stable":
        rad_features, gene_features, consensus = rg.load_stable_feature_sets(
            data_source=data_source, min_criteria=min_criteria)
        rad_df, gene_df = rg._load_raw_values(rad_features, gene_features)
        return rad_df, gene_df, consensus
 
    elif feature_set == "neutral":
        path = config.OUTPUT_DIR / data_source / "X_reduced_features.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} non trovato: esegui prima run_analysis.py con "
                f"DATA_SOURCE='{data_source}' (salva X_reduced_features.csv "
                f"come output della sola riduzione neutra, prima di ogni fit)."
            )
        X_reduced = pd.read_csv(path, index_col=0)
        rad_cols = [c for c in X_reduced.columns if c.startswith("rad__")]
        gene_cols = [c for c in X_reduced.columns if c.startswith("gen__")]
        print(f"[load_feature_set:neutral] {path}: {len(rad_cols)} feature radiomiche, "
              f"{len(gene_cols)} geni (dopo sola riduzione neutra")
        return X_reduced[rad_cols], X_reduced[gene_cols], None
 
    else:
        raise ValueError(f"feature_set '{feature_set}' non valido (usa 'stable' o 'neutral')")

        
# ---------------------------------------------------------------------------
# 1) CORRELAZIONI PER BLOCCO (rettangolare o triangolare)
# ---------------------------------------------------------------------------
def _correlation_pairs(df_a: pd.DataFrame, df_b: pd.DataFrame = None,
                        method: str = "spearman", block_name: str = "block",
                        print_info: bool = True) -> pd.DataFrame:
    """
    Calcola correlazione + p-value per coppie di colonne.

    Se df_b è None: correlazioni intra-blocco su df_a — solo il triangolo
    superiore (i<j), niente autocorrelazione feature-con-se-stessa e
    niente coppie duplicate (a,b)/(b,a). Usato per rad-rad e gen-gen.

    Se df_b è fornito: correlazioni rettangolari tra ogni colonna di df_a
    e ogni colonna di df_b — stesso comportamento di
    radiogenomics.pairwise_correlation_matrix. Usato per rad-gen.

    Ritorna un DataFrame lungo: block, feature_1, feature_2, correlation, p_value.
    """
    corr_fn = spearmanr if method == "spearman" else pearsonr
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method '{method}' non valido (usa 'spearman' o 'pearson')")

    rows = []
    if df_b is None:
        cols = list(df_a.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r, p = corr_fn(df_a[cols[i]].to_numpy(), df_a[cols[j]].to_numpy())
                rows.append({"block": block_name, "feature_1": cols[i], "feature_2": cols[j],
                             "correlation": r, "p_value": p})
    else:
        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) < len(df_a) or len(common_idx) < len(df_b):
            print(f"[_correlation_pairs:{block_name}] ATTENZIONE: allineati {len(common_idx)} "
                  f"pazienti su {len(df_a)}/{len(df_b)}.")
        df_a, df_b = df_a.loc[common_idx], df_b.loc[common_idx]
        for ca in df_a.columns:
            for cb in df_b.columns:
                r, p = corr_fn(df_a[ca].to_numpy(), df_b[cb].to_numpy())
                rows.append({"block": block_name, "feature_1": ca, "feature_2": cb,
                             "correlation": r, "p_value": p})

    df = pd.DataFrame(rows)
    if print_info:
        print(f"[_correlation_pairs:{block_name}] {len(df)} coppie testate "
              f"({method}, {'triangolo superiore' if df_b is None else 'rettangolare'})")
    return df


# ---------------------------------------------------------------------------
# 2) COSTRUZIONE DELLA EDGE LIST COMPLETA (rad-rad, gen-gen, rad-gen)
# ---------------------------------------------------------------------------
def build_edge_list(rad_df: pd.DataFrame, gene_df: pd.DataFrame,
                     method: str = None, fdr_mode: str = None,
                     fdr_alpha: float = None, print_info: bool = True) -> pd.DataFrame:
    """
    Calcola le correlazioni nei tre blocchi (rad-rad, gen-gen, rad-gen) e
    applica la correzione FDR secondo fdr_mode:

    "unified"  (default, config.NETWORK_FDR_MODE) -> un'unica correzione
        Benjamini-Hochberg su tutte le coppie insieme.
    "separate" -> tre correzioni indipendenti, una per blocco.

    Ritorna un unico DataFrame lungo con colonna "block" e "q_value",
    ordinato per q_value crescente.
    """
    method = method or config.RADIOGENOMICS_CORR_METHOD
    fdr_mode = fdr_mode or config.NETWORK_FDR_MODE
    fdr_alpha = config.NETWORK_FDR_ALPHA if fdr_alpha is None else fdr_alpha
    if fdr_mode not in ("unified", "separate"):
        raise ValueError(f"fdr_mode '{fdr_mode}' non valido (usa 'unified' o 'separate')")

    rad_rad = _correlation_pairs(rad_df, method=method, block_name="rad-rad", print_info=print_info)
    gen_gen = _correlation_pairs(gene_df, method=method, block_name="gen-gen", print_info=print_info)
    rad_gen = _correlation_pairs(rad_df, gene_df, method=method, block_name="rad-gen", 
                                 print_info=print_info)

    if fdr_mode == "unified":
        combined = pd.concat([rad_rad, gen_gen, rad_gen], ignore_index=True)
        combined["q_value"] = rg._benjamini_hochberg(combined["p_value"].to_numpy())
    else:
        parts = []
        for part in (rad_rad, gen_gen, rad_gen):
            part = part.copy()
            part["q_value"] = rg._benjamini_hochberg(part["p_value"].to_numpy())
            parts.append(part)
        combined = pd.concat(parts, ignore_index=True)

    combined = combined.sort_values("q_value").reset_index(drop=True)

    n_total = len(combined)
    n_sig = int((combined["q_value"] < fdr_alpha).sum())
    if print_info:
        print(f"\n[build_edge_list] modalità FDR = '{fdr_mode}' | {n_total} coppie totali "
              f"(rad-rad={len(rad_rad)}, gen-gen={len(gen_gen)}, rad-gen={len(rad_gen)}) | "
              f"{n_sig} archi con q<{fdr_alpha}")
    for block_name, part in combined.groupby("block"):
        n_sig_block = int((part["q_value"] < fdr_alpha).sum())
        if print_info:
            print(f"  [{block_name}] {n_sig_block}/{len(part)} coppie con q<{fdr_alpha}")
    if n_sig == 0:
        print("[build_edge_list] ATTENZIONE: nessuna coppia sopravvive alla correzione FDR. "
              "Con n=54 è un esito comune anche in presenza di segnale reale ma debole. Prima "
              "di abbassare la soglia, valuta 'separate' invece di 'unified' "
              "o guarda le coppie con q più basso in valore assoluto come segnale esplorativo, "
              "da riportare come tale e non come risultato confermato.")

    return combined


# ---------------------------------------------------------------------------
# 3) COSTRUZIONE DEL GRAFO
# ---------------------------------------------------------------------------
def build_graph(edge_long_df: pd.DataFrame, fdr_alpha: float = None,
                 consensus: pd.DataFrame = None) -> nx.Graph:
    """
    Costruisce un networkx.Graph non orientato: nodi = feature (rad__/gen__),
    archi = coppie con q_value < fdr_alpha, peso = |correlazione| 
    (il segno resta disponibile come attributo separato 'correlation', per
    distinguere in seguito legami positivi/negativi).

    Ogni nodo riceve un attributo 'domain' ('rad' o 'gen', dal prefisso di
    colonna) e, se consensus è fornito (feature_consensus.csv), gli
    attributi 'consensus_score' e 'n_criteria_present' — usati solo per
    interpretazione/plot successivi, mai per decidere quali archi tenere:
    la costruzione del grafo dipende esclusivamente dalla correlazione tra
    feature, non dal quanto quella feature "conta" nello studio ML.
    """
    fdr_alpha = config.NETWORK_FDR_ALPHA if fdr_alpha is None else fdr_alpha
    sig = edge_long_df[edge_long_df["q_value"] < fdr_alpha]

    G = nx.Graph()
    all_features = set(edge_long_df["feature_1"]) | set(edge_long_df["feature_2"])
    for feat in all_features:
        domain = "rad" if feat.startswith("rad__") else "gen"
        attrs = {"domain": domain}
        if consensus is not None and feat in consensus.index:
            attrs["consensus_score"] = float(consensus.loc[feat, "consensus_score"])
            attrs["n_criteria_present"] = int(consensus.loc[feat, "n_criteria_present"])
        G.add_node(feat, **attrs)

    for _, row in sig.iterrows():
        G.add_edge(row["feature_1"], row["feature_2"],
                    weight=abs(row["correlation"]), correlation=row["correlation"],
                    q_value=row["q_value"], block=row["block"])

    # i nodi rimasti isolati (nessun arco sopravvissuto a FDR) vengono
    # tolti dal grafo: restano comunque nella edge_long_df/consensus salvati
    # su disco, per riferimento e per capire cosa non è entrato in rete
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    print(f"\n[build_graph] q<{fdr_alpha}: {G.number_of_nodes()} nodi, "
          f"{G.number_of_edges()} archi ({len(isolated)} nodi isolati rimossi)")
    return G


# ---------------------------------------------------------------------------
# 4) STATISTICHE DI RETE
# ---------------------------------------------------------------------------
def compute_network_stats(G: nx.Graph) -> pd.DataFrame:
    """
    Statistiche nodo per nodo: grado (pesato e non), centralità di
    betweenness ed eigenvector (pesate su |correlazione|), community
    detection tramite modularità greedy (nativa in networkx, nessuna
    dipendenza aggiuntiva rispetto a python-louvain).
    """
    degree_weighted = dict(G.degree(weight="weight"))
        # restituisce esclusivamente la somma dei pesi, non include informazioni sul numero 
        # effettivo di collegamenti (il grado non pesato).
        # Quanto è complessivamente forte l'insieme delle connessioni di questa feature?
    betweenness = nx.betweenness_centrality(G, weight="weight")
        # Quanto spesso questo nodo si trova lungo i percorsi più brevi che collegano altri nodi?
        # non abbia necessariamente il maggior numero di connessioni, ma potrebbe essere un ponte 
        # tra due moduli biologici/regioni del grafo. La betweenness potrebbe quindi identificare
        # feature che occupano una posizione di intermediazione nella rete. Questo è concettualmente 
        # diverso dal dire che sia una feature molto correlata.
        # How the Weight Parameter Works: Interpretation as Distance - Edge weights are treated as 
        # distances or costs, lower weights mean shorter, more preferred paths, while higher weights 
        # mean longer paths.
    try:
        eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
            # Quanto è importante un nodo considerando anche l'importanza dei nodi a cui è collegato?
            # The function returns a dictionary mapping each node to its computed eigenvector centrality value.
    except nx.PowerIterationFailedConvergence:
        print("[compute_network_stats] ATTENZIONE: eigenvector centrality non converge "
              "(grafo probabilmente troppo sparso o disconnesso in più componenti); "
              "valori impostati a NaN per questo grafo.")
        eigenvector = {n: np.nan for n in G.nodes()}

    communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
        # Quali gruppi di nodi formano comunità naturalmente dense al loro interno?
        # La modularità cerca di capire se il grafo ha più connessioni interne alle comunità di quante ci 
        # aspetteremmo casualmente. Cerca una suddivisione del grafo in comunità che aumenti la modularità.
        # "Greedy" significa, in sostanza, che utilizza una strategia iterativa: fa modifiche locali alla 
        # partizione che migliorano la modularità, cercando progressivamente una buona soluzione. Non 
        # significa necessariamente che trovi la partizione matematicamente ottimale globale.
        # It uses the specified edge attribute to compute edge weights during modularity maximization.
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i

    rows = []
    for node in G.nodes():
        rows.append({
            "feature": node,
            "domain": G.nodes[node].get("domain"),
            "degree_weighted": degree_weighted[node],
            "degree_unweighted": G.degree(node),
            "betweenness": betweenness[node],
            "eigenvector_centrality": eigenvector[node],
            "community": community_map[node],
            "consensus_score": G.nodes[node].get("consensus_score", np.nan),
            "n_criteria_present": G.nodes[node].get("n_criteria_present", np.nan),
        })
    stats_df = pd.DataFrame(rows).sort_values("degree_weighted", ascending=False)

    print(f"\n[compute_network_stats] {len(communities)} community trovate (modularità greedy).")
    print("[compute_network_stats] Top 10 nodi per grado pesato:")
    print(stats_df[["feature", "domain", "degree_weighted", "betweenness", "community"]]
          .head(10).to_string(index=False))

    return stats_df


# ---------------------------------------------------------------------------
# 5) PLOT
# ---------------------------------------------------------------------------
def plot_network(G: nx.Graph, stats_df: pd.DataFrame, output_path,
                  seed: int = config.RANDOM_STATE, n_labels: int = 12):
    """
    Layout force-directed (spring layout pesato su |correlazione|), tarato
    per restare leggibile anche con qualche decina di nodi e centinaia di
    archi:
    - k (repulsione tra nodi) più alto del default, per allargare il grafo
      invece di farlo accartocciare al centro;
    - opacità dell'arco proporzionale al peso (|correlazione|): gli archi
      deboli quasi svaniscono, lasciando emergere visivamente la struttura
      portante senza doverla filtrare nei dati sottostanti;
    - solo i primi n_labels nodi per grado vengono etichettati, con sfondo
      bianco per restare leggibili anche sopra archi/nodi.
    """
    if G.number_of_nodes() == 0:
        print("[plot_network] grafo vuoto (nessun arco sopra soglia FDR): nessun plot generato.")
        return
 
    n = G.number_of_nodes()
    k = 3.0 / np.sqrt(n)  # più alto del default networkx (~1/sqrt(n)): nodi più distanziati
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=k, iterations=200)
 
    degree = dict(G.degree(weight="weight"))
    max_degree = max(degree.values()) or 1
    node_sizes = [80 + 350 * degree[n_] / max_degree for n_ in G.nodes()]
    node_colors = ["#4C72B0" if G.nodes[n_]["domain"] == "rad" else "#C44E52" for n_ in G.nodes()]
 
    weights = np.array([G.edges[e]["weight"] for e in G.edges()])
    w_min, w_max = weights.min(), weights.max()
    w_range = (w_max - w_min) or 1.0
    edge_alphas = 0.05 + 0.5 * (weights - w_min) / w_range  # deboli quasi invisibili
    edge_colors = ["#2E7D32" if G.edges[e]["correlation"] > 0 else "#B71C1C" for e in G.edges()]
 
    plt.figure(figsize=(15, 13))
    for (u, v), color, alpha in zip(G.edges(), edge_colors, edge_alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=alpha,
                                edge_color=color, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
 
    top_nodes = stats_df.nlargest(n_labels, "degree_weighted")["feature"].tolist()
    labels = {n_: n_.split("__", 1)[-1] for n_ in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold",
                             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
 
    plt.title("Rete radiogenomica (feature stabili) — blu=radiomica, rosso=gene\n"
              "arco verde=correlazione positiva, arco rosso=negativa "
              "(opacità ∝ |correlazione|; etichette: top-{} per grado)".format(n_labels))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_network] salvato in {output_path}")


if __name__ == "__main__":
    for feature_set in ("stable", "neutral"):
        rad_df, gene_df, consensus = load_feature_set(feature_set)
 
        for fdr_mode in ("unified", "separate"):
            out_dir = config.OUTPUT_DIR / "network" / feature_set / fdr_mode
            out_dir.mkdir(parents=True, exist_ok=True)
 
            print("\n" + "=" * 70)
            print(f"FEATURE SET = '{feature_set}' | FDR MODE = '{fdr_mode}'")
            print("=" * 70)
 
            print("\n" + "=" * 70)
            print("COSTRUZIONE EDGE LIST (rad-rad, gen-gen, rad-gen)")
            print("=" * 70)
            edge_long_df = build_edge_list(rad_df, gene_df, fdr_mode=fdr_mode)
            edge_long_df.to_csv(out_dir / "correlation_pairs_long_full.csv", index=False)
 
            print("\n" + "=" * 70)
            print("COSTRUZIONE DEL GRAFO")
            print("=" * 70)
            G = build_graph(edge_long_df, consensus=consensus)
            nx.write_graphml(G, out_dir / "network.graphml")
             
            print("\n" + "=" * 70)
            print("STATISTICHE DI RETE")
            print("=" * 70)
            stats_df = compute_network_stats(G)
            stats_df.to_csv(out_dir / "network_node_stats.csv", index=False)
 
            plot_network(G, stats_df, out_dir / "network_plot.png")
 
            print(f"[main] risultati salvati in: {out_dir}")
 
    print(f"\nTutte le combinazioni sono state salvate sotto: {config.OUTPUT_DIR / 'network'}")