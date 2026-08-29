"""
Diagnostica per lo studio di rete (network_analysis.py): la struttura
osservata è strutturale o in parte artefatto del campionamento con n=54?

Due controlli, sullo stesso principio già usato nel resto del progetto
(diagnostics.py per il modello ML, radiogenomics.py per il coefficiente RV):

1. BOOTSTRAP SULLA STABILITÀ DEGLI ARCHI (bootstrap_edge_stability)
   Ricampiona i pazienti con reinserimento, ricostruisce l'intera edge list
   e la soglia FDR da zero ad ogni iterazione, e misura in quale frazione
   dei bootstrap ciascun arco osservato sui dati originali "sopravvive".
   Stesso principio di ml_pipeline.bootstrap_stability_selection, applicato
   agli archi invece che alle feature: un arco presente nel grafo osservato
   ma instabile sotto bootstrap è un candidato debole, anche se ha passato
   la soglia FDR sui dati originali.

2. MODELLO NULLO SU DENSITÀ E MODULARITÀ (null_model_comparison)
   Confronta la modularità osservata (quanto la migliore partizione in
   community "spiega" gli archi) con quella di grafi Erdős–Rényi casuali a
   parità di nodi e di numero di archi (quindi stessa densità per
   costruzione): la struttura a community osservata è più forte di quanto
   ci si aspetterebbe da un grafo ugualmente denso ma senza organizzazione?

Entrambi i controlli sono costosi (il bootstrap rifà l'intera pipeline di
correlazione+FDR ad ogni iterazione): tienili bassi durante lo sviluppo,
alzali solo per la versione finale.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # backend non interattivo, per salvare plot da script
import matplotlib.pyplot as plt

import config
import radiogenomics as rg
import network_analysis as na


# ---------------------------------------------------------------------------
# 1) BOOTSTRAP SULLA STABILITÀ DEGLI ARCHI
# ---------------------------------------------------------------------------
def bootstrap_edge_stability(rad_df: pd.DataFrame, gene_df: pd.DataFrame,
                              n_bootstrap: int = 200, method: str = None,
                              fdr_mode: str = None, fdr_alpha: float = None,
                              random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Ritorna un DataFrame con un arco osservato per riga (feature_1,
    feature_2, selection_frequency = frazione di bootstrap in cui l'arco
    resta sotto soglia FDR). Un arco con selection_frequency bassa (es.
    <0.5) è statisticamente fragile anche se compare nel grafo osservato:
    va segnalato come tale.
    """
    method = method or config.RADIOGENOMICS_CORR_METHOD
    fdr_mode = fdr_mode or config.NETWORK_FDR_MODE
    fdr_alpha = config.NETWORK_FDR_ALPHA if fdr_alpha is None else fdr_alpha

    observed_edges = na.build_edge_list(rad_df, gene_df, method=method,
                                         fdr_mode=fdr_mode, fdr_alpha=fdr_alpha)
    observed_sig = observed_edges[observed_edges["q_value"] < fdr_alpha]
    observed_pairs = set(zip(observed_sig["feature_1"], observed_sig["feature_2"]))
    if not observed_pairs:
        print("[bootstrap_edge_stability] nessun arco osservato da validare: nessun bootstrap eseguito.")
        return pd.DataFrame(columns=["feature_1", "feature_2", "selection_frequency"])
    print(f"\n[bootstrap_edge_stability] {len(observed_pairs)} archi osservati da validare "
          f"con {n_bootstrap} bootstrap (può richiedere diversi minuti)")

    common_idx = rad_df.index.intersection(gene_df.index)
    rad_df, gene_df = rad_df.loc[common_idx], gene_df.loc[common_idx]
    n = len(common_idx)

    rng = np.random.RandomState(random_state)
    counts = {pair: 0 for pair in observed_pairs}

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)  # pazienti ricampionati con reinserimento
        # reset_index: con reinserimento l'indice avrebbe pazienti duplicati,
        # e _correlation_pairs allinea rad/gene per indice — reset a un range
        # index 0..n-1 evita qualunque ambiguità nell'allineamento
        rad_b = rad_df.iloc[idx].reset_index(drop=True)
        gene_b = gene_df.iloc[idx].reset_index(drop=True)

        edges_b = na.build_edge_list(rad_b, gene_b, method=method,
                                      fdr_mode=fdr_mode, fdr_alpha=fdr_alpha)
        sig_b = edges_b[edges_b["q_value"] < fdr_alpha]
        pairs_b = set(zip(sig_b["feature_1"], sig_b["feature_2"]))

        for pair in observed_pairs:
            if pair in pairs_b:
                counts[pair] += 1
        if (b + 1) % 20 == 0:
            print(f"[bootstrap_edge_stability] {b+1}/{n_bootstrap} bootstrap completati")

    rows = [{"feature_1": p[0], "feature_2": p[1],
             "selection_frequency": c / n_bootstrap} for p, c in counts.items()]
    stability_df = pd.DataFrame(rows).sort_values("selection_frequency", ascending=False)

    n_stable = int((stability_df["selection_frequency"] >= 0.5).sum())
    print(f"\n[bootstrap_edge_stability] frequenza di selezione: "
          f"media={stability_df['selection_frequency'].mean():.3f}, "
          f"mediana={stability_df['selection_frequency'].median():.3f}")
    print(f"[bootstrap_edge_stability] {n_stable}/{len(stability_df)} archi con "
          f"selection_frequency >= 0.5 (\"stabili\" sotto ricampionamento dei pazienti)")

    return stability_df


def plot_edge_stability(stability_df: pd.DataFrame, output_path):
    """Istogramma della frequenza di selezione bootstrap degli archi osservati."""
    plt.figure(figsize=(7, 5))
    plt.hist(stability_df["selection_frequency"], bins=20, color="#4C72B0", edgecolor="white")
    plt.axvline(0.5, color="#C44E52", linestyle="--", linewidth=1.5,
                label="soglia di stabilità (0.5)")
    plt.xlabel("Frequenza di selezione nei bootstrap")
    plt.ylabel("Numero di archi")
    plt.title("Stabilità degli archi osservati sotto ricampionamento dei pazienti")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_edge_stability] salvato in {output_path}")


# ---------------------------------------------------------------------------
# 2) MODELLO NULLO SU DENSITÀ E MODULARITÀ
# ---------------------------------------------------------------------------
def null_model_comparison(G: nx.Graph, n_null: int = 500,
                           random_state: int = config.RANDOM_STATE):
    """
    Confronta la modularità osservata con quella di grafi Erdős–Rényi
    casuali a parità di nodi e di numero di archi (nx.gnm_random_graph:
    stessa densità per costruzione, così il confronto isola l'effetto
    della struttura e non quello della densità).
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_edges == 0:
        print("[null_model_comparison] grafo senza archi: nessun confronto possibile.")
        return None, None, None

    observed_density = nx.density(G) # returns the ratio of actual edges to all possible edges in a graph G.
    observed_communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    observed_modularity = nx.algorithms.community.modularity(G, observed_communities, weight="weight")

    print(f"\n[null_model_comparison] grafo osservato: {n_nodes} nodi, {n_edges} archi, "
          f"densità={observed_density:.4f}, modularità={observed_modularity:.4f}")
    print(f"[null_model_comparison] confronto con {n_null} grafi Erdős–Rényi "
          f"(stessi nodi/archi, senza pesi né struttura)...")

    rng = np.random.RandomState(random_state)
    null_modularity = np.empty(n_null)
    for i in range(n_null):
        seed = rng.randint(0, 1000000)
        G_null = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
        comms = nx.algorithms.community.greedy_modularity_communities(G_null)
        null_modularity[i] = nx.algorithms.community.modularity(G_null, comms)

    b = int(np.sum(null_modularity >= observed_modularity))
    p_value = (b + 1) / (n_null + 1)

    print(f"[null_model_comparison] modularità nulla: "
          f"{null_modularity.mean():.4f} ± {null_modularity.std():.4f}")
    print(f"[null_model_comparison] p-value empirico: {p_value:.4f} "
          f"({b}/{n_null} grafi nulli >= alla modularità osservata)")
    if p_value >= 0.05:
        print("[null_model_comparison] ATTENZIONE: la modularità osservata non è "
              "significativamente più alta di un grafo casuale altrettanto denso. La "
              "struttura a community trovata potrebbe riflettere principalmente la densità "
              "della rete, non una vera organizzazione in moduli biologici.")
    else:
        print("[null_model_comparison] La modularità osservata è significativamente più alta "
              "di quella attesa per caso a parità di densità: la struttura a community sembra "
              "riflettere un'organizzazione reale.")

    return observed_modularity, null_modularity, p_value


def plot_null_model_comparison(observed_modularity: float, null_modularity: np.ndarray, output_path):
    plt.figure(figsize=(7, 5))
    plt.hist(null_modularity, bins=30, color="#8C8C8C", edgecolor="white",
              label="modularità su grafi Erdős–Rényi (stessa densità)")
    plt.axvline(observed_modularity, color="#C44E52", linewidth=2,
                label=f"modularità osservata = {observed_modularity:.3f}")
    plt.xlabel("Modularità")
    plt.ylabel("Numero di grafi nulli")
    plt.title("La struttura a community osservata è più forte di quanto atteso per caso?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_null_model_comparison] salvato in {output_path}")


# ---------------------------------------------------------------------------
# 3) ASSORTATIVITÀ PER DOMINIO (rad vs gen) — statistica aggiuntiva, poco costosa
# ---------------------------------------------------------------------------
def domain_assortativity(G: nx.Graph) -> float:
    """
    Coefficiente di assortatività per l'attributo 'domain' (rad/gen): >0
    indica che i nodi tendono a legarsi con nodi dello stesso dominio più
    spesso che con l'altro (i due blocchi sono internamente coerenti e
    poco connessi tra loro); ~0 indica mescolamento casuale; <0 indica il
    contrario (i nodi si legano preferenzialmente con l'altro dominio).
    """
    r = nx.attribute_assortativity_coefficient(G, "domain")
    print(f"\n[domain_assortativity] coefficiente di assortatività per dominio: {r:.4f}")
    if r > 0.2:
        print("[domain_assortativity] I due domini (radiomica/genomica) sono nettamente "
              "più coerenti al loro interno che collegati tra loro.")
    return r


# ---------------------------------------------------------------------------
# 4) RETE "CONFERMATA": solo archi che superano FDR *e* stabilità bootstrap
# ---------------------------------------------------------------------------
def build_confirmed_graph(G: nx.Graph, stability_df: pd.DataFrame,
                           stability_threshold: float = 0.5) -> nx.Graph:
    """
    Versione finale: un arco resta nel grafo solo se (a) ha superato la correzione 
    FDR e (b) ha selection_frequency >= stability_threshold nel bootstrap sui pazienti. 
    È la rete più difendibile — gli archi "fragili" (significativi sui dati originali ma
    instabili sotto ricampionamento) vengono tolti, non nascosti: restano comunque in 
    stability_df per essere citati come segnale borderline.
    """
    stable_pairs = set(
        tuple(row) for row in
        stability_df[stability_df["selection_frequency"] >= stability_threshold]
        [["feature_1", "feature_2"]].to_numpy()
    )

    G_confirmed = G.copy()
    dropped = 0
    for u, v in list(G.edges()):
        if (u, v) not in stable_pairs and (v, u) not in stable_pairs:
            G_confirmed.remove_edge(u, v)
            dropped += 1
    isolated = list(nx.isolates(G_confirmed))
    G_confirmed.remove_nodes_from(isolated)

    print(f"\n[build_confirmed_graph] soglia stabilità={stability_threshold}: "
          f"{dropped} archi fragili rimossi, {len(isolated)} nodi rimasti isolati | "
          f"rete confermata: {G_confirmed.number_of_nodes()} nodi, "
          f"{G_confirmed.number_of_edges()} archi")
    return G_confirmed


if __name__ == "__main__":
    out_dir = config.OUTPUT_DIR / "network"
    out_dir.mkdir(parents=True, exist_ok=True)

    rad_features, gene_features, consensus = rg.load_stable_feature_sets()
    rad_df, gene_df = rg._load_raw_values(rad_features, gene_features)

    print("\n" + "=" * 70)
    print("MODELLO NULLO SU DENSITÀ E MODULARITÀ")
    print("=" * 70)
    edge_long_df = na.build_edge_list(rad_df, gene_df)
    G = na.build_graph(edge_long_df, consensus=consensus)

    domain_assortativity(G)

    observed_modularity, null_modularity, p_value = null_model_comparison(G)
    if null_modularity is not None:
        plot_null_model_comparison(observed_modularity, null_modularity,
                                    out_dir / "null_model_modularity.png")
        pd.Series(null_modularity, name="modularity_null").to_csv(
            out_dir / "null_model_modularity_distribution.csv", index=False
        )

    print("\n" + "=" * 70)
    print("BOOTSTRAP SULLA STABILITÀ DEGLI ARCHI")
    print("=" * 70)
    stability_df = bootstrap_edge_stability(rad_df, gene_df, n_bootstrap=200)
    stability_df.to_csv(out_dir / "edge_stability_bootstrap.csv", index=False)
    plot_edge_stability(stability_df, out_dir / "edge_stability.png")

    print("\n" + "=" * 70)
    print("RETE CONFERMATA (FDR + stabilità bootstrap)")
    print("=" * 70)
    G_confirmed = build_confirmed_graph(G, stability_df, stability_threshold=0.5)
    confirmed_stats = na.compute_network_stats(G_confirmed)
    confirmed_stats.to_csv(out_dir / "network_confirmed_node_stats.csv", index=False)
    na.plot_network(G_confirmed, confirmed_stats, out_dir / "network_confirmed_plot.png")
    nx.write_graphml(G_confirmed, out_dir / "network_confirmed.graphml")

    with open(out_dir / "network_diagnostics_summary.txt", "w") as f:
        f.write(f"Nodi: {G.number_of_nodes()} | Archi: {G.number_of_edges()} | "
                f"Densità: {nx.density(G):.4f}\n")
        f.write(f"Assortatività per dominio (rad/gen): "
                f"{nx.attribute_assortativity_coefficient(G, 'domain'):.4f}\n")
        if null_modularity is not None:
            f.write(f"Modularità osservata: {observed_modularity:.4f} | "
                    f"nulla: {null_modularity.mean():.4f} ± {null_modularity.std():.4f} | "
                    f"p-value: {p_value:.4f}\n")
        n_stable = int((stability_df['selection_frequency'] >= 0.5).sum())
        f.write(f"Archi stabili (selection_frequency >= 0.5 su bootstrap): "
                f"{n_stable}/{len(stability_df)}\n")
        f.write(f"Rete confermata (FDR + bootstrap): {G_confirmed.number_of_nodes()} nodi, "
                f"{G_confirmed.number_of_edges()} archi\n")

    print(f"\nTutti i risultati sono stati salvati in: {out_dir}")