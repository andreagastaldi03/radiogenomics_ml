"""
Selezione di feature radiomiche a partire da un grafo di correlazione (.graphml)
costruito da network_analysis.py (nodi "rad__*" e "gen__*", archi con attributi
weight / correlation / distance / q_value / block).

Il grafo NEUTRAL non usa la label, quindi scegliere le feature da qui non
introduce leakage di etichetta (la rete è calcolata su tutti i pazienti, come
la riduzione neutra).

Criteri (select_radiomic_features):
  gene_strength : somma di |r| degli archi verso i geni. Premia le feature
                  radiomiche con più/più forti legami con l'espressione genica.
                  Considera solo feature con almeno un arco verso un gene.
  betweenness   : centralità di intermediazione (cammini minimi con
                  lunghezza = 1-|r|): feature "ponte" tra gruppi di nodi.
  eigenvector   : centralità di autovettore, calcolata per componente
                  connessa (il grafo è spesso non connesso). Nelle reti di
                  correlazione somiglia molto alla forza del nodo: premia i
                  nodi nel blocco più denso di correlazioni.
  community_rep : Louvain sul sottografo radiomica-radiomica; per ogni
                  community il rappresentante col maggior gene_strength (a
                  parità, maggiore forza interna); si prendono le community
                  con il rappresentante più legato ai geni.
  random        : controllo, estrazione casuale dai nodi radiomici.

Per i criteri a classifica (gene_strength, betweenness, eigenvector) si applica
una selezione greedy con controllo di ridondanza: un candidato viene saltato se
è collegato a una feature già scelta da un arco con |r| >= redundancy_max_abs_r.
Senza questo controllo i primi k nodi per centralità sono quasi sempre quasi
duplicati dello stesso blocco di texture.
"""

import networkx as nx
import numpy as np
import pandas as pd

import config


def load_network(path) -> nx.Graph:
    G = nx.read_graphml(path)
    # nodo partenza, nodo arrivo, dict con attr associati all'arco
    for _, _, d in G.edges(data=True):
        d["weight"] = abs(float(d.get("weight", 1.0)))
        d["correlation"] = float(d.get("correlation", d["weight"]))
        d["distance"] = float(d.get("distance", 1.0 - d["weight"]))
    return G


def stable_louvain(G: nx.Graph, seed: int = config.RANDOM_STATE, weight: str = "weight") -> list:
    """
    Louvain riproducibile. L'implementazione di networkx dipende dall'ordine di
    iterazione di insiemi di stringhe (quindi da PYTHONHASHSEED): a parità di
    seed si possono ottenere partizioni diverse tra un'esecuzione e l'altra.
    Qui il grafo viene ricostruito con nodi e archi in ordine alfabetico e le
    community restituite in ordine canonico (dimensione decrescente, poi nome).
    """
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes()))
    H.add_weighted_edges_from(
        sorted((min(u, v), max(u, v), abs(float(d.get(weight, 1.0)))) for u, v, d in G.edges(data=True)))
    comms = nx.community.louvain_communities(H, weight="weight", seed=seed)
    return sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c[0]))


def _eigenvector_per_component(G: nx.Graph) -> dict:
    out = {}
    for comp in nx.connected_components(G):
        # Ciascun comp è un set contenente i nodi che appartengono a quella specifica componente
        if len(comp) < 3:
            out.update({n: 0.0 for n in comp})
        else:
            out.update(nx.eigenvector_centrality_numpy(G.subgraph(comp), weight="weight"))
    return out
# dizionario contenente ogni nodo del grafo originale con il rispettivo valore di centralità relativo 
# alla propria componente di appartenenza


def radiomic_node_scores(G: nx.Graph, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Una riga per nodo radiomico con tutte le misure usate dai selettori."""
    rad = [n for n in G if str(n).startswith("rad__")]
    eig = _eigenvector_per_component(G)
    betw = nx.betweenness_centrality(G, weight="distance")

    H = G.subgraph(rad)
    comms = stable_louvain(G.subgraph(rad), seed=seed)
    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    # Ordina le comunità dalla più grande alla più piccola in base al numero di elementi
    # Assegna un indice progressivo (0, 1, 2, ...) a ciascuna comunità
    # Assegna quell'indice a OGNI singolo nodo contenuto in quella comunità

    rows = {}
    for n in rad:
        gene_edges = [(m, abs(d["correlation"])) for m, d in G[n].items() if str(m).startswith("gen__")]
        gene_edges.sort(key=lambda t: -t[1])
        rows[n] = {
            "degree": G.degree(n),
            "strength": G.degree(n, weight="weight"),
            "eigenvector": eig.get(n, 0.0),
            "betweenness": betw.get(n, 0.0),
            "gene_degree": len(gene_edges),
            "gene_strength": float(sum(r for _, r in gene_edges)),
            "gene_max_abs_r": gene_edges[0][1] if gene_edges else 0.0,
            "top_genes": ";".join(m.replace("gen__", "") for m, _ in gene_edges[:3]),
            "community": comm_of.get(n, -1),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _redundant(G, cand, selected, thr) -> bool:
    return any(G.has_edge(cand, s) and abs(G[cand][s]["correlation"]) >= thr for s in selected)


def select_radiomic_features(G: nx.Graph, scores: pd.DataFrame, method: str, k: int,
                             redundancy_max_abs_r: float = 0.7, rng=None) -> list:
    """Ritorna al più k nomi di feature radiomiche (meno se i candidati non bastano)."""
    if method == "random":
        rng = rng or np.random.RandomState(config.RANDOM_STATE)
        return list(rng.choice(scores.index.to_numpy(), size=min(k, len(scores)), replace=False))
    # Prende casualmente k indici (nomi di feature) senza reimmissione, baseline di controllo nullo

    if method == "community_rep":
        # miglior rappresentante per ciascun modulo
        reps = []
        for _, grp in scores.groupby("community"):
        # pandas può raggruppare le righe di scores in base all'ID numerico che è stato assegnato
            best = grp.sort_values(["gene_strength", "strength"], ascending=False).iloc[0]
            reps.append((best["gene_strength"], len(grp), best["strength"], best.name))
            # Salva una tupla
        reps.sort(key=lambda t: (-t[0], -t[1], -t[2]))
        return [r[3] for r in reps[:k]]

    if method not in ("gene_strength", "betweenness", "eigenvector"):
        raise ValueError(f"selettore '{method}' non valido")
    order = scores.sort_values(method, ascending=False)
    if method == "gene_strength":
        order = order[order["gene_strength"] > 0]   # niente legame con i geni = niente punteggio
    selected = []
    for cand in order.index:
        if len(selected) >= k:
        # se raggiunto k feature, interrompo
            break
        if not _redundant(G, cand, selected, redundancy_max_abs_r):
            selected.append(cand)
    return selected
