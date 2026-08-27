"""
Radiogenomica: la radiomica è associata alla genomica di per sé, indipendentemente
dalla label ADK/SCC?

Motivazione (diversa da tutto il resto della pipeline): run_analysis.py e
specification_curve.py rispondono a "radiomica e/o genomica predicono il
fenotipo?". Qui la domanda è un'altra: la classificazione binaria è un collo
di bottiglia — un'associazione immagine-genoma reale potrebbe non emergere
mai in un modello ADK/SCC se non è (anche) predittiva del fenotipo. Guardando
direttamente radiomica <-> genomica, senza passare per la label, si può
trovare segnale che il modello di classificazione non è progettato per vedere.

Per evitare di testare tutte le ~100 feature radiomiche contro ~decine/centinaia
di geni (migliaia di test, quasi certi falsi positivi), si usano solo le feature
già emerse come "stabili" nel resto della pipeline: quelle con consenso alto in
feature_consensus.py (stability selection + SHAP + voti specification curve).
Questo non è un modo per "pulire" un risultato scomodo: è lo stesso principio
di riduzione della dimensionalità già usato altrove nel progetto, qui applicato
prima di un test che non useremo mai per giustificare la classificazione.

Due letture complementari, entrambe senza mai usare la label:
1. Matrice di correlazione (Spearman di default: non richiede una relazione
   lineare, solo monotona) feature radiomica x gene, con correzione per test
   multipli (Benjamini-Hochberg) — dice quali coppie sembrano associate.
2. Coefficiente RV (Escoufier, 1973) tra i due blocchi + test di permutazione
   — generalizzazione multivariata della correlazione, risponde a una domanda
   più severa e più difendibile: "i due blocchi nel loro insieme sono
   associati più di quanto ci si aspetterebbe rimescolando l'abbinamento
   paziente-per-paziente?", senza fare affidamento su nessuna singola coppia.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, binomtest

import config
import data_utils


# ---------------------------------------------------------------------------
# 1) CARICAMENTO DELLE FEATURE "STABILI" (rad e gen) DAL CONSENSUS
# ---------------------------------------------------------------------------
def load_stable_feature_sets(data_source: str = "both",
                              min_criteria: int = None,
                              consensus_path=None):
    """
    Legge feature_consensus.csv (prodotto da feature_consensus.py) 
    e separa le feature "stabili" in due liste, per prefisso di colonna 
    (rad__ / gen__, vedi data_utils.load_data).

    Parametri
    ---------
    data_source : quale sottocartella di config.OUTPUT_DIR leggere. Deve
        essere "both", perché serve un unico consensus calcolato su un
        modello allenato su radiomica+genomica insieme: un consensus
        calcolato separatamente su "radiomics" e su "genomics" non
        conterrebbe entrambi i tipi di feature nello stesso file.
    min_criteria : soglia su n_criteria_present (0-3). Default
        config.RADIOGENOMICS_MIN_CRITERIA. Più alta = meno feature ma più
        difendibili (compaiono in più criteri indipendenti).

    Ritorna
    -------
    rad_features, gene_features : liste di nomi di colonna (con prefisso)
    consensus : il DataFrame completo, per riferimento/plot
    """
    min_criteria = config.RADIOGENOMICS_MIN_CRITERIA if min_criteria is None else min_criteria
    consensus_path = consensus_path or (config.OUTPUT_DIR / data_source / "feature_consensus.csv")

    if not consensus_path.exists():
        raise FileNotFoundError(
            f"{consensus_path} non trovato: esegui prima run_analysis.py, "
            f"run_specification_curve.py e feature_consensus.py con DATA_SOURCE='{data_source}'."
        )

    consensus = pd.read_csv(consensus_path, index_col=0) # Column(s) to use as row label(s)
    stable = consensus[consensus["n_criteria_present"] >= min_criteria]

    rad_features = sorted([f for f in stable.index if f.startswith("rad__")])
    gene_features = sorted([f for f in stable.index if f.startswith("gen__")])

    print(f"[load_stable_feature_sets] consensus letto da {consensus_path}")
    print(f"[load_stable_feature_sets] soglia n_criteria_present >= {min_criteria}: "
          f"{len(rad_features)} feature radiomiche, {len(gene_features)} geni stabili "
          f"(su {len(consensus)} feature totali nel consensus)")

    if not rad_features or not gene_features:
        raise ValueError(
            f"Con min_criteria={min_criteria} una delle due liste è vuota "
            f"(rad={len(rad_features)}, gen={len(gene_features)}): non è possibile "
            f"testare un'associazione tra due gruppi se uno dei due è vuoto. Abbassa "
            f"RADIOGENOMICS_MIN_CRITERIA o controlla che feature_consensus.csv contenga "
            f"davvero entrambi i tipi di feature."
        )

    return rad_features, gene_features, consensus


def _load_raw_values(rad_features, gene_features):
    """
    Recupera i valori grezzi (non ridotti/standardizzati da neutral_feature_reduction)
    delle feature richieste, direttamente da data_utils.load_data(source="both").
    Si usano i valori grezzi e non la matrice ridotta per due motivi: (1) i nomi
    di colonna restano identici indipendentemente da quale combinazione di
    parametri di riduzione era attiva quando è stato calcolato il consensus, quindi
    non c'è alcun rischio di disallineamento; (2) Spearman è invariante per
    trasformazioni monotone (quindi anche per lo standard scaling), quindi usare i
    valori grezzi non cambia il risultato del test di correlazione.
    """
    X_raw, y = data_utils.load_data(source="both", print_info=False)

    missing_rad = [f for f in rad_features if f not in X_raw.columns]
    missing_gen = [f for f in gene_features if f not in X_raw.columns]
    if missing_rad or missing_gen:
        print(f"[_load_raw_values] ATTENZIONE: {len(missing_rad)} feature radiomiche e "
              f"{len(missing_gen)} geni del consensus non sono presenti nei dati grezzi "
              f"'both' (nomi di colonna cambiati?) e vengono esclusi dal confronto.")
    rad_features = [f for f in rad_features if f in X_raw.columns]
    gene_features = [f for f in gene_features if f in X_raw.columns]

    return X_raw[rad_features], X_raw[gene_features]


# ---------------------------------------------------------------------------
# 2) MATRICE DI CORRELAZIONE PER COPPIE, CON CORREZIONE FDR
# ---------------------------------------------------------------------------
def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Correzione Benjamini-Hochberg per il controllo del False Discovery Rate,
    implementata senza dipendenze esterne (equivalente a
    statsmodels.stats.multitest.multipletests(method='fdr_bh')).

    Con molte coppie radiomica x gene testate insieme, controllare il FDR
    invece del solo p-value grezzo è essenziale: altrimenti, anche senza
    nessuna associazione reale, ci si aspetta comunque circa il 5% delle
    coppie "significative per caso" a p<0.05.
    """
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    # q_i = p_i * n / i, poi si impone la monotonicità 
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    qvals = np.empty(n)
    qvals[order] = q # per associare correttamente i qvals ai pvals
    return qvals


def pairwise_correlation_matrix(rad_df: pd.DataFrame, gene_df: pd.DataFrame,
                                 method: str = None, fdr_alpha: float = None):
    """
    Calcola la correlazione (Spearman di default) tra ogni feature radiomica
    e ogni gene, sugli stessi pazienti, senza mai usare la label.

    Ritorna
    -------
    corr_df : DataFrame (rad_features x gene_features), coefficiente di correlazione
    pval_df : DataFrame, p-value grezzo per coppia
    qval_df : DataFrame, q-value (Benjamini-Hochberg) per coppia — usa questo,
        non pval_df, per decidere quali coppie riportare come "associate"
    long_df : stessa informazione in formato lungo, ordinata per q-value
        crescente, comoda per un csv/report
    """
    method = method or config.RADIOGENOMICS_CORR_METHOD
    fdr_alpha = config.RADIOGENOMICS_FDR_ALPHA if fdr_alpha is None else fdr_alpha
    corr_fn = spearmanr if method == "spearman" else pearsonr
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method '{method}' non valido (usa 'spearman' o 'pearson')")

    common_idx = rad_df.index.intersection(gene_df.index)
    if len(common_idx) < len(rad_df) or len(common_idx) < len(gene_df):
        print(f"[pairwise_correlation_matrix] ATTENZIONE: allineati {len(common_idx)} pazienti "
              f"su {len(rad_df)} (rad) / {len(gene_df)} (gene).")
    rad_df, gene_df = rad_df.loc[common_idx], gene_df.loc[common_idx]

    rad_cols, gene_cols = list(rad_df.columns), list(gene_df.columns)
    corr = np.empty((len(rad_cols), len(gene_cols)))
    pval = np.empty((len(rad_cols), len(gene_cols)))

    for i, rc in enumerate(rad_cols):
        for j, gc in enumerate(gene_cols):
            r, p = corr_fn(rad_df[rc].to_numpy(), gene_df[gc].to_numpy())
            corr[i, j], pval[i, j] = r, p

    qval = _benjamini_hochberg(pval.ravel()).reshape(pval.shape)

    corr_df = pd.DataFrame(corr, index=rad_cols, columns=gene_cols)
    pval_df = pd.DataFrame(pval, index=rad_cols, columns=gene_cols)
    qval_df = pd.DataFrame(qval, index=rad_cols, columns=gene_cols)

    long_rows = []
    for rc in rad_cols:
        for gc in gene_cols:
            long_rows.append({
                "radiomic_feature": rc, "gene": gc,
                "correlation": corr_df.loc[rc, gc],
                "p_value": pval_df.loc[rc, gc],
                "q_value": qval_df.loc[rc, gc],
            })
    long_df = pd.DataFrame(long_rows).sort_values("q_value").reset_index(drop=True)

    n_sig = int((long_df["q_value"] < fdr_alpha).sum())
    n_total = len(long_df)
    print(f"[pairwise_correlation_matrix] metodo={method} | {n_total} coppie testate "
          f"({len(rad_cols)} rad x {len(gene_cols)} geni) | {n_sig} coppie con q<{fdr_alpha} "
          f"(FDR corretto)")
    if n_sig == 0:
        print("[pairwise_correlation_matrix] Nessuna coppia sopravvive alla correzione FDR: "
              "con questo numero di pazienti (n piccolo) è un esito comune anche in presenza "
              "di un'associazione reale ma debole — guarda comunque il coefficiente RV "
              "(rv_permutation_test), che ha più potenza perché non penalizza per test multipli "
              "sulle singole coppie.")

    return corr_df, pval_df, qval_df, long_df


def plot_correlation_heatmap(corr_df: pd.DataFrame, qval_df: pd.DataFrame, output_path,
                             fdr_alpha: float = None):
    """Heatmap della correlazione, con un asterisco sulle coppie che superano la soglia FDR."""
    fdr_alpha = config.RADIOGENOMICS_FDR_ALPHA if fdr_alpha is None else fdr_alpha

    fig_w = max(8, 0.4 * corr_df.shape[1] + 3)
    fig_h = max(6, 0.35 * corr_df.shape[0] + 2)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, label="correlazione (Spearman)")
    plt.xticks(range(corr_df.shape[1]), corr_df.columns, rotation=90, fontsize=7)
    plt.yticks(range(corr_df.shape[0]), corr_df.index, fontsize=7)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if qval_df.iloc[i, j] < fdr_alpha:
                plt.text(j, i, "*", ha="center", va="center", color="black", fontsize=10)

    plt.title(f"Radiomica x genomica (feature stabili) — * = q<{fdr_alpha} (FDR)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_correlation_heatmap] salvato in {output_path}")


# ---------------------------------------------------------------------------
# 3) COEFFICIENTE RV + TEST DI PERMUTAZIONE (associazione multivariata globale)
# ---------------------------------------------------------------------------
def rv_coefficient(A: np.ndarray, B: np.ndarray) -> float:
    """
    Coefficiente RV (Escoufier, 1973): generalizzazione multivariata del
    quadrato del coefficiente di correlazione a due gruppi di variabili
    invece di due singole variabili. Vale 0 se i due blocchi non hanno
    nessuna struttura di covarianza in comune, 1 se le strutture di
    covarianza sono identiche (a meno di scala/rotazione).

    A differenza della matrice di correlazione punto-per-punto, qui non si
    testano N coppie singole (quindi non serve una correzione per test
    multipli): si testa una statistica riassuntiva sull'intero blocco, il
    che dà più potenza quando il segnale è distribuito su molte coppie
    deboli piuttosto che concentrato su una sola coppia forte.

    A, B : array (n_pazienti x n_feature_A), (n_pazienti x n_feature_B), già
        allineati sugli stessi pazienti. Vengono centrati (non scalati: RV
        non è invariante per scala componente per componente, ma qui non è
        un problema dato che confrontiamo lo stesso A e B, con la stessa
        scala, tra la statistica osservata e quelle permutate).
    """
    A = A - A.mean(axis=0, keepdims=True) # Questo serve perché vogliamo concentrarci 
        # sulle variazioni tra pazienti, non sul livello medio assoluto della feature.
        # sottraiamo alla matrice la media delle feature lungo i pazienti
    B = B - B.mean(axis=0, keepdims=True)
    cov_AB = A.T @ B
    cov_AA = A.T @ A
    cov_BB = B.T @ B
    num = np.trace(cov_AB @ cov_AB.T) # misura quanto le due strutture sono allineate. Facendo 
        # la somma sulla diagonale tengo in considerazione i termini corrispondenti
    den = np.sqrt(np.trace(cov_AA @ cov_AA) * np.trace(cov_BB @ cov_BB))
    return float(num / den) if den > 0 else 0.0


def rv_permutation_test(rad_df: pd.DataFrame, gene_df: pd.DataFrame,
                        n_permutations: int = None, random_state: int = config.RANDOM_STATE):
    """
    Rimescola l'abbinamento PAZIENTE-per-PAZIENTE tra blocco radiomico e
    blocco genomico (mantenendo intatta la struttura di covarianza dentro
    ogni blocco) e ricalcola il coefficiente RV molte volte: se il valore
    osservato è chiaramente più alto della distribuzione ottenuta per caso,
    i due blocchi condividono più struttura di quanto ci si aspetterebbe se
    fossero indipendenti — senza mai aver guardato la label ADK/SCC.

    Stessa logica (e stessa correzione b+1/(m+1)) del test di permutazione
    già usato in diagnostics.py e specification_curve.py.
    """
    n_permutations = n_permutations or config.RADIOGENOMICS_N_PERMUTATIONS_RV

    common_idx = rad_df.index.intersection(gene_df.index)
    rad_df, gene_df = rad_df.loc[common_idx].sort_index(), gene_df.loc[common_idx].sort_index()

    A = rad_df.to_numpy(dtype=float)
    B = gene_df.to_numpy(dtype=float)
    n = A.shape[0]

    observed_rv = rv_coefficient(A, B)
    print(f"[rv_permutation_test] n_pazienti={n} | n_feature_rad={A.shape[1]} | "
          f"n_geni={B.shape[1]} | RV osservato = {observed_rv:.4f}")
    print(f"[rv_permutation_test] {n_permutations} permutazioni dell'abbinamento paziente-gene...")

    rng = np.random.RandomState(random_state)
    null_rv = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        null_rv[i] = rv_coefficient(A, B[perm])
        if (i + 1) % 500 == 0:
            print(f"[rv_permutation_test] {i+1}/{n_permutations} permutazioni completate")

    b = int(np.sum(null_rv >= observed_rv))
    p_value = (b + 1) / (n_permutations + 1)
    ci = binomtest(b, n_permutations, alternative="two-sided").proportion_ci(
        confidence_level=0.95, method="exact"
    )

    print(f"\n[rv_permutation_test] RV nullo (permutato): {null_rv.mean():.4f} ± {null_rv.std():.4f}")
    print(f"[rv_permutation_test] RV osservato: {observed_rv:.4f}")
    print(f"[rv_permutation_test] p-value empirico: {p_value:.4f} "
          f"({b}/{n_permutations} permutazioni >= al reale) | "
          f"CI 95% esatta (Clopper-Pearson): [{ci.low:.4f}, {ci.high:.4f}]")

    if p_value >= 0.05:
        print("[rv_permutation_test] ATTENZIONE: nessuna evidenza che il blocco radiomico e il "
              "blocco genomico (feature stabili) condividano più struttura di quanto atteso per "
              "caso. Non significa che non esista nessuna coppia associata (guarda comunque "
              "pairwise_correlation_matrix), ma il blocco nel suo insieme non mostra un segnale "
              "multivariato oltre il rumore.")
    else:
        print("[rv_permutation_test] Evidenza di associazione multivariata reale tra radiomica e "
              "genomica (feature stabili), indipendente dalla label ADK/SCC — coerente con una "
              "lettura radiogenomica, non solo predittiva.")

    return observed_rv, null_rv, p_value, (ci.low, ci.high)


def plot_rv_permutation_test(observed_rv: float, null_rv: np.ndarray, output_path):
    plt.figure(figsize=(7, 5))
    plt.hist(null_rv, bins=40, color="#8C8C8C", edgecolor="white",
              label="RV con abbinamento paziente-gene permutato")
    plt.axvline(observed_rv, color="#C44E52", linewidth=2,
                label=f"RV osservato = {observed_rv:.3f}")
    plt.xlabel("Coefficiente RV (radiomica <-> genomica)")
    plt.ylabel("Numero di permutazioni")
    plt.title("Test di permutazione sul coefficiente RV\n(associazione multivariata, indipendente dalla label)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_rv_permutation_test] salvato in {output_path}")


if __name__ == "__main__":
    out_dir = config.OUTPUT_DIR / "radiogenomics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rad_features, gene_features, consensus = load_stable_feature_sets()
    rad_df, gene_df = _load_raw_values(rad_features, gene_features)

    print("\n" + "=" * 70)
    print("MATRICE DI CORRELAZIONE PER COPPIE (con correzione FDR)")
    print("=" * 70)
    corr_df, pval_df, qval_df, long_df = pairwise_correlation_matrix(rad_df, gene_df)
    corr_df.to_csv(out_dir / "correlation_matrix.csv")
    qval_df.to_csv(out_dir / "qvalue_matrix.csv")
    long_df.to_csv(out_dir / "correlation_pairs_long.csv", index=False)
    plot_correlation_heatmap(corr_df, qval_df, out_dir / "correlation_heatmap.png")

    print("\nTop 15 coppie per q-value:")
    print(long_df.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("COEFFICIENTE RV + TEST DI PERMUTAZIONE (associazione globale)")
    print("=" * 70)
    observed_rv, null_rv, p_value, ci = rv_permutation_test(rad_df, gene_df)
    plot_rv_permutation_test(observed_rv, null_rv, out_dir / "rv_permutation_test.png")
    pd.Series(null_rv, name="rv_permutato").to_csv(
        out_dir / "rv_permutation_null_distribution.csv", index=False
    )
    with open(out_dir / "rv_permutation_summary.txt", "w") as f:
        f.write(f"n feature radiomiche stabili: {len(rad_df.columns)}\n")
        f.write(f"n geni stabili: {len(gene_df.columns)}\n")
        f.write(f"RV osservato: {observed_rv:.4f}\n")
        f.write(f"RV nullo (permutato): {null_rv.mean():.4f} ± {null_rv.std():.4f}\n")
        f.write(f"p-value empirico: {p_value:.4f}\n")
        f.write(f"CI 95% esatta sul p-value: [{ci[0]:.4f}, {ci[1]:.4f}]\n")

    print(f"\nTutti i risultati sono stati salvati in: {out_dir}")