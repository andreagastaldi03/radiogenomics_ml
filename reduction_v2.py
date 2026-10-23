"""
Riduzione feature "neutra" v2 (indipendente dalla label), pensata per budget
di feature molto restrittivi (circa 1 feature ogni 10 pazienti della classe
minoritaria).

NON sostituisce data_utils.py: è un modulo parallelo, così i risultati
ottenuti con la versione precedente restano riproducibili.

Cosa cambia rispetto a data_utils.neutral_feature_reduction
------------------------------------------------------------
1) FILTRO DI DISPERSIONE CORRETTO.
   data_utils.variance_filter standardizza ogni colonna e poi ne calcola la
   varianza: il risultato è 1 per qualunque feature non costante, quindi il
   filtro eliminava solo le feature esattamente costanti e la soglia
   (0.01) non aveva alcun effetto. Qui si usano criteri che hanno davvero un
   potere discriminante e non dipendono dall'unità di misura:
     - near-zero-variance: pochissimi valori distinti o un valore dominante;
     - "iqr_range": IQR / (max - min). Vicino a 0 = feature "piatta" con
       pochi outlier che gonfiano il range. Definito anche per valori
       negativi o nulli (adatto alla radiomica);
     - "robust_cv": IQR / |mediana|. Sensato solo per feature positive
       (es. espressione genica); con mediana 0 è indefinito e la feature
       viene scartata.

2) TAGLIO A NUMERO FISSO DI CLUSTER (k) PER BLOCCO.
   Invece di una soglia di correlazione (che con n=54 non controlla quante
   feature restano, e a 0.25 raggruppa feature di fatto indipendenti), si
   taglia il dendrogramma in esattamente k cluster per blocco
   (texture / forma / geni). Se k=None per un blocco si ricade sulla soglia
   di correlazione (corr_threshold), come nella versione precedente.

3) RAPPRESENTANTE DI CLUSTER CORRETTO (due modalità):
     - "medoid": la feature con la correlazione assoluta media più alta
       verso gli altri membri del cluster. La versione precedente sceglieva
       la varianza grezza massima, che premia la feature con l'unità di
       misura più grande (es. un volume), non la più rappresentativa.
     - "pca": la prima componente principale dei membri del cluster
       (z-score con media/sd del TRAINING). Più stabile e informativa di un
       singolo rappresentante, meno interpretabile; i cluster formati da
       una sola feature restano la feature originale. Il file report
       indica quanta varianza della feature composita è spiegata da PC1:
       valori bassi = cluster eterogeneo, componente poco significativa.

Uso dentro la CV
----------------
BlockReducer è un transformer scikit-learn (fit sul training, transform sul
test): può essere messo in una Pipeline per ridurre le feature DENTRO ogni
fold. Attenzione: ml_pipeline._transform_with_pipeline_steps assume che gli
step prima del classificatore non cambino le colonne, quindi la parte SHAP
va adattata se si usa BlockReducer come step di pipeline.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin

RAD_PREFIX = "rad__"
GEN_PREFIX = "gen__"
_STRIP_PREFIXES = ("rad__", "gen__", "original_")


def is_shape_column(colname: str) -> bool:
    """Stessa regola di data_utils.is_shape_column."""
    return "shape" in colname.lower()


def _short(name: str) -> str:
    for p in _STRIP_PREFIXES:
        name = name.replace(p, "")
    return name


# ---------------------------------------------------------------------------
# FILTRI DI DISPERSIONE
# ---------------------------------------------------------------------------
def near_zero_variance_mask(X: pd.DataFrame, max_dominant_freq: float = 0.9,
                            min_unique: int = 3) -> pd.Series:
    """True = feature da tenere. Scarta feature con meno di min_unique valori
    distinti o con un singolo valore ripetuto in più di max_dominant_freq dei pazienti."""
    keep = {}
    for c in X.columns:
        v = X[c].dropna()
        if v.nunique() < min_unique:
            keep[c] = False
        else:
            keep[c] = v.value_counts(normalize=True).iloc[0] <= max_dominant_freq
            # calcola la frequenza relativa di ogni valore unico presente nella colonna, iloc[0] 
            # estrae la frequenza del valore più comune
    return pd.Series(keep)


def dispersion_scores(X: pd.DataFrame, method: str) -> pd.Series:
    """Punteggio di dispersione per colonna (più alto = più dispersa)."""
    q1, med, q3 = X.quantile(0.25), X.median(), X.quantile(0.75)
    iqr = q3 - q1
    if method == "iqr_range":
        return iqr / (X.max() - X.min()).replace(0, np.nan)
    if method == "robust_cv":
        return iqr / med.abs().replace(0, np.nan)
    if method == "iqr":  # IQR grezzo, come in data_utils.select_genes (dipende dalla scala)
        return iqr
    raise ValueError(f"dispersion method '{method}' non valido: usa 'iqr_range', 'robust_cv' o 'iqr'")


def _top_fraction(scores: pd.Series, pct: float) -> list:
    n = max(1, int(len(scores) * pct))
    return scores.sort_values(ascending=False).head(n).index.tolist()


# ---------------------------------------------------------------------------
# TRANSFORMER
# ---------------------------------------------------------------------------
class BlockReducer(BaseEstimator, TransformerMixin):
    """
    Riduzione per blocchi (radiomica texture, radiomica forma, geni) con
    budget fisso k per blocco. Vedi la docstring del modulo.

    Parametri
    ---------
    k_texture, k_shape, k_gen : numero di feature finali per blocco
        (None = niente taglio a k: si usa corr_threshold per quel blocco).
    mode : "medoid" | "pca"
    corr_threshold : soglia di correlazione per i blocchi con k=None.
    method : "spearman" | "pearson" (correlazione usata per il clustering).
    separate_shape : se True le feature radiomiche di forma sono un blocco a sé.
    rad_dispersion_method / rad_dispersion_threshold : filtro di dispersione
        radiomica (tiene le feature con punteggio >= soglia). None = nessuno.
    nzv_max_dominant_freq : soglia del filtro near-zero-variance (sempre attivo).
    gen_prefilter_method / gen_prefilter_pct : pre-selezione dei geni:
        "iqr_top_pct" (IQR grezzo, come la versione precedente),
        "robust_cv_top_pct" (IQR/|mediana|), oppure "none".
    drop_nan_columns : scarta le feature con NaN (altrimenti errore).
    """

    def __init__(self, k_texture=5, k_shape=1, k_gen=5, mode="medoid",
                 corr_threshold=0.9, method="spearman", separate_shape=True,
                 rad_dispersion_method="iqr_range", rad_dispersion_threshold=0.05,
                 nzv_max_dominant_freq=0.9,
                 gen_prefilter_method="iqr_top_pct", gen_prefilter_pct=0.05,
                 drop_nan_columns=True, verbose=False):
        self.k_texture = k_texture
        self.k_shape = k_shape
        self.k_gen = k_gen
        self.mode = mode
        self.corr_threshold = corr_threshold
        self.method = method
        self.separate_shape = separate_shape
        self.rad_dispersion_method = rad_dispersion_method
        self.rad_dispersion_threshold = rad_dispersion_threshold
        self.nzv_max_dominant_freq = nzv_max_dominant_freq
        self.gen_prefilter_method = gen_prefilter_method
        self.gen_prefilter_pct = gen_prefilter_pct
        self.drop_nan_columns = drop_nan_columns
        self.verbose = verbose

    # -- utilità -----------------------------------------------------------
    def _split_blocks(self, cols):
        rad = [c for c in cols if c.startswith(RAD_PREFIX)]
        gen = [c for c in cols if c.startswith(GEN_PREFIX)]
        if self.separate_shape:
            shape = [c for c in rad if is_shape_column(c)]
            texture = [c for c in rad if not is_shape_column(c)]
        else:
            shape, texture = [], rad
        return {"texture": texture, "shape": shape, "gen": gen}

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # -- fit ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame): # if X is pd.Dataframe
            raise TypeError("BlockReducer richiede un DataFrame con colonne rad__* / gen__*")
        if self.mode not in ("medoid", "pca"):
            raise ValueError("mode deve essere 'medoid' o 'pca'")

        blocks = self._split_blocks(list(X.columns))
        if not any(blocks.values()):
            raise ValueError("[BlockReducer] nessuna colonna rad__ o gen__ trovata in X")

        k_map = {"texture": self.k_texture, "shape": self.k_shape, "gen": self.k_gen}
        prefix_map = {"texture": RAD_PREFIX, "shape": RAD_PREFIX, "gen": GEN_PREFIX}
        tag_map = {"texture": "t", "shape": "s", "gen": "g"}

        self.units_, self.filter_log_ = [], {}
        for block, cols in blocks.items():
            if not cols:
                continue
            Xb = X[cols].astype(float)
            log = {"input": len(cols)}

            # NaN
            nan_cols = Xb.columns[Xb.isna().any()].tolist()
            if nan_cols:
                if not self.drop_nan_columns:
                    raise ValueError(f"[BlockReducer] NaN nelle feature {nan_cols[:5]}...")
                Xb = Xb.drop(columns=nan_cols)
            log["dropped_nan"] = len(nan_cols)

            # near-zero-variance (sempre)
            if Xb.shape[1]:
                keep = near_zero_variance_mask(Xb, self.nzv_max_dominant_freq)
                Xb = Xb.loc[:, keep[keep].index]
            log["after_nzv"] = Xb.shape[1]

            # dispersione (radiomica) o pre-selezione (geni)
            if block in ("texture", "shape"):
                if self.rad_dispersion_method and Xb.shape[1]:
                    sc = dispersion_scores(Xb, self.rad_dispersion_method)
                    Xb = Xb.loc[:, (sc >= self.rad_dispersion_threshold).fillna(False)]
            else:
                if self.gen_prefilter_method == "iqr_top_pct" and Xb.shape[1]:
                    Xb = Xb[_top_fraction(dispersion_scores(Xb, "iqr"), self.gen_prefilter_pct)]
                elif self.gen_prefilter_method == "robust_cv_top_pct" and Xb.shape[1]:
                    sc = dispersion_scores(Xb, "robust_cv").fillna(-np.inf)
                    Xb = Xb[_top_fraction(sc, self.gen_prefilter_pct)]
                elif self.gen_prefilter_method not in ("none", "iqr_top_pct", "robust_cv_top_pct"):
                    raise ValueError(f"gen_prefilter_method '{self.gen_prefilter_method}' non valido")
            log["after_dispersion"] = Xb.shape[1]

            units = self._fit_block(Xb, k_map[block], prefix_map[block], tag_map[block], block)
            log["units"] = len(units)
            self.units_.extend(units)
            self.filter_log_[block] = log
            self._log(f"[BlockReducer] {block:<8} {log['input']} -> nzv {log['after_nzv']} "
                      f"-> dispersione {log['after_dispersion']} -> {log['units']} feature finali "
                      f"(k={k_map[block]}, mode={self.mode})")

        if not self.units_:
            raise ValueError("[BlockReducer] nessuna feature sopravvissuta ai filtri")
        self.feature_names_out_ = [u["name"] for u in self.units_]
        self.report_ = self._build_report()
        self.feature_map_ = self._build_feature_map()
        self._log(f"[BlockReducer] TOTALE: {X.shape[1]} -> {len(self.feature_names_out_)} feature")
        return self

    def _fit_block(self, Xb, k, prefix, tag, block):
        cols = list(Xb.columns)
        if not cols:
            return []
        if len(cols) == 1:
            clusters = np.array([1])
            corr_df = pd.DataFrame([[1.0]], index=cols, columns=cols)
        else:
            corr = Xb.corr(method=self.method).abs().fillna(0.0).to_numpy()
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1.0)
            corr_df = pd.DataFrame(corr, index=cols, columns=cols)
            dist = np.clip(1.0 - corr, 0.0, None)
            np.fill_diagonal(dist, 0.0)
            Z = linkage(squareform(dist, checks=False), method="average")
            # comprime la matrice quadrata nel vettore triangolare superiore 
            # linkage costruisce il dendrogramma
            if k is not None:
                clusters = (np.arange(1, len(cols) + 1) if len(cols) <= k
                            else fcluster(Z, t=int(k), criterion="maxclust"))
            else:
                clusters = fcluster(Z, t=1 - self.corr_threshold, criterion="distance")
                # output clusters è un array NumPy di numeri interi rappresentanti l'ID del cluster

        groups = {}
        for col, cid in zip(cols, clusters):
            groups.setdefault(cid, []).append(col)
            # se chiave cid non esiste -> lista vuota, altrimenti lista associata, appende feature alla chiave
        # ordine deterministico: cluster più grandi per primi
        ordered = sorted(groups.values(), key=lambda m: (-len(m), cols.index(m[0])))
        # len(m) è la cardinalità del gruppo (quante feature contiene il cluster). Il segno meno inverte l'ordine 
        # naturale crescente di sorted. In caso parità secondo criterio: m[0] è la prima feature contenuta nel gruppo, 
        # cols.index(m[0]) restituisce l'indice numerico di quella feature nella lista originale cols.

        units = []
        for j, members in enumerate(ordered, start=1):
            m = len(members)
            if m == 1:
                medoid, within = members[0], np.nan
            else:
                sub = corr_df.loc[members, members]
                # estrae la sottomatrice di correlazione tra le sole variabili del cluster
                score = (sub.sum(axis=1) - 1.0) / (m - 1)
                # somma le correlazioni con le altre escludendo se stessa e divide per il numero delle altre variabili
                medoid, within = score.idxmax(), float(score.mean())

            unit = {"block": block, "members": members, "medoid": medoid,
                    "n_members": m, "within_corr": within, "evr": np.nan}
            if self.mode == "medoid" or m == 1:
                unit.update(kind="keep", name=medoid)
            else:
                M = Xb[members].to_numpy(float)
                mu = M.mean(axis=0)
                sd = M.std(axis=0, ddof=0)
                sd[sd == 0] = 1.0
                Zs = (M - mu) / sd
                # Standardizzazione locale
                _, S, Vt = np.linalg.svd(Zs, full_matrices=False)
                loading = Vt[0]
                # segno fissato: PC1 positivamente correlata con il medoide
                med_z = Zs[:, members.index(medoid)]
                if np.dot(Zs @ loading, med_z) < 0:
                    loading = -loading
                unit.update(kind="pc", mean=mu, std=sd, loading=loading,
                            evr=float(S[0] ** 2 / np.sum(S ** 2)), 
                            # rapporto di varianza spiegata (Explained Variance Ratio)
                            name=f"{prefix}PC_{tag}{j:02d}_n{m}_{_short(medoid)}")
            units.append(unit)
        return units

    # -- report ------------------------------------------------------------
    def _build_report(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "feature_out": u["name"], "block": u["block"], "kind": u["kind"],
            "n_members": u["n_members"], "medoid": u["medoid"],
            "mean_abs_corr_within_cluster": u["within_corr"],
            "pc1_explained_variance_ratio": u["evr"],
            "members": ";".join(u["members"]),
        } for u in self.units_])

    def _build_feature_map(self) -> pd.DataFrame:
        rows = []
        for u in self.units_:
            for i, mem in enumerate(u["members"]):
                rows.append({"feature_in": mem, "feature_out": u["name"], "block": u["block"],
                             "is_medoid": mem == u["medoid"],
                             "pc1_loading": float(u["loading"][i]) if u["kind"] == "pc" else np.nan})
        return pd.DataFrame(rows)

    # -- transform ---------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "units_"):
            raise RuntimeError("BlockReducer non è ancora stato fittato")
        out = {}
        for u in self.units_:
            if u["kind"] == "keep":
                out[u["name"]] = X[u["members"][0]].to_numpy(float)
            else:
                Zs = (X[u["members"]].to_numpy(float) - u["mean"]) / u["std"]
                out[u["name"]] = Zs @ u["loading"]
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


def reduce_features_v2(X: pd.DataFrame, print_info: bool = True, **params):
    """
    Equivalente di data_utils.neutral_feature_reduction per la v2: fit e
    transform sull'intero dataset (va bene: non usa la label). Ritorna
    (X_reduced, reducer) — reducer.report_ e reducer.feature_map_ dicono
    quali feature originali stanno dietro a ciascuna feature finale.
    """
    reducer = BlockReducer(verbose=print_info, **params)
    X_reduced = reducer.fit(X).transform(X)
    return X_reduced, reducer