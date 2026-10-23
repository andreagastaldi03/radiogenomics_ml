"""
Rilevamento e gestione del backend GPU (con paracadute automatico su CPU).

Architettura
------------
- XGBoost  -> GPU nativa di xgboost (device="cuda"), nessuna libreria extra.
- Elastic Net / Random Forest / SVM lineare -> RAPIDS cuML (API compatibile
  con scikit-learn). Lo StandardScaler resta quello di scikit-learn su CPU:
  con poche decine di pazienti il costo è nullo, e così `_transform_with_
  pipeline_steps` e SHAP continuano a funzionare identici.
  -- API sta per Application Programming Interface, è il "pannello di controllo" 
     o il punto di contatto con cui un pezzo di software permette a un programmatore 
     di comunicare con lui. Stabilisce quali funzioni puoi chiamare, come si chiamano, 
     quali parametri accettano e cosa restituiscono.

Il paracadute opera su tre livelli:
1. rilevamento: se cuML / CUDA / xgboost-GPU non sono utilizzabili, il modello
   viene costruito sulla sua versione scikit-learn / xgboost-CPU;
2. costruzione: se il costruttore cuML fallisce (versione diversa, parametro
   non supportato), si torna alla CPU per quel modello;
3. esecuzione: se un fit/predict su GPU solleva un'eccezione durante la
   nested CV, l'intero modello viene rieseguito su CPU (vedi
   ml_pipeline.nested_cv_evaluate).

Config letta da config.py (tutte opzionali, con default sensati):
    USE_GPU    = "auto" | "gpu" | "cpu"
    GPU_MODELS = {"elastic_net": True, "random_forest": True, ...}
"""

import functools
import warnings

import numpy as np

import config

ALL_MODELS = ("elastic_net", "random_forest", "svm_linear", "xgboost")

_warned = set()


def _warn_once(msg: str):
    if msg not in _warned:
        _warned.add(msg)
        print(f"[GPU] {msg}")


# ---------------------------------------------------------------------------
# RILEVAMENTO
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1) 
# decoratore della libreria standard di Python (functools) che implementa una 
# cache per i risultati di una funzione, memorizzando solo l'ultimo risultato 
# calcolato. LRU (Least Recently Used), quando la cache raggiunge il limite 
# massimo di spazio e arriva un nuovo risultato, viene cancellato l'elemento 
# a cui non si accede da più tempo.
def cuml_available() -> bool:
    """True se c'è almeno una GPU CUDA visibile e cuML è importabile."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        import cuml
        try:
            # predizioni sempre come numpy: evita sorprese con pandas/cupy a valle
            cuml.set_global_output_type("numpy")
        except Exception:
            pass
        return True
    except Exception:
        return False


def xgb_gpu_kwargs() -> dict:
    """Parametri GPU di XGBoost: sintassi >=2.0 oppure legacy (<2.0)."""
    import xgboost as xgb
    major = int(xgb.__version__.split(".")[0]) # Questa riga estrae il primo numero della versione
    if major >= 2:
        return {"device": "cuda", "tree_method": "hist"}
    return {"tree_method": "gpu_hist"}
# XGBoost ha modificato radicalmente il modo in cui gestisce l'accelerazione hardware 
# a partire dalla versione 2.0


@functools.lru_cache(maxsize=1)
def xgboost_gpu_available() -> bool:
    """Prova reale: un mini-fit su GPU (una build xgboost solo-CPU lo rifiuta)."""
    try:
        import xgboost as xgb
        rng = np.random.RandomState(0)
        X, y = rng.rand(20, 3), np.array([0, 1] * 10)
        # una matrice di 20 righe e 3 colonne di numeri casuali + array con 10 zeri e 10 uni alternati
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb.XGBClassifier(n_estimators=2, **xgb_gpu_kwargs()).fit(X, y)
        return True
    except Exception:
        return False


def is_available(model_name: str) -> bool:
    """La GPU è davvero utilizzabile per questo modello (hardware + librerie)?"""
    if model_name == "xgboost":
        return xgboost_gpu_available()
    return cuml_available()


def resolve_backends(model_names=ALL_MODELS) -> dict:
    """
    Decide per ogni modello se usare la GPU, combinando config.USE_GPU,
    config.GPU_MODELS e la disponibilità reale. Ritorna {modello: bool}.
    """
    mode = str(getattr(config, "USE_GPU", "auto")).lower()
    # cerca l'attr USE_GPU nel modulo config. Se non esiste, usa "auto" come fallback 
    enabled = getattr(config, "GPU_MODELS", None) or {}
    # recupera un dizionario con i consensi per i singoli modelli. 
    out = {}
    for name in model_names:
        if mode == "cpu" or not enabled.get(name, True):
            out[name] = False
        elif is_available(name):
            out[name] = True
        else:
            out[name] = False
            if mode == "gpu":  # richiesta esplicita ma impossibile: avvisa forte
                _warn_once(f"USE_GPU='gpu' ma la GPU non è utilizzabile per '{name}' "
                           f"-> uso la CPU.")
    return out


def print_backend_report(model_names=ALL_MODELS):
    backends = resolve_backends(model_names)
    print("\n" + "=" * 70)
    print(f"BACKEND DI CALCOLO (config.USE_GPU = {getattr(config, 'USE_GPU', 'auto')!r})")
    print("=" * 70)
    print(f"  cuML + CUDA disponibili : {cuml_available()}")
    try:
        import xgboost  # noqa: F401
        print(f"  XGBoost GPU disponibile : {xgboost_gpu_available()}")
    except ImportError:
        print("  XGBoost                 : non installato")
    for name, g in backends.items():
        print(f"  {name:<15} -> {'GPU' if g else 'CPU'}")
    return backends


# ---------------------------------------------------------------------------
# UTILITÀ SUGLI ESTIMATORI
# ---------------------------------------------------------------------------
def to_float32(X):
    """Passo di pipeline per cuML RF: DataFrame/array -> ndarray float32."""
    return np.asarray(X, dtype=np.float32)


def is_cuml_estimator(est) -> bool:
    return type(est).__module__.startswith("cuml")
    # Controlla se un generico oggetto/modello est appartiene alla suite RAPIDS cuML.


def pipeline_uses_gpu(pipe) -> bool:
    """True se il classificatore finale della pipeline gira su GPU."""
    try:
        clf = pipe.named_steps["clf"]
    except (AttributeError, KeyError):
        return False
    if is_cuml_estimator(clf):
        return True
    if type(clf).__module__.startswith("xgboost"):
        p = clf.get_params()
        return p.get("device") in ("cuda", "gpu") or p.get("tree_method") == "gpu_hist"
    return False


def linear_coef_intercept(clf):
    """(coef 1D, intercept scalare) come numpy puri, da sklearn o cuML."""
    coef = np.asarray(clf.coef_, dtype=float).ravel()
    intercept = float(np.asarray(clf.intercept_, dtype=float).ravel()[0])
    return coef, intercept
# Estrae i coefficienti angolari/pesi (coef_) e il termine noto/intercetta (intercept_) 
# da un modello lineare addestrato, restituendoli in un formato NumPy 1D standard