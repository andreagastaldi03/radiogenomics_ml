"""
Pipeline di modellazione supervisionata con nested cross-validation.

Perché nested CV: con n=54 una singola divisione train/test dà stime di
performance molto instabili. La CV esterna stima la performance "onesta",
quella interna sceglie iperparametri/feature SENZA mai vedere il fold esterno
di test, evitando l'ottimismo da leakage.

Il modulo produce anche:
- un ranking di importanza delle feature aggregato su tutti i fold (stability)
- valori SHAP per l'interpretazione locale/globale del modello migliore
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non interattivo, necessario per salvare plot da script
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)

import config
import gpu_utils

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] Libreria 'shap' non trovata: pip install shap per l'analisi di interpretabilità")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] Libreria 'xgboost' non trovata: pip install xgboost per l'analisi di interpretabilità")

# ---------------------------------------------------------------------------
# DEFINIZIONE MODELLI E GRIGLIE DI IPERPARAMETRI
# ---------------------------------------------------------------------------
def _wants_gpu(use_gpu, name: str) -> bool:
    """use_gpu può essere bool (tutti i modelli) o dict {modello: bool}.
    Anche se richiesta, la GPU viene usata solo se realmente disponibile."""
    wanted = use_gpu.get(name, False) if isinstance(use_gpu, dict) else bool(use_gpu)
    # isinstance(object, classinfo) -> mi dice se object è un oggetto classinfo (return True or False)
    return bool(wanted) and gpu_utils.is_available(name)
# underscore (_) indica una funzione privata o interna


def get_model_grid(use_gpu=False):
    """
    Restituisce un dizionario {nome_modello: (pipeline, param_grid)}.

    use_gpu : bool oppure dict {modello: bool}. DEFAULT False (= comportamento
        originale su CPU): così qualunque altro modulo che chiama
        get_model_grid()/build_pipeline_from_best_params() resta invariato.
        La GPU si attiva solo passando esplicitamente use_gpu
        (run_all_models lo fa usando gpu_utils.resolve_backends()).
        Se la GPU richiesta non è disponibile, o il costruttore cuML fallisce,
        il modello viene costruito su CPU (paracadute).

    Elastic Net è il modello principale consigliato: fa selezione automatica
    delle feature (L1) gestendo al contempo la collinearità (L2), ed è
    interpretabile tramite i coefficienti.

    Random Forest e SVM lineare sono inclusi come confronto / cross-check
    dell'importanza delle feature (via feature_importances_ o SHAP).

    NB: i nomi dei parametri della griglia (clf__C, clf__l1_ratio, ...) sono
    identici tra CPU e GPU, quindi i best_params trovati su un backend sono
    riutilizzabili sull'altro.
    """
    models = {}

    # --- Elastic Net (logistic regression con penalità elasticnet) ---
    en_pipe = None
    if _wants_gpu(use_gpu, "elastic_net"):
        try:
            from cuml.linear_model import LogisticRegression as cuLogReg
            # cuML: solver unico (qn / OWL-QN), nessun random_state
            en_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", cuLogReg(penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                 max_iter=5000, tol=1e-4)),
            ])
        except Exception as exc:
            gpu_utils._warn_once(f"elastic_net: costruzione cuML fallita ({exc!r}) -> CPU")
    if en_pipe is None:
        en_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=5000,
                l1_ratio=0.5,
                random_state=config.RANDOM_STATE
            )),
        ])
    # elastic net contiene both L1 e L2, C e l1 cercano un bilancio tra le due
    en_grid = {
        "clf__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        "clf__l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
    }
    models["elastic_net"] = (en_pipe, en_grid)

    # oggetto pipeline permette di definire in un unico oggetto i vari passaggi necessari all'
    # addestramento del modello. permette di segnare in ordine i vari step, chiamandoli sequenzialmente
    # durante il processo di fit e di predict (oggetti devono implementare fit e transform method). 
    # definisco l'ordine degli step, prima nella fase di fit (en_pipe.fit(X_train, y_train)) applico
    # riscalamento (= scaler.fit_transform(X_train)) poi addestro il modello 
    # (= clf.fit(X_train_scaled, y_train)). ugualmente quando predico (en_pipe.predict(X_test)) esegue
    # scaler.transform(X_test) scalando il dataset coi parametri imparati prima e poi restituisce le
    # predizioni finali (= clf.predict(X_test_scalato))
    # inoltre io salvo un oggetto (pipeline, griglia), struttura di montaggio e insieme di test che 
    # vogliamo applicare. la griglia ha una struttura clf__parametro, perchè potrei aver parametri 
    # associati a diversi step, e quindi il programma deve sapere dove quel parametro rientra. significa
    # prendi clf e imposta il suo parametro ai valori segnati.
    
    # --- Random Forest ---
    rf_pipe, rf_grid = None, None
    if _wants_gpu(use_gpu, "random_forest"):
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
            # n_streams=1 -> risultati riproducibili a parità di seed.
            # stream sequenza di operazioni e calcoli che vengono eseguiti in ordine sulla scheda.
            # cuML richiede max_depth intero: None (illimitata) -> 16 (il default
            # cuML); con n=54 gli alberi non arrivano comunque a quella profondità.
            rf_pipe = Pipeline([
                ("to_float32", FunctionTransformer(gpu_utils.to_float32)),
                ("clf", cuRF(random_state=config.RANDOM_STATE, n_streams=1, n_bins=32)),
            ])
            rf_grid = {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [3, 5, 16],
                "clf__min_samples_leaf": [1, 3, 5],
            }
        except Exception as exc:
            gpu_utils._warn_once(f"random_forest: costruzione cuML fallita ({exc!r}) -> CPU")
            rf_pipe, rf_grid = None, None
    if rf_pipe is None:
        rf_pipe = Pipeline([
            ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE)),
        ])
        rf_grid = {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [3, 5, None],
            "clf__min_samples_leaf": [1, 3, 5],
        }
    models["random_forest"] = (rf_pipe, rf_grid)

    # --- SVM lineare ---
    svm_pipe = None
    if _wants_gpu(use_gpu, "svm_linear"):
        try:
            from cuml.svm import SVC as cuSVC
            svm_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    # convertire distanza geometrica in probabilità calibrata. 
                    # CalibratedClassifierCV prende i margini continui dell'SVM 
                    # e ci adatta sopra una sigmoide per mapparli nell'intervallo [0, 1].
                    cuSVC(kernel="linear", random_state=config.RANDOM_STATE),
                    ensemble=False
                    # Di default, CalibratedClassifierCV esegue una Cross-Validation interna a 5 fold
                    # SVM singola in un piccolo ensemble, impostando ensemble=False il modello finale
                    # conservato è uno solo, addestrato sull'intero dataset
                ))
            ])
        except Exception as exc:
            gpu_utils._warn_once(f"svm_linear: costruzione cuML fallita ({exc!r}) -> CPU")
    if svm_pipe is None:
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(kernel="linear", random_state=config.RANDOM_STATE),
                ensemble=False
            ))
        ])
    svm_grid = {
        "clf__estimator__C": [0.001, 0.01, 0.1, 1, 10],
    }
    models["svm_linear"] = (svm_pipe, svm_grid)

    # --- XGBoost (opzionale) ---
    if XGB_AVAILABLE:
        xgb_extra = gpu_utils.xgb_gpu_kwargs() if _wants_gpu(use_gpu, "xgboost") else {}
        xgb_pipe = Pipeline([
            ("clf", XGBClassifier(
                eval_metric="logloss", random_state=config.RANDOM_STATE,
                **xgb_extra
            )),
        ])
        xgb_grid = {
            "clf__n_estimators": [100, 300], # numero di alberi
            "clf__max_depth": [2, 3, 4], # profondità alberi
            "clf__learning_rate": [0.01, 0.05, 0.1], # tasso approfondimento gradient descend 
        }
        models["xgboost"] = (xgb_pipe, xgb_grid)

    return models

# eXtreme Gradient Boosting, costruisce alberi decisionali in sequenza (non in parallelo come random
# forest, dove poi viene presa la media o maggioranza delle risposte). ogni albero decisionale non parte
# da zero ma riparte dal correggere gli errori commessi dal precedente. utilizza gradiente per capire in
# che direzione correggere gli errori del precedente. è un algoritmo che su pochi dati tende ad andare 
# in overfitting molto facilmente, quindi modelli più regolarizzati spesso funzionano meglio. 
# un albero viene creato nello spazio delle feature, facendo un taglio in questo spazio n-dim in modo da
# separare i valori delle feature così che vi siano solo pazienti della classe 1 o 0.
# un xgboost invece di aggiornare i parametri interni di un vecchio albero, XGBoost aggiunge un nuovo 
# albero intero per correggere l'errore residuo del modello precedente. la logica del grad descend non 
# si applica a parametri continui del modello, ma corregge l'output con un albero nuovo. ogni nuovo 
# albero prova a prevedere l'errore residuo fatto dalla somma di tutti gli alberi costruiti fino a quel
# momento. il nuovo albero prevede questo errore, quindi se sommato agli alberi precedenti dovrebbe 
# aggiustare la predizione (scalato con un parametro learning rate).
# l'idea è che questi alberi creati in sequenza siano indipendenti e congelati una volta determinati. 
# non assegnano una classe o una probabilità, ma un numero continuo chiamato peso ("w"). se l'albero 
# zero associa 0.5 a tutti, ci sarà un errore di + o - 0.5 dipendente dalla classe (0 o 1). quindi i 
# pazienti entrano nell'albero 1, qui vengono divisi nei diversi nodi in base a tagli rispetto a 
# particolari feature, e nel nodo finale arrivano a un peso "w" che si somma allo 0.5 iniziale. il valore
# del peso viene calcolato scrivendo la loss = g (der prima) * w + 0.5 * h * w^2 (taylor secondo ordine)
# e facendone il minimo rispetto a w, quindi g^2/h (più eventuali termini di correzione-penalità). 
# grazie a un w ottenuto così, riusciamo a verificare se dal nodo padre dividendolo in due nodi dx e sx 
# si migliora lo score finale del paziente, e questo è il gain. ogni albero i pazienti avranno un aumento
# o riduzione del loro score a partire dai diversi alberi in successione. alla fine, si applica un 
# softmax per trasformare il punteggio in probabilità e attribuire una classe.

# Mappa usata per scegliere automaticamente l'explainer SHAP giusto in base
# al modello risultato migliore (per AUC) nella nested CV, invece di forzare
# sempre lo stesso modello.
MODEL_TYPE_MAP = {
    "elastic_net": "linear",
    "svm_linear": "linear",
    "random_forest": "tree",
    "xgboost": "tree",
}

def majority_vote_params(best_params_list):
    """
    Determina la combinazione di iperparametri più frequentemente scelta
    come "migliore" tra i fold esterni della nested CV.

    Perché: la nested CV sceglie iperparametri potenzialmente diversi ad ogni
    fold esterno (normale, con n=54 il tuning è instabile). Per la stability
    selection bootstrap serve un solo set di iperparametri fissi (rifare un
    grid search dentro ognuno dei centinaia di bootstrap sarebbe troppo
    costoso e statisticamente ridondante). Il voto di maggioranza è un modo
    semplice e trasparente per sceglierli in modo data-driven, non arbitrario.
    """
    param_tuples = [tuple(sorted(p.items())) for p in best_params_list]
    most_common, count = Counter(param_tuples).most_common(1)[0]
    print(f"\n[majority_vote_params] combinazione più frequente: {dict(most_common)} "
          f"(scelta in {count}/{len(best_params_list)} fold)")
    return dict(most_common)
        # ogni nested cv testa gli iperparam, ma la cv esterna produce diversi possibili valori degli 
        # iperparam (uno per ogni fold). trasforma queste coppie iperparam-valore in tupla. poi viene
        # contato quante volte, sul numero di fold, questi iperparam sono scelti. prendo il più frequente 
        # e lo trasformo in dictionary di nuovo. voto di maggioranza per evitare fluttuazioni del valore
        # per scelta di fold diversi
        
        
def build_pipeline_from_best_params(model_name: str, best_params: dict, use_gpu=False):
    """
    Ricostruisce (non fittata) la pipeline di un modello con un set di
    iperparametri fissato, qualunque sia il tipo di modello — usata per
    rieseguire lo stesso modello "migliore" fuori dalla nested CV (test di
    permutazione, batch effect), senza doverlo hardcodare a elastic_net.

    use_gpu=False di default (CPU): i chiamanti che lanciano molte istanze in
    processi paralleli (permutazioni, spec curve) non devono condividere la GPU.
    """
    models = get_model_grid(use_gpu=use_gpu)
    if model_name not in models:
        raise ValueError(f"Modello '{model_name}' non trovato. Disponibili: {list(models.keys())}")
    pipe, _ = models[model_name]
    pipe = clone(pipe)
    pipe.set_params(**best_params)
    return pipe

# ---------------------------------------------------------------------------
# NESTED CROSS VALIDATION
# ---------------------------------------------------------------------------
def nested_cv_evaluate(X: pd.DataFrame, y: pd.Series, model_name: str,
                        pipe, grid, n_outer=config.N_OUTER_FOLDS,
                        n_inner=config.N_INNER_FOLDS, random_state=config.RANDOM_STATE):
    """
    Wrapper con paracadute: se la pipeline gira su GPU e qualunque fit/predict
    solleva un'eccezione, l'INTERO modello viene rieseguito da capo su CPU
    (così i fold non sono mai un mix di backend).
    """
    if gpu_utils.pipeline_uses_gpu(pipe):
        try:
            # GPU: GridSearchCV in serie (n_jobs=1) perché più processi non
            # devono contendersi lo stesso contesto CUDA; error_score="raise"
            # per far emergere gli errori invece di trasformarli in NaN.
            res = _nested_cv_core(X, y, model_name, pipe, grid, n_outer, n_inner,
                                  random_state, search_n_jobs=1, error_score="raise")
            # Lanciare n_jobs > 1 con modelli GPU creerebbe più processi Python che 
            # tentano di comunicare simultaneamente con la scheda video, causando gravi problemi
            res["backend"] = "gpu"
            return res
        except Exception as exc:
            print(f"\n[GPU] {model_name}: errore su GPU ({exc!r}).\n"
                  f"[GPU] -> rieseguo l'intero modello su CPU.")
            pipe, grid = get_model_grid(use_gpu=False)[model_name]
    res = _nested_cv_core(X, y, model_name, pipe, grid, n_outer, n_inner,
                          random_state, search_n_jobs=-1, error_score=np.nan)
    res["backend"] = "cpu"
    return res


def _nested_cv_core(X: pd.DataFrame, y: pd.Series, model_name: str,
                    pipe, grid, n_outer, n_inner, random_state,
                    search_n_jobs=-1, error_score=np.nan):
    """
    Esegue nested CV per un singolo modello e ritorna metriche per ciascun
    fold esterno + i migliori iperparametri scelti ad ogni fold.

    Ritorna un dict con liste di metriche (una entry per fold esterno) e la
    lista dei modelli fittati (utile per estrarre feature importance/SHAP).
    """
    y_bin = (y == config.POSITIVE_CLASS).astype(int) # se y uguale alla classe positiva 1, altrimenti 0

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state) # mantiene la 
        # stessa proporzione di casi 1 e 0, in modo che non ci siano fold senza esempi di una classe
        # questo Cross Validation è per valutazione prestazioni modello

    results = {
        "auc": [], "balanced_accuracy": [], "f1": [],
        "best_params": [], "confusion_matrices": [],
        "fitted_models": [], "test_indices": [],
        "model_name": model_name,
    }

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_bin)):
            # split genera indici di posizione per train e test set, enumerate aggiunge un counter per 
            # tenere conto dell'iterazione che stiamo considerando
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state) # creo un 
            # Cross Validation sul train set per tuning degli iperparametri
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv,
                              n_jobs=search_n_jobs, error_score=error_score) # testa tutte le 
            # combinazioni di iperparametri definite in "grid" usando come metrica ROC-AUC
        search.fit(X_train, y_train) # esegue effettivamente la ricerca, allena il modello sulla 
            # combinazione di iperparametri. una volta trovata quella vincente, riaddestra la pipeline 
            # con quei valori

        best_model = search.best_estimator_ # prendo il modello migliore
        y_pred = best_model.predict(X_test) # lo alleno sul test set, estrae la classe (non la prob)
        y_proba = best_model.predict_proba(X_test)[:, 1] # ritorna la probabilità continua, assegnata 
            # alla classe positiva. il metodo restituisce una matrice a due colonne, prob per ogni classe
            # quindi seleziono solo colonna con prob per classe 1

        results["auc"].append(roc_auc_score(y_test, y_proba)) # curva roc misura capacità di separare le
            # due classi a qualsiasi soglia di prob. algoritmo prende lista delle prob, varia la soglia
            # di classificazione e confronta predizioni con label y test. a ogni soglia calcola la coppia
            # (1 - specificità, sensibilità) e disegna un punto sul grafico. auc area sotto la curva -> 1
        results["balanced_accuracy"].append(balanced_accuracy_score(y_test, y_pred)) # accuracy è 
            # (TP+TN)/(TP+TN+FP+FN). balanced è invece media tra sensibilità e specificità, quindi 
            # = (sensib + specif) / 2 = 1/2 ( tp/(tp+fn) + tn/(tn+fp) )
        results["f1"].append(f1_score(y_test, y_pred)) # media armonica tra precision and recall
        results["best_params"].append(search.best_params_)
        results["confusion_matrices"].append(confusion_matrix(y_test, y_pred))
        results["fitted_models"].append(best_model)
        results["test_indices"].append(test_idx)

        print(f"[{model_name}] fold {fold_i+1}/{n_outer} | "
              f"AUC={results['auc'][-1]:.3f} | "
              f"BalAcc={results['balanced_accuracy'][-1]:.3f} | "
              f"best_params={search.best_params_}")

    print(f"\n[{model_name}] RISULTATI NESTED CV (media ± sd su {n_outer} fold esterni)")
    for metric in ["auc", "balanced_accuracy", "f1"]:
        vals = results[metric]
        print(f"  {metric}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    return results


def run_all_models(X: pd.DataFrame, y: pd.Series):
    """
    Esegue nested CV per tutti i modelli definiti in get_model_grid()
    e ritorna un dizionario {model_name: results}.
    """
    backends = gpu_utils.resolve_backends(gpu_utils.ALL_MODELS)
    models = get_model_grid(use_gpu=backends)
    all_results = {}
    for name, (pipe, grid) in models.items():
        print(f"\n{'='*70}\nModello: {name} "
              f"[{'GPU' if gpu_utils.pipeline_uses_gpu(pipe) else 'CPU'}]\n{'='*70}")
        all_results[name] = nested_cv_evaluate(X, y, name, pipe, grid)
    return all_results

# ---------------------------------------------------------------------------
# POOLED OUT-OF-FOLD METRICS
# ---------------------------------------------------------------------------
def compute_pooled_oof_metrics(results: dict, X: pd.DataFrame, y: pd.Series):
    """
    Calcola le metriche pooled out-of-fold usando le stesse predizioni
    della nested CV esterna. Non rifitta i modelli e non modifica gli split.

    Per ogni outer fold:
        - recupera il modello già addestrato;
        - recupera i pazienti del test fold;
        - calcola probabilità e classe predetta.

    Le predizioni vengono poi ricomposte per tutti i pazienti e
    utilizzate per calcolare:
        - pooled ROC-AUC
        - pooled balanced accuracy
        - pooled F1

    Returns
    -------
    metrics : dict
        Dizionario con le tre metriche pooled.

    oof_df : pd.DataFrame
        Predizioni OOF paziente-per-paziente.
    """

    # --------------------------------------------------------------
    # Target binario
    # --------------------------------------------------------------
    y_bin = (y == config.POSITIVE_CLASS).astype(int)
    n_samples = len(X)
    # Probabilità predette
    oof_probability = np.full(n_samples, np.nan, dtype=float)
        # Return a new array of given shape and type, filled with fill_value.
    # Classe predetta
    oof_prediction = np.full(n_samples, -1, dtype=int)
    # Outer fold di appartenenza
    oof_fold = np.full(n_samples, -1, dtype=int)

    # --------------------------------------------------------------
    # Recupero delle predizioni OOF
    # --------------------------------------------------------------
    for fold_i, (fold_model, test_idx) in enumerate(zip(results["fitted_models"], 
                                            results["test_indices"]), start=1):
            # zip prende i due dict e crea un terzo dict in cui il primo elemento è la coppia  
            # di primi elementi dei due dict passati come oggetto, enumerate aggiunge un 
            # counter che inizia da 1 e non da zero
        test_idx = np.asarray(test_idx, dtype=int)
        X_test = X.iloc[test_idx]
        # Probabilità della classe positiva
        y_proba = fold_model.predict_proba(X_test)[:, 1]
        # Classe predetta dal modello.
        y_pred = fold_model.predict(X_test)
        # Convertiamo eventualmente le classi originali
        # nella codifica binaria 0/1.
        y_pred_bin = y_pred.astype(int)

        # ----------------------------------------------------------
        # Controllo: ogni paziente deve comparire una sola volta
        # come test nella CV esterna.
        # ----------------------------------------------------------
        if np.any(~np.isnan(oof_probability[test_idx])):
            raise RuntimeError(
                f"Alcuni pazienti compaiono in più di un outer test fold (fold {fold_i})."
            )
            # prende oof_prob e lo valuta in tutti gli indici di test, ~np.isnan restituisce 
            # true solo se NOT a NaN, any restituisce true solo se c'è almeno un true.
            # al primo giro/fold tutto funziona, poi se però ci sono degli indici che erano già stati
            # riempiti ai giri precedenti che ritornano in fold successivi scatta il meccanismo
            # di controllo 

        # ----------------------------------------------------------
        # Salvataggio delle predizioni
        # ----------------------------------------------------------
        oof_probability[test_idx] = y_proba
        oof_prediction[test_idx] = y_pred_bin
        oof_fold[test_idx] = fold_i

    # --------------------------------------------------------------
    # Controllo finale
    # --------------------------------------------------------------
    missing_mask = np.isnan(oof_probability) # true se presenti NaN

    if missing_mask.any(): # true se almeno un true
        missing_patients = X.index[missing_mask].tolist()
        raise RuntimeError(
            "Mancano predizioni OOF per alcuni pazienti: "
            f"{missing_patients}"
        )

    # Controllo anche sulle classi predette
    if np.any(oof_prediction < 0):
        raise RuntimeError(
            "Mancano alcune predizioni di classe OOF."
        )

    # --------------------------------------------------------------
    # POOLED METRICS
    # --------------------------------------------------------------
    pooled_auc = roc_auc_score(y_bin, oof_probability)
    pooled_balanced_accuracy = (balanced_accuracy_score(y_bin, oof_prediction))
    pooled_f1 = f1_score(y_bin, oof_prediction, zero_division=0)

    # --------------------------------------------------------------
    # Tabella paziente-per-paziente
    # --------------------------------------------------------------
    oof_df = pd.DataFrame({
        "patient": X.index,
        "y_true": y_bin.to_numpy(),
        "oof_probability": oof_probability,
        "oof_prediction": oof_prediction,
        "outer_test_fold": oof_fold,
    })

    metrics = {
        "pooled_oof_auc": pooled_auc,
        "pooled_oof_balanced_accuracy": (pooled_balanced_accuracy),
        "pooled_oof_f1": pooled_f1,
    }

    return metrics, oof_df


def bootstrap_pooled_auc_ci(y_true: np.ndarray, proba: np.ndarray,
                             n_boot: int = None, random_state: int = None):
    """
    CI bootstrap (percentile, 95%) sulla AUC pooled out-of-fold.
 
    Ricampiona con reinserimento i pazienti dalle predizioni OOF già
    salvate (che sono già state calcolate da compute_pooled_oof_metrics). 
    Risponde alla domanda "quanto è incerta questa stima puntuale di 
    AUC pooled, con n=54?" — un singolo numero nasconde quest'incertezza.
 
    Parametri
    ---------
    y_true, proba : array 1D, stessa lunghezza, allineati per paziente
        (le colonne "y_true" e "oof_probability" di oof_df).
    n_boot : numero di ricampionamenti bootstrap (default: config.N_BOOTSTRAP_AUC_CI)
    random_state : seed (default: config.RANDOM_STATE)
 
    Ritorna
    -------
    observed_auc : AUC pooled sui dati osservati (dovrebbe coincidere con
        quella già calcolata da compute_pooled_oof_metrics)
    ci_low, ci_high : CI 95% percentile della AUC pooled
    boot_aucs : tutte le AUC bootstrap, per il plot/salvataggio
    """
    n_boot = n_boot if n_boot is not None else config.N_BOOTSTRAP_AUC_CI
    random_state = config.RANDOM_STATE if random_state is None else random_state
 
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    observed_auc = roc_auc_score(y_true, proba)
 
    rng = np.random.RandomState(random_state)
    boot_aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        # con classi sbilanciate un ricampionamento può capitare tutto in
        # una classe sola: l'AUC non è definita, si ricampiona finché non serve
        while len(np.unique(y_b)) < 2:
            idx = rng.randint(0, n, size=n)
            y_b = y_true[idx]
        boot_aucs[i] = roc_auc_score(y_b, proba[idx])
 
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    return observed_auc, ci_low, ci_high, boot_aucs
 
    
def plot_auc_bootstrap_ci(boot_aucs: np.ndarray, observed_auc: float, ci_low: float,
                           ci_high: float, model_name: str, output_path):
    """
    Istogramma della distribuzione bootstrap della AUC pooled, con CI 95% 
    e riferimento al caso (0.5).
    """
    plt.figure(figsize=(7, 5))
    plt.hist(boot_aucs, bins=40, color="#8C8C8C", edgecolor="white")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="AUC = 0.5 (caso)")
    plt.axvline(observed_auc, color="#4C72B0", linewidth=2,
                label=f"AUC pooled osservata = {observed_auc:.3f}")
    plt.axvline(ci_low, color="#C44E52", linestyle=":", linewidth=1.5,
                label=f"CI 95% [{ci_low:.3f}, {ci_high:.3f}]")
    plt.axvline(ci_high, color="#C44E52", linestyle=":", linewidth=1.5)
    plt.xlabel("AUC pooled (bootstrap sui pazienti)")
    plt.ylabel("Numero di bootstrap")
    plt.title(f"Incertezza della AUC pooled — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_auc_bootstrap_ci] salvato in {output_path}")

    
def run_pooled_oof_analysis(all_results: dict, X: pd.DataFrame, y: pd.Series, output_dir=None):
    """
    Calcola le metriche pooled OOF per tutti i modelli.

    Per ogni modello salva:
        <model>_oof_predictions.csv

    e produce:
        pooled_oof_model_comparison.csv
    """

    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print("\n" + "=" * 70)
    print("POOLED OUT-OF-FOLD METRICS")
    print("=" * 70)

    for model_name, res in all_results.items():
        metrics, oof_df = compute_pooled_oof_metrics(res, X, y)

        # ----------------------------------------------------------
        # Salvataggio delle predizioni individuali
        # ----------------------------------------------------------
        oof_df.to_csv(
            output_dir
            / f"{model_name}_oof_predictions.csv",
            index=False)
        
        # ----------------------------------------------------------
        # CI bootstrap sulla AUC pooled (riusa le predizioni OOF appena
        # calcolate, nessun nuovo fit)
        # ----------------------------------------------------------
        auc_observed, auc_ci_low, auc_ci_high, boot_aucs = bootstrap_pooled_auc_ci(
            oof_df["y_true"].to_numpy(), oof_df["oof_probability"].to_numpy()) 
        pd.Series(boot_aucs, name="pooled_auc_bootstrap").to_csv(
            output_dir / f"{model_name}_pooled_auc_bootstrap.csv", index=False)
        plot_auc_bootstrap_ci(
            boot_aucs, auc_observed, auc_ci_low, auc_ci_high, model_name,
            output_dir / f"{model_name}_pooled_auc_bootstrap.png")

        # ----------------------------------------------------------
        # Risultati fold-level
        # ----------------------------------------------------------
        auc_mean = np.mean(res["auc"])
        auc_sd = np.std(res["auc"])
        balacc_mean = np.mean(res["balanced_accuracy"])
        balacc_sd = np.std(res["balanced_accuracy"])
        f1_mean = np.mean(res["f1"])
        f1_sd = np.std(res["f1"])

        # ----------------------------------------------------------
        # Riga comparativa
        # ----------------------------------------------------------
        rows.append({
            "model": model_name,
            # AUC
            "auc_mean_fold": auc_mean,
            "auc_sd_fold": auc_sd,
            "pooled_oof_auc": (metrics["pooled_oof_auc"]),
            # Balanced accuracy
            "balanced_accuracy_mean_fold": (balacc_mean),
            "balanced_accuracy_sd_fold": (balacc_sd),
            "pooled_oof_balanced_accuracy": 
                        (metrics["pooled_oof_balanced_accuracy"]),
            # F1
            "f1_mean_fold": f1_mean,
            "f1_sd_fold": f1_sd,
            "pooled_oof_f1": (metrics["pooled_oof_f1"]),
        })

        # ----------------------------------------------------------
        # Stampa
        # ----------------------------------------------------------
        print(f"\n[{model_name}]")
        print(f"  AUC: "
            f"{auc_mean:.3f} ± {auc_sd:.3f}"
            f" | pooled = "
            f"{metrics['pooled_oof_auc']:.3f}")
        print(f"  BalAcc: "
            f"{balacc_mean:.3f} ± {balacc_sd:.3f}"
            f" | pooled = "
            f"{metrics['pooled_oof_balanced_accuracy']:.3f}")
        print(f"  F1: "
            f"{f1_mean:.3f} ± {f1_sd:.3f}"
            f" | pooled = "
            f"{metrics['pooled_oof_f1']:.3f}")

    # --------------------------------------------------------------
    # Tabella finale
    # --------------------------------------------------------------
    pooled_summary = (pd.DataFrame(rows).sort_values("pooled_oof_auc", ascending=False)
                      .reset_index(drop=True))
    pooled_summary.to_csv(
        output_dir
        / "pooled_oof_model_comparison.csv",
        index=False,)

    return pooled_summary

# ---------------------------------------------------------------------------
# STABILITY SELECTION (bootstrap) — quali feature emergono ripetutamente
# ---------------------------------------------------------------------------
def _bootstrap_importance(pipe, X_boot, y_boot, model_type) -> pd.Series:
    """
    Importanza delle feature per un bootstrap, in scala comparabile tra modelli:
    valore assoluto del coefficiente per i lineari, feature_importances_ per gli alberi.
    """
    pipe.fit(X_boot, y_boot)
    clf = pipe.named_steps["clf"]
    if model_type == "linear":
        coef, _ = gpu_utils.linear_coef_intercept(clf)
        return pd.Series(np.abs(coef), index=X_boot.columns)
    elif model_type == "tree":
        if not hasattr(clf, "feature_importances_"):
            raise RuntimeError("Questo stimatore non espone feature_importances_ "
                               "(es. RF cuML): usa la versione CPU per la stability selection.")
        return pd.Series(clf.feature_importances_, index=X_boot.columns)
    else:
        raise ValueError(f"model_type '{model_type}' non gestito in _bootstrap_importance")

def bootstrap_stability_selection(X: pd.DataFrame, y: pd.Series,
                                   model_name: str, best_params: dict,
                                   n_bootstrap: int = config.N_BOOTSTRAP,
                                   top_k_features: int = None,
                                   random_state: int = config.RANDOM_STATE,
                                   use_gpu=False):
    """    
    Stability selection generalizzata a qualunque modello di MODEL_TYPE_MAP.
    "Selezionata" = tra le top_k_features per importanza in quel bootstrap
    (default: sqrt(n_feature), euristica comune quando non c'è un criterio
    di sparsità nativo come per l'Elastic Net).

    use_gpu : default False. 200 fit sequenziali su ~50 pazienti sono dominati
    dal costo di lancio dei kernel: la CPU è quasi sempre più veloce. La Random
    Forest cuML non espone feature_importances_, quindi per lei resta sempre CPU.

    NB il criterio qui è "top-K", non "diverso da zero".
    Con L1 forte i due criteri spesso coincidono nella pratica, ma non sono
    matematicamente la stessa cosa.
    """
    model_type = MODEL_TYPE_MAP.get(model_name)
    if model_type is None:
        raise ValueError(f"Stability selection non supportata per '{model_name}': "
                          f"aggiungilo a MODEL_TYPE_MAP ('linear' o 'tree').")

    if model_type == "tree" and model_name == "random_forest":
        use_gpu = False  # cuML RF: niente feature_importances_
    top_k_features = top_k_features or max(5, int(np.sqrt(X.shape[1])))
    y_bin = (y == config.POSITIVE_CLASS).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    rng = np.random.RandomState(random_state)
    selection_counts = pd.Series(0.0, index=X.columns)

    for b in range(n_bootstrap):
        boot_idx = rng.choice(len(X), size=len(X), replace=True) # creo un campion bootstrap, dimensione
            # di X, sorteggiato in modo casuale, possibilità di reinserimento, quindi più copie di un 
            # paziente e altri completamente assenti
        X_boot, y_boot = X.iloc[boot_idx], y_bin.iloc[boot_idx]
        
        if y_boot.nunique() < 2: # controlla ci siano entrambe le classi 
            continue  # bootstrap degenere, salta

        pipe = build_pipeline_from_best_params(model_name, best_params, use_gpu=use_gpu)
        importance = _bootstrap_importance(pipe, X_boot, y_boot, model_type)

        top_feats = importance.sort_values(ascending=False).head(top_k_features).index
        selection_counts[top_feats] += 1

        if (b + 1) % 50 == 0:
            print(f"[stability_selection] {b+1}/{n_bootstrap} bootstrap completati")

    stability_freq = selection_counts / n_bootstrap
    stable_features = stability_freq[
        stability_freq >= config.STABILITY_SELECTION_THRESHOLD
    ].sort_values(ascending=False)

    print(f"\n[stability_selection] modello={model_name} | top_k={top_k_features} | "
          f"{len(stable_features)} feature stabili (>= "
          f"{config.STABILITY_SELECTION_THRESHOLD*100:.0f}% dei bootstrap)")

    return stability_freq, stable_features

# ---------------------------------------------------------------------------
# SHAP — interpretabilità del modello 
# ---------------------------------------------------------------------------
def _transform_with_pipeline_steps(fitted_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Applica tutti gli step della pipeline tranne l'ultimo (il classificatore)."""
    X_transformed = X.copy()
    for step_name, step in fitted_pipeline.steps[:-1]: # applica tutte le trasformazioni della pipeline a
            # X tranne il classificatore, che è l'ultimo (da qui il [:-1])
        X_transformed = pd.DataFrame(
            step.transform(X_transformed), columns=X.columns, index=X.index
        )
    return X_transformed

def shap_analysis(fitted_pipeline, X: pd.DataFrame, model_type="tree", background_data=None):
    """
    Calcola valori SHAP per il modello fittato, valutati sui punti in X.

    model_type: "tree" per RandomForest/XGBoost, "linear" per Elastic Net/SVM lineare.

    background_data: dataset di riferimento per l'explainer lineare (serve a
    stimare media/correlazioni delle feature). Per non introdurre leakage,
    passare SEMPRE il training set del fold (mai il test set su cui si sta
    spiegando il modello) — vedi out_of_fold_shap più sotto, che lo fa
    automaticamente.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("Installa 'shap' con: pip install shap")

    clf = fitted_pipeline.named_steps["clf"] # estrazione del modello

    # applica le trasformazioni precedenti (es. scaler) prima di passare a SHAP
    X_transformed = _transform_with_pipeline_steps(fitted_pipeline, X)

    if model_type == "tree" and gpu_utils.is_cuml_estimator(clf):
        raise ValueError("TreeExplainer non supporta la RF cuML: usa out_of_fold_shap(..., y=y), "
                         "che la rifitta su CPU con gli stessi iperparametri.")

    if model_type == "tree": # non serve passargli il bg data, il modello di tree sa già che a un certo
            # nodo c'è una proporzione di dati che vanno divisi a dx o a sx. quindi il valore medio è 
            # %dx * output medio dx + $sx * output medio sx, questa viene confrontata con output con 
            # feature
        explainer = shap.TreeExplainer(clf) # algoritmi ottimizzati a seconda del modello
        shap_values = explainer.shap_values(X_transformed) 
    elif model_type == "linear":
        bg = background_data if background_data is not None else X
        bg_transformed = _transform_with_pipeline_steps(fitted_pipeline, bg)
        # Estrae SEMPRE (coef, intercept), bypassando l'incompatibilità di SHAP 
        # con cuML o CalibratedClassifierCV
        coef_tuple = gpu_utils.linear_coef_intercept(clf)
        explainer = shap.LinearExplainer(coef_tuple, bg_transformed)
        shap_values = explainer.shap_values(X_transformed) # spiega il modello, considerando il paziente
            # che ti viene passato 
    else:
        raise ValueError("model_type deve essere 'tree' o 'linear'")

    return explainer, shap_values, X_transformed
        # restituisce (in ordine) oggetto shap che contiene logica del calcolo; matrice di numeri con 
        # stessa forma di X, dove ogni cella contiene valore shap per quel paziente e feature; dati pre
        # trasformati usati per il calcolo
        
def _cpu_twin_for_shap(results: dict, fold_i: int, X_train, y_train):
    """Rifitta su CPU, con gli stessi iperparametri del fold, un modello ad
    alberi che SHAP non può spiegare direttamente (RF cuML)."""
    name = results["model_name"]
    cpu_pipe, _ = get_model_grid(use_gpu=False)[name]
    cpu_pipe = clone(cpu_pipe).set_params(**results["best_params"][fold_i])
    y_bin = (y_train == config.POSITIVE_CLASS).astype(int)
    return cpu_pipe.fit(X_train, y_bin)


def out_of_fold_shap(results: dict, X: pd.DataFrame, model_type: str, y: pd.Series = None):
    """
    Calcola SHAP "out-of-fold": ogni paziente viene spiegato usando il
    modello del fold esterno in cui quel paziente era nel test set, quindi
    MAI col modello che lo ha visto in training.

    Perché conta: calcolare SHAP sul training set (come in un fit singolo)
    è ottimistico, soprattutto per modelli ad albero che possono aver
    "memorizzato" pattern specifici di quei pazienti. Questo approccio dà
    un'importanza delle feature coerente con la logica della nested CV già
    usata per le performance, invece di rompere quella logica solo per SHAP.

    y : necessario SOLO se il modello è una RF cuML (GPU): SHAP non la supporta,
    quindi per ogni fold si rifitta su CPU una RF con gli stessi iperparametri
    sullo stesso training set. Le SHAP spiegano quindi un modello equivalente
    ma non identico a quello usato per le predizioni (stessa famiglia, stessi
    parametri, algoritmo di split diverso).

    Richiede che results provenga da nested_cv_evaluate (contiene
    fitted_models e test_indices per ciascun fold esterno).

    Ritorna
    -------
    shap_df : DataFrame (pazienti x feature) con un valore SHAP per cella,
              allineato all'indice originale di X.
    mean_abs_shap : Series ordinata, importanza media |SHAP| per feature.
    """
    n_samples, n_features = X.shape
    shap_matrix = np.full((n_samples, n_features), np.nan)
    all_indices = np.arange(n_samples)

    for fold_i, (fold_model, test_idx) in enumerate(zip(results["fitted_models"], results["test_indices"])): # zip() function
            # is used to combine two or more iterables into a single iterator of tuples.
        train_idx = np.setdiff1d(all_indices, test_idx) #  finds the set difference of two arrays. It 
            # returns a sorted, 1D array of unique values in the first input array that are not present
            # in the second input array.
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]

        if model_type == "tree" and gpu_utils.is_cuml_estimator(fold_model.named_steps["clf"]):
            if y is None:
                raise ValueError("Per spiegare una RF cuML passa y a out_of_fold_shap(..., y=y).")
            gpu_utils._warn_once("SHAP: RF cuML non supportata, uso una RF CPU rifittata "
                                 "con gli stessi iperparametri di ogni fold.")
            fold_model = _cpu_twin_for_shap(results, fold_i, X_train_fold, y.iloc[train_idx])

        _, shap_values, _ = shap_analysis(
            fold_model, X_test_fold, model_type=model_type, background_data=X_train_fold
        ) # gli passo il modello allenato su 4 fold su 5, e come dataset il 5 fold non visto dal modello.
            # quindi sto valutando la shap sul fold che vede il paziente come test e non come train.
            # background data è usato per calcolare l'effetto della particolare feature. io non posso 
            # semplicemente eliminarla dal modello, il modello non lo accetta, quindi la sostituisco con
            # tutti i valori che assume nel training, ne faccio media e confronto l'effetto tra feature
            # vera del paziente e quella ottenuta dal dataset completo
        sv = shap_values
        if isinstance(sv, list):
            sv = sv[1]              # lista [classe_0, classe_1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]         # (n_campioni, n_feature, n_classi) -> classe positiva
        shap_matrix[test_idx, :] = sv

    if np.isnan(shap_matrix).any():
        missing = X.index[np.isnan(shap_matrix).any(axis=1)].tolist()
        raise RuntimeError(
            f"Alcuni pazienti non sono coperti da nessun fold di test: {missing}. "
            "Controlla che gli outer fold di nested_cv_evaluate coprano tutto il dataset."
        )

    shap_df = pd.DataFrame(shap_matrix, index=X.index, columns=X.columns)
    mean_abs_shap = shap_df.abs().mean(axis=0).sort_values(ascending=False)

    print(f"\n[out_of_fold_shap] SHAP calcolata out-of-fold per {n_samples} pazienti, "
          f"{n_features} feature (model_type={model_type})")

    return shap_df, mean_abs_shap

# SHAP (SHapley Additive exPlanations), se modelli AI visti come scatole nere SHAP apre la scatola e dice 
# esattamente ogni feature quanto conta nel risultato finale. valuta non solo la predizione finale del 
# modello (corretta o no), ma anche il merito/contributo di ogni feature al risultato per ogni paziente. 

# ---------------------------------------------------------------------------
# PLOT SHAP — salvati su file per essere ispezionati/riusati 
# ---------------------------------------------------------------------------
def plot_shap_bar(mean_abs_shap: pd.Series, output_path: str, top_n: int = 20):
    """
    Bar chart delle top_n feature per importanza percentuale relativa SHAP.
    Mostra quanto ciascuna feature contribuisce in percentuale all'impatto totale.
    """
    # 1. Calcolo la percentuale sul totale di tutti i valori SHAP
    shap_percentage = (mean_abs_shap / mean_abs_shap.sum()) * 100
    
    # 2. Seleziono le top_n e le ordino in modo crescente (necessario per plt.barh)
    top = shap_percentage.head(top_n).sort_values()
    
    # 3. Creazione del grafico
    plt.figure(figsize=(8, 0.35 * len(top) + 1))
    plt.barh(top.index, top.values, color="#4C72B0")
    
    # 4. Aggiornamento etichette
    plt.xlabel("Impatto Relativo SHAP (%)")
    plt.title(f"Top {len(top)} feature per importanza SHAP (%)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[plot_shap_bar] salvato in {output_path} (valori in percentuale)")


def plot_shap_summary(shap_df: pd.DataFrame, X: pd.DataFrame, output_path, max_display: int = 20):
    """
    Beeswarm plot: mostra per le feature più importanti sia la magnitudine
    dell'effetto SHAP sia la direzione (valore alto/basso della feature ->
    spinge la predizione verso una classe o l'altra). Più informativo del
    solo bar chart perché mostra anche il segno dell'effetto.
    """
    shap.summary_plot(shap_df.values, X, feature_names=X.columns,
                       max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_shap_summary] salvato in {output_path}")


def plot_shap_dependence(shap_df: pd.DataFrame, X: pd.DataFrame, feature: str, output_path):
    """
    Dependence plot per una singola feature: relazione tra il suo valore e
    il suo effetto SHAP sulla predizione, colorato per interazione con la
    feature più correlata. Utile per capire se l'effetto è lineare, a
    soglia, o non monotono — informazione che il solo coefficiente
    dell'Elastic Net non darebbe.
    """
    shap.dependence_plot(feature, shap_df.values, X, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_shap_dependence] salvato in {output_path}")



# ---------------------------------------------------------------------------
# BENCHMARK CPU vs GPU — per decidere QUALI modelli conviene mandare su GPU
# ---------------------------------------------------------------------------
def benchmark_backends(X: pd.DataFrame, y: pd.Series, n_repeats: int = 3):
    """
    Cronometra fit+predict_proba di UN set di iperparametri per modello su CPU e
    GPU, con i dati reali. Con ~50 pazienti la CPU spesso vince (overhead di
    lancio kernel e trasferimenti): usalo prima di fissare config.GPU_MODELS.
    Il primo fit GPU (warm-up, inizializzazione CUDA) viene scartato.
    """
    import time
    y_bin = (y == config.POSITIVE_CLASS).astype(int)
    cpu_models = get_model_grid(use_gpu=False)
    gpu_models = get_model_grid(use_gpu={n: True for n in cpu_models})
    rows = []
    for name, (cpu_pipe, grid) in cpu_models.items():
        gpu_pipe = gpu_models[name][0]
        if not gpu_utils.pipeline_uses_gpu(gpu_pipe):
            rows.append({"model": name, "cpu_s": None, "gpu_s": None, "note": "GPU non disponibile"})
            continue
        params = {k: v[-1] if k.endswith("n_estimators") else v[0] for k, v in grid.items()}
        times = {}
        for label, pipe in (("cpu_s", cpu_pipe), ("gpu_s", gpu_pipe)):
            pipe = clone(pipe).set_params(**{k: v for k, v in params.items() if v is not None})
            pipe.fit(X, y_bin)  # warm-up
            t0 = time.perf_counter()
            for _ in range(n_repeats):
                pipe.fit(X, y_bin)
                pipe.predict_proba(X)
            times[label] = (time.perf_counter() - t0) / n_repeats
        rows.append({"model": name, **times, "note": ""})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df