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
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)

import config

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
def get_model_grid():
    """
    Restituisce un dizionario {nome_modello: (pipeline, param_grid)}.

    Elastic Net è il modello principale consigliato: fa selezione automatica
    delle feature (L1) gestendo al contempo la collinearità (L2), ed è
    interpretabile tramite i coefficienti.

    Random Forest e SVM lineare sono inclusi come confronto / cross-check
    dell'importanza delle feature (via feature_importances_ o SHAP).
    """
    models = {}

    # --- Elastic Net (logistic regression con penalità elasticnet) ---
    en_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            random_state=config.RANDOM_STATE
        )),
    ])
    # elastic net contiene both L1 e L2, C e l1 cercano un bilancio tra le due
    en_grid = {
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0],
        "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
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
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, random_state=config.RANDOM_STATE)),
    ])
    svm_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
    }
    models["svm_linear"] = (svm_pipe, svm_grid)

    # --- XGBoost (opzionale) ---
    if XGB_AVAILABLE:
        xgb_pipe = Pipeline([
            ("clf", XGBClassifier(
                eval_metric="logloss", random_state=config.RANDOM_STATE
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
    selection bootstrap serve UN solo set di iperparametri fissi (rifare un
    grid search dentro ognuno dei centinaia di bootstrap sarebbe troppo
    costoso e statisticamente ridondante). Il voto di maggioranza è un modo
    semplice e trasparente per sceglierli in modo data-driven, non arbitrario.
    """
    param_tuples = [tuple(sorted(p.items())) for p in best_params_list]
    most_common, count = Counter(param_tuples).most_common(1)[0]
    print(f"[majority_vote_params] combinazione più frequente: {dict(most_common)} "
          f"(scelta in {count}/{len(best_params_list)} fold)")
    return dict(most_common)
        # ogni nested cv testa gli iperparam, ma la cv esterna produce diversi possibili valori degli 
        # iperparam (uno per ogni fold). trasforma queste coppie iperparam-valore in tupla. poi viene
        # contato quante volte, sul numero di fold, questi iperparam sono scelti. prendo il più frequente 
        # e lo trasformo in dictionary di nuovo. voto di maggioranza per evitare fluttuazioni del valore
        # per scelta di fold diversi
        
# ---------------------------------------------------------------------------
# NESTED CROSS VALIDATION
# ---------------------------------------------------------------------------
def nested_cv_evaluate(X: pd.DataFrame, y: pd.Series, model_name: str,
                        pipe, grid, n_outer=config.N_OUTER_FOLDS,
                        n_inner=config.N_INNER_FOLDS, random_state=config.RANDOM_STATE):
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
        "fitted_models": [], "test_indices": []
    }

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_bin)):
            # split genera indici di posizione per train e test set, enumerate aggiunge un counter per 
            # tenere conto dell'iterazione che stiamo considerando
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state) # creo un 
            # Cross Validation sul train set per tuning degli iperparametri
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1) # testa tutte le 
            # combinazioni di iperparametri definite in "grid" usando come metrica ROC-AUC
        search.fit(X_train, y_train) # esegue effettivamente la ricerca, allena il modello sulla 
            # combinazione di iperparametri. una volta trovata quella vincente, riaddestra la pipeline 
            # con quei valori

        best_model = search.best_estimator_ # prendo il modello migliore
        y_pred = best_model.predict(X_test) # lo alleno sul test set, estrae la classe (non la prob)
        y_proba = best_model.predict_proba(X_test)[:, 1] # ritorna la probabilità continua, assegnata 
            # alla classe positiva. il metodo restituisce una mtrice a due colonne, prob per ogni classe
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
    models = get_model_grid()
    all_results = {}
    for name, (pipe, grid) in models.items():
        print(f"\n{'='*70}\nModello: {name}\n{'='*70}")
        all_results[name] = nested_cv_evaluate(X, y, name, pipe, grid)
    return all_results


# ---------------------------------------------------------------------------
# STABILITY SELECTION (bootstrap) — quali feature emergono ripetutamente
# ---------------------------------------------------------------------------
def bootstrap_stability_selection(X: pd.DataFrame, y: pd.Series,
                                   C: float, l1_ratio: float,
                                   n_bootstrap=config.N_BOOTSTRAP,
                                   threshold=config.STABILITY_SELECTION_THRESHOLD,
                                   random_state=config.RANDOM_STATE):
    """
    Rifitta un Elastic Net su n_bootstrap campioni bootstrap del dataset e
    conta quante volte ciascuna feature riceve un coefficiente non nullo.

    Con n=54 questo è più informativo di un singolo fit: ti dice quali
    feature sono "affidabilmente" rilevanti e non un artefatto del
    particolare campione osservato. Queste sono le feature da riportare
    come risultato principale, non i coefficienti di un singolo fit.
    
    IMPORTANTE: C e l1_ratio NON hanno un default arbitrario qui apposta.
    Vanno passati quelli scelti dalla nested CV (vedi majority_vote_params
    su all_results["elastic_net"]["best_params"]), altrimenti la stability
    selection userebbe un modello diverso da quello effettivamente validato
    come migliore in run_all_models().
    """
    y_bin = (y == config.POSITIVE_CLASS).astype(int)
    n_samples = X.shape[0]
    selection_counts = pd.Series(0, index=X.columns) # crea df di zeri, indicizzata coi nomi delle
        # features di X

    rng = np.random.RandomState(random_state) # inizializza generatore di numeri casuali con seed fisso

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            C=C, l1_ratio=l1_ratio, random_state=random_state
        )),
    ])

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True) # creo un campion bootstrap, dimensione
            # di X, sorteggiato in modo casuale, possibilità di reinserimento, quindi più copie di un 
            # paziente e altri completamente assenti
        X_boot, y_boot = X.iloc[idx], y_bin.iloc[idx]

        if y_boot.nunique() < 2: # controlla ci siano entrambe le classi 
            continue  # bootstrap degenere, salta

        pipe.fit(X_boot, y_boot) # addestro la pipe sul campione boot
        coefs = pipe.named_steps["clf"].coef_.ravel() # prende coeff associati alle feature e li 
            # appiattisce in un vettore 1d (ravel())
        selected = X.columns[np.abs(coefs) > 1e-8] # scelgo le label delle colonne aventi coeff maggiori
            # di zero (1e-8 per prob precisione numeri al computer)  
        selection_counts[selected] += 1

    stability_freq = selection_counts / n_bootstrap
    stable_features = stability_freq[stability_freq >= threshold].sort_values(ascending=False)

    print(f"\n[stability_selection] {len(stable_features)} feature stabili "
          f"(selezionate in >={threshold*100:.0f}% dei {n_bootstrap} bootstrap)")
    print(stable_features)

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

    if model_type == "tree":
        explainer = shap.TreeExplainer(clf) # algoritmi ottimizzati a seconda del modello
        shap_values = explainer.shap_values(X_transformed)
    elif model_type == "linear":
        bg = background_data if background_data is not None else X
        bg_transformed = _transform_with_pipeline_steps(fitted_pipeline, bg)
        explainer = shap.LinearExplainer(clf, bg_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        raise ValueError("model_type deve essere 'tree' o 'linear'")

    return explainer, shap_values, X_transformed
        # restituisce (in ordine) oggetto shap che contiene logica del calcolo; matrice di numeri con 
        # stessa forma di X, dove ogni cella contiene valore shap per quel paziente e feature; dati pre
        # trasformati usati per il calcolo
        
def out_of_fold_shap(results: dict, X: pd.DataFrame, model_type: str):
    """
    Calcola SHAP "out-of-fold": ogni paziente viene spiegato usando il
    modello del fold esterno in cui quel paziente era nel test set, quindi
    MAI col modello che lo ha visto in training.

    Perché conta: calcolare SHAP sul training set (come in un fit singolo)
    è ottimistico, soprattutto per modelli ad albero che possono aver
    "memorizzato" pattern specifici di quei pazienti. Questo approccio dà
    un'importanza delle feature coerente con la logica della nested CV già
    usata per le performance, invece di rompere quella logica solo per SHAP.

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

    for fold_model, test_idx in zip(results["fitted_models"], results["test_indices"]): # zip() function
            # is used to combine two or more iterables into a single iterator of tuples.
        train_idx = np.setdiff1d(all_indices, test_idx) #  finds the set difference of two arrays. It 
            # returns a sorted, 1D array of unique values in the first input array that are not present
            # in the second input array.
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]

        _, shap_values, _ = shap_analysis(
            fold_model, X_test_fold, model_type=model_type, background_data=X_train_fold
        )
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
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
# PLOT SHAP — salvati su file per essere ispezionati/riusati (es. nello studio di rete)
# ---------------------------------------------------------------------------
def plot_shap_bar(mean_abs_shap: pd.Series, output_path, top_n: int = 20):
    """Bar chart delle top_n feature per importanza media |SHAP|."""
    top = mean_abs_shap.head(top_n).sort_values()
    plt.figure(figsize=(8, 0.35 * len(top) + 1))
    plt.barh(top.index, top.values, color="#4C72B0")
    plt.xlabel("mean |SHAP value|")
    plt.title(f"Top {len(top)} feature per importanza SHAP")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_shap_bar] salvato in {output_path}")


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

