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
            C=0.1, l1_ratio=0.5, random_state=random_state
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
# SHAP — interpretabilità del modello (usare sul modello finale, fit su tutti i dati
# o su un modello rappresentativo tra i fold)
# ---------------------------------------------------------------------------
def shap_analysis(fitted_pipeline, X: pd.DataFrame, model_type="tree"):
    """
    Calcola valori SHAP per il modello fittato.

    model_type: "tree" per RandomForest/XGBoost, "linear" per Elastic Net/SVM lineare.
    Ritorna l'oggetto shap_values e l'explainer, utili per i plot successivi
    (shap.summary_plot, shap.dependence_plot, ecc. — da eseguire in notebook/script separato
    per la visualizzazione).
    """
    if not SHAP_AVAILABLE:
        raise ImportError("Installa 'shap' con: pip install shap")

    clf = fitted_pipeline.named_steps["clf"] # estrazione del modello

    # applica le trasformazioni precedenti (es. scaler) prima di passare a SHAP
    X_transformed = X.copy()
    for step_name, step in fitted_pipeline.steps[:-1]: # applica tutte le trasformazioni della pipeline a
            # X tranne il classificatore, che è l'ultimo (da qui il [:-1])
        X_transformed = pd.DataFrame(
            step.transform(X_transformed), columns=X.columns, index=X.index
        )

    if model_type == "tree":
        explainer = shap.TreeExplainer(clf) # algoritmi ottimizzati a seconda del modello
        shap_values = explainer.shap_values(X_transformed)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(clf, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        raise ValueError("model_type deve essere 'tree' o 'linear'")

    return explainer, shap_values, X_transformed
        # restituisce (in ordine) oggetto shap che contiene logica del calcolo; matrice di numeri con 
        # stessa forma di X, dove ogni cella contiene valore shap per quel paziente e feature; dati pre
        # trasformati usati per il calcolo

# SHAP (SHapley Additive exPlanations), se modelli AI visti come scatole nere SHAP apre la scatola e dice 
# esattamente ogni feature quanto conta nel risultato finale. valuta non solo la predizione finale del 
# modello (corretta o no), ma anche il merito/contributo di ogni feature al risultato per ogni paziente. 
