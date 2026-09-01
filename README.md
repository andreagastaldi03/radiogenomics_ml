# Radiogenomics Analysis Pipeline

## Descrizione del Progetto
Questa repository contiene una pipeline computazionale per l'analisi radiogenomica. Il framework integra tecniche di Machine Learning e di Network Analysis per estrarre, valutare e interpretare feature complesse. L'obiettivo è fornire un ambiente modulare e riproducibile per l'addestramento dei modelli, la valutazione della loro robustezza e l'analisi relazionale dei dati.

## Funzionalità Principali
*   **Machine Learning Pipeline:** Supporto per l'addestramento, l'ottimizzazione e la validazione incrociata nidificata di modelli predittivi (come Elastic Net, Random Forest, SVM, XGBoost), inclusa l'interpretazione dell'importanza delle feature tramite l'analisi dei valori SHAP.
*   **Network Analysis:** Costruzione e analisi di reti per esplorare la topologia e le relazioni latenti all'interno dei dati.
*   **Integrazione ML-Network:** Moduli dedicati per unire le feature estratte dai modelli di apprendimento automatico con le metriche di rete.
*   **Specification Curve Analysis (SCA):** Strumenti per testare la robustezza delle decisioni analitiche e l'affidabilità dei risultati variando le specifiche del modello.
*   **Diagnostica e Validazione statistica:** Script per il confronto delle distribuzioni, test di significatività e diagnostica delle performance predittive.

## Struttura della Repository
La base di codice è organizzata nei seguenti moduli[cite: 1]:

*   **Core Machine Learning**
    *   `ml_pipeline.py`: Moduli per l'addestramento e la validazione dei modelli ML.
    *   `radiogenomics.py`: Modulo per lo studio delle feature abbandonando il collo di bottiglia dell'etichetta binaria ADK/SCC.
*   **Core Network Analysis**
    *   `network_analysis.py`: Logica per la costruzione dei grafi e l'estrazione delle metriche di rete.
    *   `ml_network_bridge.py`: Interfaccia per la combinazione dei risultati ML e di rete.
*   **Analisi di Robustezza e Diagnostica**
    *   `specification_curve.py` / `network_specification_curve.py`: Implementazione della SCA per entrambi i domini.
    *   `diagnostics.py` / `network_diagnostics.py`: Strumenti per la valutazione delle performance e il controllo di qualità.
    *   `delong.py`: Implementazione statistica (es. test di DeLong per il confronto delle curve ROC).
*   **Selezione e Consenso delle Feature**
    *   `feature_consensus.py` / `consensus_significance.py`: Moduli per valutare la stabilità e la significatività delle feature selezionate.
*   **Utility e Configurazione**
    *   `data_utils.py`: Funzioni di supporto per il preprocessing e la manipolazione dei dati.
    *   `config.py`: Parametri di configurazione globale del progetto.
    *   `compare_data_sources.py`: Routine per l'allineamento e il confronto di dataset eterogenei.
*   **Entry Points (Esecuzione)**
    *   `run_analysis.py` / `run_diagnostics.py` / `run_specification_curve.py`: Script principali per avviare le analisi in batch.

## Installazione
Clonare la repository (assicurarsi di aver configurato correttamente le chiavi SSH) e installare le dipendenze in un ambiente virtuale.
```bash
pip install -r requirements.txt
