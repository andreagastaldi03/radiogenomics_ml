# Pipeline ML per studio radiogenomico (ADK vs SCC)

## Logica generale

Il problema centrale con questi dati è **n=54 pazienti vs centinaia di feature**.
Ogni scelta della pipeline è pensata per limitare l'overfitting e per dare stime
oneste della performance, mantenendo allo stesso tempo un output interpretabile
(coefficienti, feature stabili) che sarà riusato nello studio di rete.

## Struttura dei file

- `config.py` — tutti i parametri e percorsi in un unico posto. Modifica qui.
- `data_utils.py` — caricamento dati e **riduzione feature neutra** (varianza +
  ridondanza via clustering di correlazione). Non guarda mai la label: questo è
  intenzionale, per evitare circolarità tra selezione e stima delle performance,
  e per poter riusare lo stesso set ridotto nello studio di rete.
- `ml_pipeline.py` — nested cross-validation, modelli (Elastic Net, Random Forest,
  SVM lineare, XGBoost opzionale), stability selection via bootstrap, SHAP.
- `run_analysis.py` — orchestratore, salva tutti i risultati in `outputs/`.

## Come procedere passo-passo

1. **Prepara i dati**: metti in `data/` i tre CSV attesi (`radiomics_features.csv`,
   `genomics_features.csv`, `labels.csv`) con `patient_id` come chiave comune.
   Se hai già un unico file, basta adattare `data_utils.load_data()`.

2. **Prima esecuzione**: lancia con `config.DATA_SOURCE = "both"`.
   ```
   pip install -r requirements.txt
   python run_analysis.py
   ```

3. **Confronta le tre sorgenti dati**: rilancia con `"radiomics"` e `"genomics"`
   separatamente. Confronta `model_comparison_summary.csv` tra le tre run.
   Questo confronto — quanto separa ADK/SCC l'imaging da solo, la genomica da
   sola, o l'integrazione — è già un risultato scientifico rilevante per il tuo
   obiettivo di caratterizzazione, indipendentemente dalla performance assoluta.

4. **Guarda la stabilità, non solo la performance**: con n=54 la metrica più
   utile spesso non è l'AUC in sé (che avrà una varianza alta tra fold/seed),
   ma `stable_features_final.csv` — le feature/geni che l'Elastic Net seleziona
   consistentemente su centinaia di bootstrap. Sono quelle su cui costruirai
   l'interpretazione biologica e il collegamento con lo studio di rete.

5. **Ripeti la nested CV con più seed** (vedi `config.RANDOM_SEEDS_MULTI_RUN`):
   con un campione così piccolo, una singola run di CV può essere fortunata o
   sfortunata. Se vuoi, posso aggiungerti uno script che itera su più seed e
   produce un box-plot delle AUC per avere un'idea onesta della variabilità.

## Cose da NON fare (errori comuni con n<<p)

- Non fare feature selection supervisionata (es. filtrare per differenza tra
  gruppi) su tutto il dataset prima dello split train/test: è leakage, gonfia
  artificialmente la performance riportata. Nel codice, la selezione supervisionata
  avviene solo dentro la CV (Elastic Net con L1 dentro ogni fold).
- Non fidarti di una singola AUC da un singolo split: usa sempre nested CV o
  ripetizioni multiple.
- Non concludere che un gene/feature è "importante" da un solo fit: usa la
  stability selection bootstrap.
