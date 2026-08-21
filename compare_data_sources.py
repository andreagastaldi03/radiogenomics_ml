"""
Confronto statistico tra due sorgenti dati (es. "genomics" vs "both") sugli
stessi pazienti, riusando le predizioni out-of-fold già salvate da
run_pooled_oof_analysis (nessun nuovo fit).

Domanda a cui risponde: "aggiungere la radiomica ai geni migliora davvero
le predizioni, o la differenza osservata è dentro il rumore atteso con
n=54?"

Metodo: bootstrap appaiato sulla differenza di AUC pooled. Ad ogni
iterazione si ricampionano i PAZIENTI (stessi indici per entrambe le
condizioni), preservando l'accoppiamento paziente-per-paziente — è quello
che rende il confronto più potente di un test naive tra due AUC
indipendenti, perché elimina la variabilità dovuta a "quali pazienti sono
nel campione" e isola quella dovuta a "quale sorgente dati uso".

Dato lo stesso insieme di pazienti e due modelli che producono due serie di 
probabilità predette, la performance AUC del modello B è significativamente 
diversa da quella del modello A?

Uso:
    python compare_data_sources.py --model elastic_net --source_a genomics --source_b both
(richiede aver già girato run_analysis.py con entrambi i DATA_SOURCE)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import config


def paired_bootstrap_auc_diff(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray,
                               n_boot: int = 2000, random_state: int = config.RANDOM_STATE):
    """
    Ritorna
    -------
    observed_diff : AUC(b) - AUC(a) sui dati osservati
    ci_low, ci_high : CI 95% percentile della differenza
    p_value : due code, per H0: differenza = 0
    boot_diffs : tutte le differenze bootstrap, per il plot
    """
    n = len(y_true) # dimensione del dataset
    observed_diff = roc_auc_score(y_true, proba_b) - roc_auc_score(y_true, proba_a) 
        # reali differenze tra auc dei due modelli

    rng = np.random.RandomState(random_state)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n) # creo oggetto di dim n, con numeri random tra 0 e n
        y_b = y_true[idx]
        # con classi sbilanciate un ricampionamento può capitare tutto in una
        # classe sola: l'AUC non è definita, si ricampiona finché non serve
        while len(np.unique(y_b)) < 2:
            idx = rng.randint(0, n, size=n)
            y_b = y_true[idx]
        auc_a = roc_auc_score(y_b, proba_a[idx])
        auc_b = roc_auc_score(y_b, proba_b[idx]) # stessi pazienti per entrambe le prob, per 
            # entrambi i modelli, auc relative agli stessi pazienti
        boot_diffs[i] = auc_b - auc_a

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5]) # prende il 2,5 e il 97,5 percentile, 
        # valori plausibili dovrebbero essere compresi tra questi due valori - interv di confid del
        # 95%
    if observed_diff >= 0:
        p_value = min(2 * (boot_diffs <= 0).mean(), 1.0)
    else:
        p_value = min(2 * (boot_diffs >= 0).mean(), 1.0)

    return observed_diff, ci_low, ci_high, p_value, boot_diffs


def plot_bootstrap_diff(boot_diffs, observed_diff, source_a, source_b, output_path):
    plt.figure(figsize=(7, 5))
    plt.hist(boot_diffs, bins=40, color="#8C8C8C", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1, label="nessuna differenza")
    plt.axvline(observed_diff, color="#C44E52", linewidth=2,
                label=f"differenza osservata = {observed_diff:+.3f}")
    plt.xlabel(f"AUC({source_b}) - AUC({source_a})")
    plt.ylabel("Numero di bootstrap")
    plt.title(f"Bootstrap appaiato: {source_b} aggiunge segnale a {source_a}?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_bootstrap_diff] salvato in {output_path}")


def compare_sources(model_name: str, source_a: str, source_b: str, n_boot: int = 2000):
    path_a = config.OUTPUT_DIR / source_a / f"{model_name}_oof_predictions.csv"
    path_b = config.OUTPUT_DIR / source_b / f"{model_name}_oof_predictions.csv"
    df_a, df_b = pd.read_csv(path_a), pd.read_csv(path_b)

    # allineamento sui pazienti in comune: "both" può avere un sottoinsieme
    # diverso da "genomics" da sola, se a qualcuno manca una delle due fonti
    merged = df_a.merge(df_b, on="patient", suffixes=(f"_{source_a}", f"_{source_b}"))
    n_dropped = min(len(df_a), len(df_b)) - len(merged)
    if n_dropped > 0:
        print(f"[compare_sources] ATTENZIONE: {n_dropped} pazienti presenti in una sola "
              f"delle due sorgenti sono stati esclusi dal confronto appaiato.")

    y_true_a = merged[f"y_true_{source_a}"].to_numpy()
    y_true_b = merged[f"y_true_{source_b}"].to_numpy()
    if not np.array_equal(y_true_a, y_true_b):
        raise RuntimeError("Le etichette non coincidono tra le due sorgenti per gli stessi "
                            "pazienti: controlla l'allineamento in data_utils.load_data.")

    proba_a = merged[f"oof_probability_{source_a}"].to_numpy()
    proba_b = merged[f"oof_probability_{source_b}"].to_numpy()

    observed_diff, ci_low, ci_high, p_value, boot_diffs = paired_bootstrap_auc_diff(
        y_true_a, proba_a, proba_b, n_boot=n_boot
    )

    print(f"\n[compare_sources] modello={model_name} | {source_a} vs {source_b} | n={len(merged)}")
    print(f"  AUC {source_a}: {roc_auc_score(y_true_a, proba_a):.3f}")
    print(f"  AUC {source_b}: {roc_auc_score(y_true_b, proba_b):.3f}")
    print(f"  differenza ({source_b} - {source_a}): {observed_diff:+.3f} "
          f"[95% CI {ci_low:+.3f}, {ci_high:+.3f}] | p={p_value:.4f}")
    if ci_low < 0 < ci_high:
        print(f"  -> il CI include lo 0: NON c'è evidenza sufficiente che {source_b} "
              f"migliori su {source_a} con questo campione.")
    else:
        print(f"  -> il CI esclude lo 0: {source_b} sembra offrire un vantaggio reale.")

    out_dir = config.OUTPUT_DIR / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_bootstrap_diff(boot_diffs, observed_diff, source_a, source_b,
                         out_dir / f"bootstrap_diff_{model_name}_{source_a}_vs_{source_b}.png")
    pd.Series(boot_diffs, name="auc_diff_bootstrap").to_csv(
        out_dir / f"bootstrap_diff_{model_name}_{source_a}_vs_{source_b}.csv", index=False)

    return observed_diff, ci_low, ci_high, p_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="elastic_net")
    parser.add_argument("--source_a", default="genomics")
    parser.add_argument("--source_b", default="both")
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()
    compare_sources(args.model, args.source_a, args.source_b, args.n_boot)