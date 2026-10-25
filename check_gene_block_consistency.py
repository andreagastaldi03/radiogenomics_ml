"""
Controllo: perché lo stesso blocco genico (5 componenti PCA) dà AUC molto diverse
nella specification curve v2 (genomics|pca|k_small) e in quella di rete
(rad_selector='none')? Confronta i dati in ingresso, i parametri e le feature
prodotte dai due percorsi di codice, e rilancia la valutazione di entrambi.

Uso:  python check_gene_block_consistency.py
"""

import numpy as np
import pandas as pd

import config
import config_network_selection as cn
import data_utils
import reduction_v2
import specification_curve as sc
import specification_curve_v2 as sc2


def main():
    print("PARAMETRI ATTUALI")
    print("  GENE_SELECTION_METHOD:", getattr(config, "GENE_SELECTION_METHOD", None),
          "| GENE_IQR_TOP_PCT:", config.GENE_IQR_TOP_PCT, "| RANDOM_STATE:", config.RANDOM_STATE)
    print("  GENE_REDUCTION (rete):", cn.GENE_REDUCTION)

    Xg, yg = data_utils.load_data(source="genomics", print_info=False)
    Xb, yb = data_utils.load_data(source="both", print_info=False)
    gen_b = [c for c in Xb.columns if c.startswith("gen__")]
    print("\nDATI IN INGRESSO")
    print(f"  genomics: {Xg.shape} | both: {Xb.shape} ({len(gen_b)} colonne geniche)")
    print("  stessi pazienti, stesso ordine:", list(Xg.index) == list(Xb.index))
    print("  stesse etichette:", bool((yg.reindex(Xb.index).values == yb.values).all()))
    common = [c for c in Xg.columns if c in Xb.columns]
    print(f"  colonne geniche comuni: {len(common)} su {Xg.shape[1]} (genomics) / {len(gen_b)} (both)")
    if common:
        diff = np.abs(Xg.loc[Xb.index, common].to_numpy(float) - Xb[common].to_numpy(float)).max()
        print(f"  massima differenza di valore sulle colonne comuni: {diff:.3g}")

    spec = dict(data_source="genomics", reduction_mode="pca", budget="k_small")
    P_v2 = sc2._reduce(Xg, spec)
    red_net = reduction_v2.BlockReducer(k_texture=None, k_shape=None,
                                        gen_prefilter_pct=config.GENE_IQR_TOP_PCT,
                                        **cn.GENE_REDUCTION).fit(Xb[gen_b])
    P_net = red_net.transform(Xb[gen_b])

    print("\nFEATURE PRODOTTE")
    print("  v2 :", list(P_v2.columns))
    print("  net:", list(P_net.columns))
    # Pazienti comuni a entrambi i percorsi
    common_idx = P_v2.index.intersection(P_net.index)
    C = np.abs(np.corrcoef(P_v2.loc[common_idx].to_numpy().T, 
                           P_net.loc[common_idx].to_numpy().T)[:P_v2.shape[1], P_v2.shape[1]:])
    print("  |correlazione| tra le componenti dei due percorsi (righe v2, colonne net):")
    print(pd.DataFrame(C, index=range(1, C.shape[0] + 1), columns=range(1, C.shape[1] + 1)).round(2).to_string())

    common_idx = P_v2.index.intersection(P_net.index)
    y_common = (yb.loc[common_idx] == config.POSITIVE_CLASS).astype(int).reset_index(drop=True)

    print("\nVALUTAZIONE ATTUALE (linear, stesso seed - solo pazienti comuni)")
    for name, P in (("v2 ", P_v2.loc[common_idx]), ("net", P_net.loc[common_idx])):
        res = sc._cv_eval(P.reset_index(drop=True), y_common, "linear", 5,
                          random_state=config.RANDOM_STATE, n_jobs_model=-1)
        print(f"  {name}: auc_pooled = {res[2]:.3f}")
    print("\nNei file caricati: v2 = 0.868, net = 0.663. Se qui i due valori coincidono, "
          "la differenza viene da codice/config usati al momento delle esecuzioni.")


if __name__ == "__main__":
    main()
