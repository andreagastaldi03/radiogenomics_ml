"""
Entry point della specification curve con selezione radiomica guidata dalla
rete (file NUOVO). Solo curva descrittiva + tabelle di lettura.

Uso:
    python run_specification_curve_net.py
Output in OUTPUT_DIR/network_selection/:
    spec_curve_net_results.csv          una riga per (specifica x modello)
    spec_curve_net.png                  la curva
    spec_curve_net_pivot.csv            AUC pooled: selettore/rete x (modello, k)
    selected_features_by_network.csv    quali feature ha scelto ogni selettore e perché
    radiomic_node_scores_<rete>.csv     tutte le misure di rete per nodo radiomico
    random_draws.csv                    le singole estrazioni del controllo casuale
"""

import pandas as pd

import config
import config_network_selection as cn
import network_feature_selection as nfs
import specification_curve as sc
import specification_curve_net as scn


def main():
    out = cn.OUTPUT_ROOT
    out.mkdir(parents=True, exist_ok=True)

    for name, path in cn.NETWORK_GRAPHML.items():
        G = nfs.load_network(path)
        nfs.radiomic_node_scores(G).sort_values("gene_strength", ascending=False)\
            .to_csv(out / f"radiomic_node_scores_{name}.csv")

    spec_df, selected_df, draws_df = scn.run_specification_curve_net()
    spec_df.to_csv(out / "spec_curve_net_results.csv", index=False)
    selected_df.to_csv(out / "selected_features_by_network.csv", index=False)
    draws_df.to_csv(out / "random_draws.csv", index=False)

    keys = ["data_source", "rad_selector", "network", "k_rad", "model_type"]
    sc.plot_specification_curve(spec_df, spec_keys=keys, output_path=out / "spec_curve_net.png")

    pivot = spec_df.pivot_table(index=["data_source", "rad_selector", "network"],
                                columns=["model_type", "k_rad"], values="auc_pooled")
    pivot.round(3).to_csv(out / "spec_curve_net_pivot.csv")
    print("\nAUC POOLED OOF (righe: sorgente/selettore/rete; colonne: modello/budget)\n")
    print(pivot.round(3).to_string())
    print("\nNB: random_net_nodes = mediana di N estrazioni casuali; guarda auc_pooled_p05/p95 "
          "in spec_curve_net_results.csv. Un selettore vale solo se sta sopra quella fascia.")
    print(f"\nRisultati salvati in: {out}")


if __name__ == "__main__":
    main()
