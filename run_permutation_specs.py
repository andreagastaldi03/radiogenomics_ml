"""
Entry point dei test di permutazione sulle specifiche (file NUOVO).

Esempi:
    python run_permutation_specs.py --curve net --test labels --n_perm 100 --dry_run
    python run_permutation_specs.py --curve net --test both   --n_perm 200
    python run_permutation_specs.py --curve v2  --test labels --n_perm 100

--curve v2 : specifiche di specification_curve_v2 (legacy / medoid / pca x budget)
--curve net: specifiche di specification_curve_net (selettori di rete)
--test labels : permutazione delle etichette + maxT su tutte le specifiche
--test block  : permutazione del solo blocco radiomico (solo --curve net)
--dry_run : esegue solo il passaggio reale e stampa la stima dei tempi

Output in OUTPUT_DIR/permutation_specs/<curva>/.
"""

import argparse

import pandas as pd

import config
import data_utils
import permutation_specs as ps


def main():
    ap = argparse.ArgumentParser()
    # ArgumentParser che gestirà il parsing dei flag passati all'avvio dello script
    ap.add_argument("--curve", choices=["v2", "net"], default="net")
    ap.add_argument("--test", choices=["labels", "block", "both"], default="labels")
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--models", default="linear,tree")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    models = a.models.split(",")

    out = config.OUTPUT_DIR / "permutation_specs" / a.curve
    out.mkdir(parents=True, exist_ok=True)

    X, y = data_utils.load_data(source="both", print_info=False)

    if a.curve == "v2":
        if a.test != "labels":
            raise SystemExit("--test block richiede --curve net (serve un blocco genomico fisso)")
        specs = ps.build_specs_v2(X)
        X_genes = rad_sets = None
    else:
        X_genes, rad_sets = ps.build_net_blocks(X)
        specs = ps.specs_from_net_blocks(X_genes, rad_sets)

    if a.test in ("labels", "both"):
        res, null = ps.label_permutation_maxstat(specs, y, models, a.n_perm, n_jobs=a.n_jobs,
                                                 dry_run=a.dry_run)
        if res is not None:
            res.to_csv(out / "label_permutation_results.csv", index=False)
            null.columns = [f"{s}|{m}" for s, m in null.columns]
            null.to_csv(out / "label_permutation_null_auc.csv", index=False)
            ps.plot_label_permutation(
                res, pd.DataFrame({(r.spec, r.model_type): null[f"{r.spec}|{r.model_type}"]
                                   for r in res.itertuples()}),
                out / "label_permutation_curve.png")
            print("\nPERMUTAZIONE ETICHETTE (p_adj_family: corretto per aver provato tutte le "
                  "specifiche della stessa data_source)")
            print(res[["spec", "model_type", "auc_real", "null_p95", "p_raw", "p_adj_family",
                       "p_adj_global"]].round(3).to_string(index=False))

    if a.test in ("block", "both") and a.curve == "net":
        res, null = ps.block_permutation_added_value(rad_sets, X_genes, y, models, a.n_perm,
                                                     n_jobs=a.n_jobs, dry_run=a.dry_run)
        if res is not None:
            res.to_csv(out / "block_permutation_results.csv", index=False)
            null.columns = [f"{s}|{m}" for s, m in null.columns]
            null.to_csv(out / "block_permutation_null_diff.csv", index=False)
            print("\nVALORE AGGIUNTO DELLA RADIOMICA AI GENI (blocco radiomico permutato)")
            print(res[["spec", "model_type", "auc_genes", "auc_both", "auc_diff_real", "null_p95",
                       "p_raw", "p_adj_global"]].round(3).to_string(index=False))

    print(f"\nRisultati in: {out}")


if __name__ == "__main__":
    main()
