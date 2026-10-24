"""
Entry point: prevedibilità dei geni dalla radiomica, condizionata al fenotipo
(file NUOVO). Output in OUTPUT_DIR/gene_predictability/<covariate>/.

    python run_gene_predictability.py --dry_run
    python run_gene_predictability.py --n_perm 2000
    python run_gene_predictability.py --n_perm 2000 --covariates class_volume

Lancialo due volte (class e class_volume): il secondo mostra cosa resta dopo
aver tolto anche l'effetto del volume tumorale.
"""

import argparse

import config
import data_utils
import gene_predictability as gp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covariates", choices=["class", "class_volume"], default="class")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--n_repeats", type=int, default=3, help="ripetizioni della 5-fold CV")
    ap.add_argument("--k_topk", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    out = config.OUTPUT_DIR / "gene_predictability" / a.covariates
    out.mkdir(parents=True, exist_ok=True)

    X, y = data_utils.load_data(source="both", print_info=False)
    res, null_c, null_u, tinfo = gp.run_gene_predictability(
        X, y, covariates=a.covariates, n_perm=a.n_perm, n_repeats=a.n_repeats,
        k_topk=a.k_topk, n_jobs=a.n_jobs, dry_run=a.dry_run)
    tinfo.to_csv(out / "targets_info.csv", index=False)
    if res is None:
        return

    res.to_csv(out / "gene_predictability_results.csv", index=False)
    for nm, nl in (("cond", null_c), ("unc", null_u)):
        nl.columns = [f"{t}|{p}" for t, p in nl.columns]
        nl.to_csv(out / f"null_dR2_{nm}.csv", index=False)
    gp.plot_gene_predictability(res, out / "gene_predictability.png")

    cols = ["target", "family", "predictor", "dR2_cond", "dR2_cond_null_p95", "p_raw_cond",
            "p_adj_cond_within_predictor", "dR2_unc", "p_raw_unc", "r2_covariates_only"]
    print("\nPREVEDIBILITÀ DEI GENI DALLA RADIOMICA (primi 25 per dR2 condizionato)")
    print(res[cols].head(25).round(3).to_string(index=False))
    print(f"\nRisultati in: {out}")


if __name__ == "__main__":
    main()
