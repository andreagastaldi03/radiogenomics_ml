import config
import specification_curve as sc

def main(run_joint_test: bool = True):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spec_df, feature_votes, feature_votes_by_model, feature_stats_long_df = sc.run_specification_curve()

    spec_df.to_csv(config.OUTPUT_DIR / "specification_curve_results.csv", index=False)
    feature_votes.to_csv(config.OUTPUT_DIR / "feature_votes_across_specs.csv",    
                         header=["n_specs_selected"])
    feature_stats_long_df.to_csv(config.OUTPUT_DIR / "feature_stats_per_spec_long.csv", index=False)

    n_specs_per_model = spec_df.groupby("model_type").size()
    for model_type, votes in feature_votes_by_model.items():
        votes.to_csv(config.OUTPUT_DIR / f"feature_votes_across_specs_{model_type}.csv",
                      header=["n_specs_selected"])
        sc.plot_feature_votes(
            votes, n_total_specs=n_specs_per_model[model_type],
            output_path=config.OUTPUT_DIR / f"feature_votes_{model_type}.png"
        )

    spec_keys = list(sc.SPEC_GRID.keys()) + ["model_type"]
    sc.plot_specification_curve(
        spec_df, spec_keys=spec_keys,
        output_path=config.OUTPUT_DIR / "specification_curve.png"
    )

    # ------------------------------------------------------------------
    # Test di significatività congiunto (parte inferenziale della SCA).
    # Rifà l'intera curva su una griglia RIDOTTA (sc.REDUCED_SPEC_GRID)
    # config.N_PERMUTATIONS_SPEC_CURVE volte: può richiedere molto tempo,
    # per saltarlo lancia main(run_joint_test=False).
    # ------------------------------------------------------------------
    if run_joint_test:
        real_spec_df, real_stat, null_stats, p_value = sc.joint_significance_test()

        real_spec_df.to_csv(config.OUTPUT_DIR / "joint_test_real_curve_reduced_grid.csv", index=False)
        import pandas as pd
        pd.Series(null_stats, name=f"{config.SPEC_CURVE_SUMMARY_STAT}_auc_pooled_permutato").to_csv(
            config.OUTPUT_DIR / "joint_significance_test_null_distribution.csv", index=False
        )
        sc.plot_joint_significance_test(
            null_stats, real_stat, config.SPEC_CURVE_SUMMARY_STAT,
            config.OUTPUT_DIR / "joint_significance_test.png"
        )
        with open(config.OUTPUT_DIR / "joint_significance_test_summary.txt", "w") as f:
            f.write(f"Griglia ridotta: {sc.REDUCED_SPEC_GRID}\n")
            f.write(f"Statistica riassuntiva: {config.SPEC_CURVE_SUMMARY_STAT} di auc_pooled\n")
            f.write(f"Valore osservato: {real_stat:.4f}\n")
            f.write(f"Valore nullo (permutato): {null_stats.mean():.4f} ± {null_stats.std():.4f}\n")
            f.write(f"p-value empirico: {p_value:.4f}\n")

if __name__ == "__main__":
    main()