import config
import specification_curve as sc

def main(run_joint_test: bool = True):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True, run_source_comparison_test: bool = True,
         source_comparison_pair=("genomics", "both"))

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
        real_spec_df, real_stat, null_stats, p_value, p_ci_low, p_ci_high = sc.joint_significance_test()

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
            
    # ------------------------------------------------------------------
    # Test di significatività congiunto appaiato: la sorgente B (es. "both")
    # aggiunge davvero segnale rispetto alla sorgente A (es. "genomics"),
    # attraverso tutte le combinazioni di preprocessing insieme? Generalizza
    # compare_data_sources.py (un solo modello) a tutta la griglia ridotta.
    # ------------------------------------------------------------------
    if run_source_comparison_test:
        source_a, source_b = source_comparison_pair
        (real_diff_df, real_diff_stat, null_diff_stats,
         p_value_diff, p_ci_low_diff, p_ci_high_diff) = sc.joint_significance_test_source_comparison(
            source_a=source_a, source_b=source_b
        )
 
        suffix = f"{source_a}_vs_{source_b}"
        real_diff_df.to_csv(config.OUTPUT_DIR / f"joint_test_source_comparison_{suffix}.csv", index=False)
        import pandas as pd
        pd.Series(null_diff_stats, name=f"{config.SPEC_CURVE_SUMMARY_STAT}_diff_permutato").to_csv(
            config.OUTPUT_DIR / f"joint_test_source_comparison_{suffix}_null_distribution.csv", 
            index=False
        )
        sc.plot_joint_significance_test(
            null_diff_stats, real_diff_stat, config.SPEC_CURVE_SUMMARY_STAT,
            config.OUTPUT_DIR / f"joint_test_source_comparison_{suffix}.png",
            xlabel=f"{config.SPEC_CURVE_SUMMARY_STAT} di [auc_pooled({source_b}) -" 
                   f" auc_pooled({source_a})]",
            title=f"{source_b} aggiunge segnale a {source_a}? (test congiunto appaiato)"
        )
        with open(config.OUTPUT_DIR / f"joint_test_source_comparison_{suffix}_summary.txt", "w") as f:
            f.write(f"Confronto appaiato: {source_b} vs {source_a}\n")
            f.write(f"Statistica riassuntiva: {config.SPEC_CURVE_SUMMARY_STAT} della differenza"
                    f" auc_pooled\n")
            f.write(f"Valore osservato: {real_diff_stat:+.4f}\n")
            f.write(f"Valore nullo (permutato): {null_diff_stats.mean():+.4f} ±"
                    f" {null_diff_stats.std():.4f}\n")
            f.write(f"p-value empirico: {p_value_diff:.4f}\n")
            f.write(f"CI 95% esatta sul p-value: [{p_ci_low_diff:.4f}, {p_ci_high_diff:.4f}]\n")


if __name__ == "__main__":
    main()