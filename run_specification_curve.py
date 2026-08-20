import config
import specification_curve as sc

def main():
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

if __name__ == "__main__":
    main()