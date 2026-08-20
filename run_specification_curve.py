import config
import specification_curve as sc

def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spec_df, feature_votes = sc.run_specification_curve()

    spec_df.to_csv(config.OUTPUT_DIR / "specification_curve_results.csv", index=False)
    feature_votes.to_csv(config.OUTPUT_DIR / "feature_votes_across_specs.csv",
                          header=["n_specs_selected"])

    sc.plot_specification_curve(
        spec_df, spec_keys=list(sc.SPEC_GRID.keys()),
        output_path=config.OUTPUT_DIR / "specification_curve.png"
    )
    sc.plot_feature_votes(
        feature_votes, n_total_specs=len(spec_df),
        output_path=config.OUTPUT_DIR / "feature_votes_across_specs.png"
    )

if __name__ == "__main__":
    main()