"""
Entry point della specification curve v2 (file NUOVO: run_specification_curve.py
resta invariato). Solo curva descrittiva, nessun test di permutazione.

Uso:
    python run_specification_curve_v2.py
Per ridurre il rumore dello split con n piccolo: main(n_repeats=5).
I risultati vanno in OUTPUT_DIR/reduction_v2/specification_curve/.
"""

import config
import specification_curve as sc
import specification_curve_v2 as sc2


def main(n_repeats: int = 1):
    out_dir = config.OUTPUT_DIR / "reduction_v2" / "specification_curve"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_df = sc2.run_specification_curve_v2(n_repeats=n_repeats)
    spec_df.to_csv(out_dir / "specification_curve_v2_results.csv", index=False)

    spec_keys = list(sc2.SPEC_GRID_V2.keys()) + ["model_type"]
    sc.plot_specification_curve(
        spec_df, spec_keys=spec_keys,
        output_path=out_dir / "specification_curve_v2.png"
    )

    summary = sc2.summarize_by_axis(spec_df, spec_keys)
    summary.to_csv(out_dir / "specification_curve_v2_summary_by_axis.csv", index=False)
    print("\nRIEPILOGO PER ASSE (descrittivo)\n" + summary.to_string(index=False))
    print(f"\nRisultati salvati in: {out_dir}")


if __name__ == "__main__":
    main()
