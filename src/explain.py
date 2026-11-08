"""Export reproducible SHAP plots for a saved winner estimator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.utils import MODELS, read_frame


def export(phase: str = "post", variant: str = "results-only") -> None:
    """Explain up to 200 held-out rows and export one local waterfall."""
    bundle = joblib.load(MODELS / f"{variant}_{phase}_best.joblib")
    sample = read_frame(MODELS / f"{variant}_{phase}_best_test.parquet").head(200)
    item = bundle["targets"]["winner"]
    x = item["preprocessor"].transform(sample[bundle["columns"]])
    explanation = shap.TreeExplainer(item["model"])(x)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    explanation.feature_names = list(item["preprocessor"].get_feature_names_out())
    shap.plots.beeswarm(explanation, show=False, max_display=15)
    plt.savefig(MODELS / f"{variant}_{phase}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    shap.plots.waterfall(explanation[0], show=False, max_display=15)
    plt.savefig(MODELS / f"{variant}_{phase}_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Exported SHAP plots: {sample.iloc[0].race_id}, {sample.iloc[0].driver}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pre", "post"], default="post")
    parser.add_argument(
        "--variant", choices=["results-only", "enriched", "observed-weather"], default="results-only"
    )
    args = parser.parse_args()
    export(args.phase, args.variant)
