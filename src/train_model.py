"""Chronological gradient-boosting comparison, calibration and held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.feature_engineering import CATEGORICAL, build_dataset, feature_columns
from src.utils import DATA, LOG, MODELS, save_frame, setup
from src.weather_features import WEATHER_FEATURES

TARGETS = ["winner", "podium", "top10", "dnf", "finish"]
FAMILIES = ["sklearn", "xgboost", "lightgbm"]


def preprocessor(columns: list[str]) -> ColumnTransformer:
    """Fit ordinal vocabulary and imputation on the training period only."""
    numeric = [c for c in columns if c not in CATEGORICAL]
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                numeric,
            ),
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def estimator(family: str, target: str, params: dict, y: pd.Series, early: bool = True):
    """Construct a deterministic CPU estimator with class weighting where supported."""
    regression = target == "finish"
    common = {
        "n_estimators": params.get("n_estimators", 250),
        "learning_rate": params.get("learning_rate", 0.05),
        "max_depth": params.get("max_depth", 3),
        "subsample": params.get("subsample", 0.85),
        "random_state": 42,
    }
    if family == "sklearn":
        cls = GradientBoostingRegressor if regression else GradientBoostingClassifier
        return cls(**common)
    extra = {
        k: params.get(k, default)
        for k, default in [
            ("colsample_bytree", 0.9),
            ("min_child_weight", 5),
            ("reg_alpha", 0.1),
            ("reg_lambda", 2),
        ]
    }
    if not regression:
        extra["scale_pos_weight"] = float((y == 0).sum() / max((y == 1).sum(), 1))
    if family == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBRegressor if regression else XGBClassifier
        return cls(
            **common,
            **extra,
            n_jobs=2,
            tree_method="hist",
            early_stopping_rounds=25 if early else None,
            eval_metric="mae" if regression else "logloss",
        )
    from lightgbm import LGBMClassifier, LGBMRegressor

    cls = LGBMRegressor if regression else LGBMClassifier
    return cls(
        **common,
        **extra,
        n_jobs=2,
        verbosity=-1,
        subsample_freq=1,
        importance_type="gain",
        num_leaves=min(31, 2 ** common["max_depth"]),
    )


def fit_estimator(
    family: str, target: str, params: dict, x: np.ndarray, y: pd.Series, xv: np.ndarray, yv: pd.Series
):
    """Use chronological validation for stopping, including the sklearn baseline."""
    model = estimator(family, target, params, y)
    if family == "sklearn":
        model.fit(x, y)
        stages = model.staged_predict(xv) if target == "finish" else model.staged_predict_proba(xv)
        losses = [
            mean_absolute_error(yv, p) if target == "finish" else log_loss(yv, p, labels=[0, 1])
            for p in stages
        ]
        params = {**params, "n_estimators": int(np.argmin(losses)) + 1}
        model = estimator(family, target, params, y, early=False).fit(x, y)
    elif family == "xgboost":
        model.fit(x, y, eval_set=[(xv, yv)], verbose=False)
    else:
        from lightgbm import early_stopping, log_evaluation

        model.fit(x, y, eval_set=[(xv, yv)], callbacks=[early_stopping(25, verbose=False), log_evaluation(0)])
    return model


def tune(family: str, train: pd.DataFrame, columns: list[str], trials: int) -> dict:
    """Optuna expanding-season CV on winner loss, wholly inside training years."""
    years = sorted(train.year.unique())
    folds = years[-2:]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.65, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
        }
        scores = []
        for year in folds:
            a, b = train[train.year < year], train[train.year == year]
            prep = preprocessor(columns)
            xa, xb = prep.fit_transform(a[columns]), prep.transform(b[columns])
            model = fit_estimator(family, "winner", params, xa, a.winner, xb, b.winner)
            scores.append(log_loss(b.winner, model.predict_proba(xb)[:, 1], labels=[0, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    return study.best_params


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Calibrate binary outputs, normalize winners, and rank finishing scores."""
    result = frame.copy()
    for target, item in bundle["targets"].items():
        x = item["preprocessor"].transform(frame[bundle["columns"]])
        if target == "finish":
            result["finish_score"] = item["model"].predict(x)
        else:
            raw = item["model"].predict_proba(x)[:, 1]
            calibrated = item["calibrator"].predict_proba(logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1))[
                :, 1
            ]
            result[f"{target}_probability"] = calibrated
    result["winner_probability"] /= result.groupby("race_id").winner_probability.transform("sum")
    result["podium_probability"] = np.maximum(result.podium_probability, result.winner_probability)
    result["top10_probability"] = np.maximum(result.top10_probability, result.podium_probability)
    result["predicted_position"] = result.groupby("race_id").finish_score.rank(method="first").astype(int)
    return result


def evaluate(pred: pd.DataFrame) -> dict:
    """Race-level hit rates and driver-level proper scores on held-out events."""
    groups = list(pred.groupby("race_id"))
    winner_accuracy = np.mean([g.loc[g.winner_probability.idxmax(), "winner"] for _, g in groups])
    top3_hit = np.mean([g.nlargest(3, "winner_probability").winner.sum() for _, g in groups])
    race_loss = np.mean(
        [-np.log(max(g.loc[g.winner == 1, "winner_probability"].sum(), 1e-12)) for _, g in groups]
    )
    metrics = {
        "races": len(groups),
        "winner_accuracy": float(winner_accuracy),
        "winner_in_top3": float(top3_hit),
        "winner_race_log_loss": float(race_loss),
        "winner_brier": float(brier_score_loss(pred.winner, pred.winner_probability)),
        "finish_MAE": float(mean_absolute_error(pred.finish, pred.finish_score)),
        "rank_MAE": float(mean_absolute_error(pred.finish, pred.predicted_position)),
        "spearman": float(
            np.mean([g.finish.corr(g.predicted_position, method="spearman") for _, g in groups])
        ),
        "pole_proxy_accuracy": float(
            np.mean([g.sort_values(["grid", "driver"]).iloc[0].winner for _, g in groups])
        ),
        "championship_baseline_accuracy": float(
            np.mean([g.sort_values(["championship_position", "driver"]).iloc[0].winner for _, g in groups])
        ),
    }
    for target in ("winner", "podium", "top10", "dnf"):
        metrics[f"{target}_log_loss"] = float(
            log_loss(pred[target], pred[f"{target}_probability"], labels=[0, 1])
        )
    if "actual_grid" in pred:
        metrics["pole_sitter_accuracy"] = float(
            np.mean(
                [g.loc[g.actual_grid.where(g.actual_grid > 0, np.inf).idxmin(), "winner"] for _, g in groups]
            )
        )
    return metrics


def train(trials: int = 5) -> dict:
    """Reserve final three seasons for validation, calibration and testing."""
    setup()
    frame = build_dataset()
    weather_coverage = {col: float(frame[col].notna().mean()) for col in WEATHER_FEATURES}
    if any(value < 0.95 for value in weather_coverage.values()):
        raise ValueError(
            "Observed weather coverage below 95%; run python src/weather_features.py --download first."
        )
    # Only complete classifications with exactly one winner are usable.
    frame = frame.groupby("race_id").filter(lambda g: len(g) >= 15 and g.winner.sum() == 1)
    years = sorted(frame.year.unique())
    if len(years) < 6:
        raise ValueError(
            "At least six seasons required for chronological CV and independent calibration/test."
        )
    validation, calibration, testing = map(int, years[-3:])
    train_df = frame[frame.year < validation]
    val_df, cal_df, test_df = [frame[frame.year == y] for y in (validation, calibration, testing)]
    report = {
        "split": {
            "train": sorted(map(int, train_df.year.unique())),
            "validation": validation,
            "calibration": calibration,
            "test": testing,
        },
        "models": [],
        "best": {},
    }
    report["provenance"] = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "feature_sha256": hashlib.sha256((DATA / "features.parquet").read_bytes()).hexdigest(),
        "rows": len(frame),
        "races": int(frame.race_id.nunique()),
        "trials_per_family": trials,
        "timing_sessions": len(list((DATA / "sessions").glob("*/laps.parquet"))),
        "session_checkpoints": len(list((DATA / "sessions").glob("*/summary.parquet"))),
        "weather_sessions": len(list((DATA / "sessions").glob("*/weather_data.parquet"))),
        "test_race_ids": sorted(test_df.race_id.unique().tolist()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("fastf1", "scikit-learn", "xgboost", "lightgbm", "pandas", "numpy")
        },
    }
    for variant, phase in [
        (v, p) for v in ("results-only", "enriched", "observed-weather") for p in ("pre", "post")
    ]:
        columns = feature_columns(phase, variant)
        candidates = {}
        for family in FAMILIES:
            LOG.info("Training %s / %s / %s", variant, phase, family)
            params = tune(family, train_df, columns, trials)
            bundle = {
                "phase": phase,
                "variant": variant,
                "uses_observed_race_weather": variant == "observed-weather",
                "family": family,
                "columns": columns,
                "targets": {},
                "calibration_year": calibration,
                "test_year": testing,
                "trained_through": str(cal_df.race_date.max()),
                "params": params,
            }
            bundle["provenance"] = report["provenance"]
            selection_losses = {}
            for target in TARGETS:
                prep = preprocessor(columns)
                x = prep.fit_transform(train_df[columns])
                xv, xc = prep.transform(val_df[columns]), prep.transform(cal_df[columns])
                model = fit_estimator(family, target, params, x, train_df[target], xv, val_df[target])
                item = {"preprocessor": prep, "model": model}
                if target == "finish":
                    selection_losses[target] = float(mean_absolute_error(val_df[target], model.predict(xv)))
                else:
                    selection_losses[target] = float(
                        log_loss(val_df[target], model.predict_proba(xv), labels=[0, 1])
                    )
                    raw = model.predict_proba(xc)[:, 1]
                    item["calibrator"] = LogisticRegression(C=1.0, random_state=42).fit(
                        logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1), cal_df[target]
                    )
                bundle["targets"][target] = item
            bundle["selection_losses"] = selection_losses
            joblib.dump(bundle, MODELS / f"{variant}_{phase}_{family}.joblib")
            candidates[family] = bundle
            pred = predict_bundle(bundle, test_df)
            metrics = evaluate(pred)
            report["models"].append({"variant": variant, "phase": phase, "family": family, **metrics})
            save_frame(pred, MODELS / f"{variant}_{phase}_{family}_test.parquet")
        # Model selection is validation-only, separately for each requested target.
        winners = {
            target: min(candidates, key=lambda f: candidates[f]["selection_losses"][target])
            for target in TARGETS
        }
        best = {
            **candidates[winners["winner"]],
            "family": "best",
            "targets": {target: candidates[family]["targets"][target] for target, family in winners.items()},
            "selected_families": winners,
        }
        for target, item in best["targets"].items():
            joblib.dump({"columns": columns, **item}, MODELS / f"{variant}_{phase}_{target}_best.joblib")
        joblib.dump(best, MODELS / f"{variant}_{phase}_best.joblib")
        pred = predict_bundle(best, test_df)
        save_frame(pred, MODELS / f"{variant}_{phase}_best_test.parquet")
        report["models"].append({"variant": variant, "phase": phase, "family": "best", **evaluate(pred)})
        report["best"][f"{variant}_{phase}"] = winners
        item = best["targets"]["winner"]
        names = item["preprocessor"].get_feature_names_out()
        save_frame(
            pd.DataFrame({"feature": names, "importance": item["model"].feature_importances_}).sort_values(
                "importance", ascending=False
            ),
            MODELS / f"{variant}_{phase}_importance.parquet",
        )
    # Both variants use exactly the same events, labels and chronological split.
    metrics = pd.DataFrame(report["models"])
    left = metrics[metrics.variant == "results-only"].set_index(["phase", "family"])
    right = metrics[metrics.variant == "enriched"].set_index(["phase", "family"])
    numeric = list(left.select_dtypes(include="number").columns)
    difference = (right[numeric] - left[numeric]).reset_index()
    report["variant_difference_enriched_minus_results_only"] = difference.to_dict(orient="records")
    observed = metrics[metrics.variant == "observed-weather"].set_index(["phase", "family"])
    observed_difference = (observed[numeric] - left[numeric]).reset_index()
    report["observed_difference_minus_results_only"] = observed_difference.to_dict(orient="records")
    report["weather_coverage"] = weather_coverage
    report["observed_weather_warning"] = (
        "Hindsight race-window observations, not deployable pre-race forecasts. A timestamped forecast is required for future races. Track temperature is unsupported and excluded."
    )
    observed_difference.to_csv(MODELS / "observed_weather_difference.csv", index=False)
    report["availability"] = {
        name: float(frame[name].mean())
        for name in ("has_practice_data", "has_weather_data", "has_strategy_data")
    }
    report["availability"]["historical_weather"] = float(
        frame[["history_AirTemp", "history_Humidity", "history_WindSpeed", "history_Precipitation"]]
        .notna()
        .any(axis=1)
        .mean()
    )
    metrics.to_csv(MODELS / "variant_metrics.csv", index=False)
    difference.to_csv(MODELS / "variant_difference.csv", index=False)
    (MODELS / "variant_comparison.md").write_text(
        "# Model variant comparison\n\nIdentical training/validation/calibration/test events for all three variants. "
        "Results-only includes qualifying for post-qualifying models; enriched adds optional timing and availability flags. Observed-weather is a separate hindsight analysis using race-window observations, not a deployable forecast.\n\n"
        + "```text\n"
        + metrics[
            ["variant", "phase", "family", "winner_accuracy", "winner_brier", "finish_MAE", "spearman"]
        ].to_string(index=False)
        + "\n```\n\n"
        + "Differences (enriched minus results-only): positive accuracy/Spearman is better; negative loss/MAE/Brier is better.\n\n```text\n"
        + difference.to_string(index=False)
        + "\n```\n\nTiming availability fractions: "
        + json.dumps(report["availability"])
        + "\n"
    )
    (MODELS / "evaluation.json").write_text(json.dumps(report, indent=2))
    write_evaluation(report, metrics, difference, observed_difference)
    print(json.dumps(report, indent=2))
    return report


def write_evaluation(
    report: dict, metrics: pd.DataFrame, difference: pd.DataFrame, observed_difference: pd.DataFrame
) -> None:
    """Publish all test results without using them to change selection decisions."""
    lines = [
        "# Verified evaluation",
        "",
        "Training 2018–2023; validation/model selection 2024; independent Platt calibration 2025; test: available 2026 races.",
        "",
        "**2026 is reporting-only.** Every target's family is selected by 2024 validation loss; tuning uses expanding windows within 2018–2023. All variants share the same race rows and split.",
        "",
        "## Weather coverage and interpretation",
        "",
        report["observed_weather_warning"],
        "",
        "Race window: two hourly bins beginning at floor(race start UTC hour). Matched on circuit latitude/longitude and exact UTC date/hour. Air temperature, humidity and wind are averaged; precipitation is summed. weather_is_wet is 1 when the window precipitation is >0, else 0; missing precipitation stays missing.",
        "",
        "| Feature | Non-null driver-race rows |",
        "| --- | ---: |",
    ]
    lines += [f"| {name} | {value:.2%} |" for name, value in report["weather_coverage"].items()]
    lines += [
        "| weather_TrackTemp | 0% — unavailable from archive, excluded |",
        "",
        "The 95% coverage gate passes for supported weather fields. Air temperature has not been substituted for track temperature.",
        "",
        "Results-only and enriched exclude all current-race observed weather. Enriched uses optional practice/strategy and lagged published weather. Observed-weather adds retrospective race-window weather and must not be interpreted as a genuine pre-race forecast.",
        "",
        "## Held-out metrics",
        "",
        "| Variant | Phase | Family | Winner accuracy | Winner in top 3 | Brier | Finish MAE | Spearman |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row.variant} | {row.phase} | {row.family} | {row.winner_accuracy:.1%} | {row.winner_in_top3:.1%} | {row.winner_brier:.4f} | {row.finish_MAE:.2f} | {row.spearman:.3f} |"
        )
    lines += [
        "",
        "Best means validation-selected per target, not the best 2026 result. No models are reselected after viewing these metrics.",
        "",
        f"Actual pole-sitter baseline: {metrics.iloc[0].pole_sitter_accuracy:.1%}. Championship baseline: {metrics.iloc[0].championship_baseline_accuracy:.1%}.",
        "",
        "## Metric differences",
        "",
        "Enriched minus results-only (same test races). Positive accuracy/Spearman is better; negative loss/Brier/MAE is better.",
        "",
        "```text",
        difference.to_string(index=False),
        "```",
        "",
        "Observed-weather minus results-only is a hindsight comparison, not evidence of live forecast improvement.",
        "",
        "```text",
        observed_difference.to_string(index=False),
        "```",
        "",
        "## Provenance and limitations",
        "",
        "```json",
        json.dumps(report["provenance"], indent=2),
        "```",
        "",
        "Availability fractions: " + json.dumps(report["availability"]),
        "",
        "FastF1 timing remains unavailable (HTTP 401; mirror 404). Archive reanalysis is gridded historical weather, not a circuit sensor. Precipitation is a wetness proxy, not measured track wetness. Models have a small 2026 test sample and a major regulation shift. Provider revisions and reconstructed standings remain limitations. See [README](README.md).",
        "",
        "Full precision: [evaluation JSON](models/evaluation.json); [metrics CSV](models/variant_metrics.csv).",
    ]
    (MODELS.parent / "EVALUATION.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    train(args.trials)
