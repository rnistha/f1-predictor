# Verified evaluation

Training 2018–2023; validation/model selection 2024; independent Platt calibration 2025; test: available 2026 races.

**2026 is reporting-only.** Every target's family is selected by 2024 validation loss; tuning uses expanding windows within 2018–2023. All variants share the same race rows and split.

## Weather coverage and interpretation

Hindsight race-window observations, not deployable pre-race forecasts. A timestamped forecast is required for future races. Track temperature is unsupported and excluded.

Race window: two hourly bins beginning at floor(race start UTC hour). Matched on circuit latitude/longitude and exact UTC date/hour. Air temperature, humidity and wind are averaged; precipitation is summed. weather_is_wet is 1 when the window precipitation is >0, else 0; missing precipitation stays missing.

| Feature | Non-null driver-race rows |
| --- | ---: |
| weather_AirTemp | 100.00% |
| weather_Humidity | 100.00% |
| weather_WindSpeed | 100.00% |
| weather_Rainfall | 100.00% |
| weather_is_wet | 100.00% |
| weather_TrackTemp | 0% — unavailable from archive, excluded |

The 95% coverage gate passes for supported weather fields. Air temperature has not been substituted for track temperature.

Results-only and enriched exclude all current-race observed weather. Enriched uses optional practice/strategy and lagged published weather. Observed-weather adds retrospective race-window weather and must not be interpreted as a genuine pre-race forecast.

## Held-out metrics

| Variant | Phase | Family | Winner accuracy | Winner in top 3 | Brier | Finish MAE | Spearman |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| results-only | pre | sklearn | 36.8% | 68.4% | 0.0352 | 3.92 | 0.643 |
| results-only | pre | xgboost | 15.8% | 68.4% | 0.0388 | 4.26 | 0.565 |
| results-only | pre | lightgbm | 31.6% | 68.4% | 0.0366 | 4.00 | 0.647 |
| results-only | pre | best | 36.8% | 68.4% | 0.0352 | 3.92 | 0.643 |
| results-only | post | sklearn | 63.2% | 94.7% | 0.0259 | 3.84 | 0.672 |
| results-only | post | xgboost | 52.6% | 94.7% | 0.0272 | 3.77 | 0.657 |
| results-only | post | lightgbm | 47.4% | 89.5% | 0.0308 | 3.72 | 0.662 |
| results-only | post | best | 63.2% | 94.7% | 0.0259 | 3.72 | 0.662 |
| enriched | pre | sklearn | 36.8% | 68.4% | 0.0358 | 3.98 | 0.639 |
| enriched | pre | xgboost | 21.1% | 63.2% | 0.0409 | 4.39 | 0.564 |
| enriched | pre | lightgbm | 47.4% | 63.2% | 0.0355 | 3.88 | 0.643 |
| enriched | pre | best | 36.8% | 68.4% | 0.0358 | 3.98 | 0.639 |
| enriched | post | sklearn | 47.4% | 89.5% | 0.0302 | 3.90 | 0.676 |
| enriched | post | xgboost | 63.2% | 89.5% | 0.0277 | 4.05 | 0.589 |
| enriched | post | lightgbm | 63.2% | 94.7% | 0.0319 | 3.80 | 0.672 |
| enriched | post | best | 47.4% | 89.5% | 0.0302 | 3.90 | 0.676 |
| observed-weather | pre | sklearn | 36.8% | 68.4% | 0.0359 | 3.93 | 0.655 |
| observed-weather | pre | xgboost | 21.1% | 68.4% | 0.0390 | 4.47 | 0.523 |
| observed-weather | pre | lightgbm | 47.4% | 63.2% | 0.0355 | 3.94 | 0.634 |
| observed-weather | pre | best | 36.8% | 68.4% | 0.0359 | 3.93 | 0.655 |
| observed-weather | post | sklearn | 52.6% | 94.7% | 0.0294 | 3.85 | 0.680 |
| observed-weather | post | xgboost | 68.4% | 94.7% | 0.0262 | 4.03 | 0.647 |
| observed-weather | post | lightgbm | 36.8% | 84.2% | 0.0312 | 3.79 | 0.679 |
| observed-weather | post | best | 52.6% | 94.7% | 0.0294 | 3.85 | 0.680 |

Best means validation-selected per target, not the best 2026 result. No models are reselected after viewing these metrics.

Actual pole-sitter baseline: 68.4%. Championship baseline: 36.8%.

## Metric differences

Enriched minus results-only (same test races). Positive accuracy/Spearman is better; negative loss/Brier/MAE is better.

```text
phase   family  races  winner_accuracy  winner_in_top3  winner_race_log_loss  winner_brier  finish_MAE  rank_MAE  spearman  pole_proxy_accuracy  championship_baseline_accuracy  winner_log_loss  podium_log_loss  top10_log_loss  dnf_log_loss  pole_sitter_accuracy
  pre  sklearn      0         0.000000        0.000000              0.050601      0.000623    0.061886  0.043062 -0.004042                  0.0                             0.0         0.002724         0.001710        0.000523     -0.002422                   0.0
  pre  xgboost      0         0.052632       -0.052632              0.014182      0.002116    0.125902  0.052632 -0.000951                  0.0                             0.0         0.002519         0.008754        0.007796      0.000904                   0.0
  pre lightgbm      0         0.157895       -0.052632             -0.149262     -0.001075   -0.121283  0.138756 -0.004042                  0.0                             0.0        -0.007397         0.004328       -0.001522      0.029723                   0.0
  pre     best      0         0.000000        0.000000              0.050601      0.000623    0.061886  0.043062 -0.004042                  0.0                             0.0         0.002724         0.001710        0.000523     -0.002422                   0.0
 post  sklearn      0        -0.157895       -0.052632              0.219630      0.004331    0.060535  0.023923  0.004279                  0.0                             0.0         0.013244         0.012085        0.036523     -0.021012                   0.0
 post  xgboost      0         0.105263       -0.052632              0.037068      0.000511    0.277973  0.406699 -0.067996                  0.0                             0.0         0.002897         0.004461        0.012895     -0.004412                   0.0
 post lightgbm      0         0.157895        0.052632              0.180143      0.001159    0.075824 -0.028708  0.010045                  0.0                             0.0         0.009353        -0.007749        0.000756     -0.033574                   0.0
 post     best      0        -0.157895       -0.052632              0.219630      0.004331    0.181025 -0.100478  0.014146                  0.0                             0.0         0.013244         0.012085        0.014237      0.008571                   0.0
```

Observed-weather minus results-only is a hindsight comparison, not evidence of live forecast improvement.

```text
phase   family  races  winner_accuracy  winner_in_top3  winner_race_log_loss  winner_brier  finish_MAE  rank_MAE  spearman  pole_proxy_accuracy  championship_baseline_accuracy  winner_log_loss  podium_log_loss  top10_log_loss  dnf_log_loss  pole_sitter_accuracy
  pre  sklearn      0         0.000000        0.000000              0.100345      0.000680    0.012223 -0.062201  0.012006                  0.0                             0.0         0.005038        -0.000193        0.005696     -0.012017                   0.0
  pre  xgboost      0         0.052632        0.000000             -0.063915      0.000203    0.208941  0.267943 -0.041190                  0.0                             0.0        -0.002437        -0.011506       -0.003219      0.006597                   0.0
  pre lightgbm      0         0.157895       -0.052632             -0.149198     -0.001075   -0.061333  0.052632 -0.013492                  0.0                             0.0        -0.007394         0.001675        0.002736      0.023146                   0.0
  pre     best      0         0.000000        0.000000              0.100345      0.000680    0.012223 -0.062201  0.012006                  0.0                             0.0         0.005038        -0.000193        0.022571      0.039539                   0.0
 post  sklearn      0        -0.105263        0.000000              0.107238      0.003525    0.003940  0.033493  0.007727                  0.0                             0.0         0.008406         0.012242        0.024228     -0.021729                   0.0
 post  xgboost      0         0.157895        0.000000              0.020726     -0.000972    0.263041  0.023923 -0.010283                  0.0                             0.0         0.000074         0.004310        0.024655     -0.056626                   0.0
 post lightgbm      0        -0.105263       -0.052632              0.107288      0.000440    0.071336 -0.114833  0.016345                  0.0                             0.0         0.004645        -0.009890        0.000799     -0.036708                   0.0
 post     best      0        -0.105263        0.000000              0.107238      0.003525    0.124430 -0.090909  0.017593                  0.0                             0.0         0.008406         0.012242        0.026095     -0.036708                   0.0
```

## Provenance and limitations

```json
{
  "created_at": "2026-11-08T21:14:17.318846+00:00",
  "feature_sha256": "f5721776c49a8f11379b6c0f4c2dd0950a2b083bdc00d25919dde0eae2b3d6d1",
  "rows": 3876,
  "races": 192,
  "trials_per_family": 5,
  "timing_sessions": 0,
  "session_checkpoints": 124,
  "weather_sessions": 124,
  "test_race_ids": [
    "2026_01",
    "2026_02",
    "2026_03",
    "2026_04",
    "2026_05",
    "2026_06",
    "2026_07",
    "2026_08",
    "2026_09",
    "2026_10",
    "2026_11",
    "2026_12",
    "2026_13",
    "2026_14",
    "2026_15",
    "2026_16",
    "2026_17",
    "2026_18",
    "2026_19"
  ],
  "versions": {
    "fastf1": "3.9.2",
    "scikit-learn": "1.9.1",
    "xgboost": "3.4.1",
    "lightgbm": "4.7.0",
    "pandas": "2.3.3",
    "numpy": "2.5.4"
  }
}
```

Availability fractions: {"has_practice_data": 0.0, "has_weather_data": 0.0, "has_strategy_data": 0.0, "historical_weather": 0.9863261093911249}

FastF1 timing remains unavailable (HTTP 401; mirror 404). Archive reanalysis is gridded historical weather, not a circuit sensor. Precipitation is a wetness proxy, not measured track wetness. Models have a small 2026 test sample and a major regulation shift. Provider revisions and reconstructed standings remain limitations. See [README](README.md).

Full precision: [evaluation JSON](models/evaluation.json); [metrics CSV](models/variant_metrics.csv).
