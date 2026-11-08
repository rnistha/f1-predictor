# Model variant comparison

Identical training/validation/calibration/test events for all three variants. Results-only includes qualifying for post-qualifying models; enriched adds optional timing and availability flags. Observed-weather is a separate hindsight analysis using race-window observations, not a deployable forecast.

```text
         variant phase   family  winner_accuracy  winner_brier  finish_MAE  spearman
    results-only   pre  sklearn         0.368421      0.035191    3.918552  0.642723
    results-only   pre  xgboost         0.157895      0.038816    4.259263  0.564623
    results-only   pre lightgbm         0.315789      0.036568    3.996588  0.647419
    results-only   pre     best         0.368421      0.035191    3.918552  0.642723
    results-only  post  sklearn         0.631579      0.025855    3.841302  0.672145
    results-only  post  xgboost         0.526316      0.027211    3.771382  0.656869
    results-only  post lightgbm         0.473684      0.030766    3.720811  0.662278
    results-only  post     best         0.631579      0.025855    3.720811  0.662278
        enriched   pre  sklearn         0.368421      0.035814    3.980438  0.638682
        enriched   pre  xgboost         0.210526      0.040932    4.385164  0.563672
        enriched   pre lightgbm         0.473684      0.035493    3.875305  0.643377
        enriched   pre     best         0.368421      0.035814    3.980438  0.638682
        enriched  post  sklearn         0.473684      0.030186    3.901837  0.676424
        enriched  post  xgboost         0.631579      0.027722    4.049355  0.588873
        enriched  post lightgbm         0.631579      0.031926    3.796636  0.672323
        enriched  post     best         0.473684      0.030186    3.901837  0.676424
observed-weather   pre  sklearn         0.368421      0.035871    3.930775  0.654730
observed-weather   pre  xgboost         0.210526      0.039019    4.468203  0.523433
observed-weather   pre lightgbm         0.473684      0.035493    3.935255  0.633927
observed-weather   pre     best         0.368421      0.035871    3.930775  0.654730
observed-weather  post  sklearn         0.526316      0.029379    3.845242  0.679872
observed-weather  post  xgboost         0.684211      0.026239    4.034423  0.646587
observed-weather  post lightgbm         0.368421      0.031206    3.792147  0.678623
observed-weather  post     best         0.526316      0.029379    3.845242  0.679872
```

Differences (enriched minus results-only): positive accuracy/Spearman is better; negative loss/MAE/Brier is better.

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

Timing availability fractions: {"has_practice_data": 0.0, "has_weather_data": 0.0, "has_strategy_data": 0.0, "historical_weather": 0.9863261093911249}
