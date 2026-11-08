# F1 Race Lab

A Python/FastF1 project for driver win, podium, top-10 and non-finish probabilities,
plus a complete finishing-order forecast and a seven-tab Streamlit explorer.
Models are trained on real downloaded results; no synthetic fallback data is used.

## Run locally

Python 3.10+ is required (this installation uses a project-local Python 3.12).

```bash
cd '/Users/nistharajbhandari/Documents/f1 prediction/f1_predictor'
source .venv/bin/activate
pip install -r requirements.txt
python src/data_collection.py
python src/weather_features.py --download
python src/train_model.py
streamlit run app.py --server.address 127.0.0.1
```

For a new checkout, create the environment first with `python3 -m venv .venv`.
On macOS, XGBoost/LightGBM require OpenMP (`brew install libomp`). This machine
has no Apple developer tools or Homebrew, so Python and an ARM OpenMP runtime
were installed inside the parent project directory instead; no system Python
was changed. Moving this virtual environment requires recreating it.

The default collector attempts all results and timing sessions from 2018 through
the current year. This can be slow and depends on upstream timing access. To get
started with results and qualifying only:

```bash
python src/data_collection.py --mode core
python src/weather_features.py --download
python src/train_model.py --trials 5
python src/predict.py --year 2026 --round 19
streamlit run app.py
```

To resume timing downloads or fetch one event:

```bash
python src/data_collection.py --mode sessions --start 2026 --end 2026 --rounds 19
python src/data_collection.py --mode full --start 2018 --end 2026
```

Successful sessions and completed historical seasons are reused. `--refresh`
refreshes season result/schedule files; the current season is refreshed by default.
Each timing session is appended to `data/collection_log.csv` as `success`, `missing`, or `failed`, with lap/weather coverage, source, timestamp and error details. Partial checkpoints retain driver rows and do not prevent retries. Season-level errors remain in JSON. FastF1 caching is enabled in `cache/`. Do not run multiple
collectors or trainers against the same output files simultaneously.


## Hardened collection and model variants

```bash
# Diagnose DNS/proxy restrictions separately from HTTP authentication or missing feeds.
python src/data_collection.py --check-network
# Laps + weather only: no telemetry or race-control messages.
python src/data_collection.py --mode sessions --start 2026 --end 2026 --rounds 19 --attempts 3 --backoff 2 --session-delay 2
# Retry only sessions whose latest CSV entry is missing or failed (includes partial sessions).
python src/data_collection.py --mode sessions --retry-failed --start 2018 --end 2026
# Recover weather without repeatedly requesting an inaccessible lap feed.
python src/data_collection.py --mode sessions --weather-only --start 2018 --end 2026 --rounds 1 10 19
# Train BOTH variants on the same event rows and chronological split.
python src/weather_features.py --download
python src/train_model.py --trials 5
python src/predict.py --year 2026 --round 19 --variant results-only
python src/predict.py --year 2026 --round 19 --variant enriched
```

`--check-network` exits 0 only if every probe succeeds, otherwise 1, and performs
no collection. It probes schedule, results, live timing, timing mirror and weather
archive endpoints separately. HTTP 401/403/404 identifies an endpoint refusal or
missing resource, not proof that all networking is blocked.

Retries use exponential delays (default 2, 4 seconds for three attempts). HTTP 429
honors numeric/date `Retry-After`; FastF1's rolling hourly limit pauses 3,601 seconds
and resumes the same session without clearing rate counters. The wait is logged
and interruptible in 30-second chunks. At most three rate-limit pauses occur per
operation by default; exhausted attempts remain recoverable in the CSV. FastF1
soft-failure logs are monitored for swallowed rate-limit exceptions. Use Ctrl-C
to interrupt a long backfill, then resume from checkpoints. Only one collector
should write the dataset at a time.

Success means laps, weather and a driver summary were retained. Missing means an
unavailable/not-yet-completed session or partial coverage. Failed means a transport,
API or persistence error. A session with recovered weather but no laps remains
`missing`, so `--retry-failed` will revisit it. The append-only CSV preserves attempt
history; the latest status controls resumption. Cached good observations are not
replaced by NaNs on a failed retry. Missing practice laps retain every known driver
with NaN pace and `has_practice_data=0`.

Open-Meteo uses circuit latitude/longitude, the session's UTC date/time (race time
for Race), and hourly air temperature, relative humidity, wind in m/s,
precipitation in mm, and surface pressure. Data is cached by location and day;
track temperature stays NaN. Session summaries store `weather_source` and
`weather_available_at`. This is **reanalysis, not a forecast or track sensor**.
A conservative seven-day publication delay prevents it from becoming current-race
practice weather. Once that cutoff has passed, it can enter previous-race weather
history. Reanalysis may be revised; the delay is a conservative modeling convention,
not a guarantee of historical publication timestamps. No current-race observed
weather or strategy enters either pre-race model. The separate observed-weather variant below is explicitly retrospective.

- **results-only**: identity/context, prior results/form/history, plus qualifying
  in post-qualifying mode. Excludes practice, strategy, weather and availability flags.
- **enriched**: the same base plus optional timing, historical weather and explicit
  practice/weather/strategy availability flags. Missing values never drop race rows.

Artifacts are named `{variant}_{phase}_{family}.joblib`; best models are selected
within each variant using validation data. The Streamlit sidebar selects the
variant explicitly, and the prediction displays the actual saved variant used.
There is no silent model substitution. The performance tab reports all variants
and `enriched - results-only` differences on identical held-out races. See
[comparison](models/variant_comparison.md), [metrics CSV](models/variant_metrics.csv),
and [differences CSV](models/variant_difference.csv). Older unprefixed artifacts
are retained for reference and are no longer loaded by prediction or the app.

Weather attribution: [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api)
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).


## Observed race-weather analysis

```bash
# Use cached archive files, downloading missing circuit/date combinations.
python src/weather_features.py --download
# Rebuild features and retrain all three variants with identical time splits.
python src/train_model.py --trials 5
python src/predict.py --year 2026 --round 19 --variant observed-weather
```

[weather_features.py](src/weather_features.py) matches each race to archive data by
circuit latitude/longitude and the UTC race start hour. The fixed window is the
start-hour bin and the following hour, independent of actual race duration. The
resulting event table is [race_weather.parquet](data/race_weather.parquet).
`weather_AirTemp`, `weather_Humidity` and `weather_WindSpeed` are means; `weather_Rainfall`
(and its explicit-unit alias `weather_Precipitation`) is summed precipitation in mm.
`weather_is_wet` is 1 when precipitation is positive; missing precipitation is not
labeled dry. **Rainfall is a precipitation proxy, not observed track wetness.**

The old `fp_AirTemp`, `fp_TrackTemp`, `fp_Humidity`, `fp_WindSpeed`, `fp_Rainfall`
names are removed. Optional native practice-weather fields have a separate
`practice_weather_*` prefix. **Open-Meteo cannot supply track-surface temperature**:
`weather_TrackTemp` remains NaN and is excluded from all model allowlists.

[weather_coverage.json](data/weather_coverage.json) reports row-weighted non-null
coverage. Training fails if any supported observed-weather input has coverage
below 95%. The unsupported track-temperature field is explicitly reported as an
exception, never filled with air temperature or a guessed value.

Observed-weather adds race-window values to the enriched model. **These values
were not known before the race.** Its results are a hindsight analysis, labeled
in the app and [EVALUATION.md](EVALUATION.md), and are not evidence of real-time
forecast quality. Prediction refuses future races for this variant. A production
forecast version would need weather forecasts issued before the information cutoff
and archived forecast vintages for honest training/evaluation. Results-only and
enriched remain available for pre-race use; enriched may use weather from earlier
races only after its conservative publication cutoff.

2026 remains a test-only season for every variant: no Optuna search, early stopping,
calibration or family selection uses its outcomes. Weather coverage checks inspect
missingness, not test labels. Models are not changed in response to test metrics.

## Files

| File | Responsibility |
| --- | --- |
| [src/data_collection.py](src/data_collection.py) | Paginated FastF1 Ergast/Jolpica results, schedules, timing and weather; atomic parquet checkpoints |
| [src/feature_engineering.py](src/feature_engineering.py) | Event-level prior-history features and explicit model allowlists |
| [src/weather_features.py](src/weather_features.py) | Race-hour matching, coverage validation and retrospective weather features |
| [src/collection_support.py](src/collection_support.py) | Rate-aware retries, network probes and archive fallback |
| [src/metadata.py](src/metadata.py) | Circuit character and season-aware engine supplier annotations |
| [src/train_model.py](src/train_model.py) | Three boosting families, expanding-season Optuna CV, early stopping, calibration, test metrics |
| [src/predict.py](src/predict.py) | `predict_race(year, round)`, pre/post qualifying selection, historical cutoff checks |
| [src/explain.py](src/explain.py) | Export winner-model SHAP summary and per-driver waterfall plots |
| [src/utils.py](src/utils.py) | Paths, cache, retry, atomic persistence |
| [app.py](app.py) | Prediction, actual comparison, form, weather, circuit, pace and performance tabs |
| [tests/](tests/) | Leakage, chronology, probability, ranking and app integration checks |

`data/raw/` retains season checkpoints. `data/sessions/` stores laps, results,
weather and summaries per timing session. Lap track-status codes remain in the lap table; race-control messages are not requested.
`data/features.parquet` contains both features and labels, but **only the explicit
allowlist enters the estimator**. Observed `weather_*` columns are allowlisted only for the observed-weather variant; `weather_TrackTemp` is never modeled. `models/` contains all family/phase bundles,
the best artifact for each target, test predictions, importance and evaluation JSON.
Models are local trusted joblib files; never load untrusted pickle/joblib files.

## Features and availability

- Driver and team identities, circuit, country, coarse circuit type and power-unit supplier.
- Year, round and separate pre-2022, 2022–2025 and 2026 regulation eras.
- Qualifying Q1/Q2/Q3 observations; Q1-to-pole and Q1-to-best-teammate gaps.
  Q1 is compared with Q1 to avoid comparing different qualifying phases.
- Qualifying position is the **grid proxy**. Final penalties, pit-lane starts and
  late grid changes are not supplied by a reliable pre-race grid feed here.
- Prior 3/5-race finish, points, grid, positions gained and non-finish rate.
- Driver/team circuit finish history; driver wins and podiums; historical
  pole conversion; absolute positions gained as an imperfect overtaking proxy.
- Prior-round season points, including sprint points, and entrant-relative ranks.
  These are reconstructed standings: current-weekend sprint points, official
  countback and off-track point adjustments are not modeled. The app's standings
  charts explicitly show Grand Prix points only.
- Team reliability, mechanical-status keyword rate, performance trend, teammate
  form comparison, age and observed race experience since 2018.
- When timing is available: relative practice pace, sector medians, long-run
  rank and practice weather; lagged race pit/stint/tyre and interruption summaries.

Long-run pace is a median of clean laps in stints with at least five valid laps,
after a 112%-of-best-lap filter. It is **not measured race-fuel pace**: fuel loads,
engine settings and run plans are unknown. Pit duration includes pit-lane transit,
not stationary service time. SC/VSC/red-flag summaries count transitions in sampled
lap track-status codes, so short interruptions may be missed.

Current-race weather, strategy and lap times are excluded from the deployable results-only and enriched models. The explicitly retrospective observed-weather model uses observed race-window weather. The weather tab may show retrospective race weather, clearly labeled.
There is no external forecast feed. The separate observed-weather analysis uses retrospective race-window observations and is not a pre-race forecast. Practice features use sessions before
qualifying starts, even on older sprint weekends where FP2 followed qualifying.
Unknown/missing inputs remain NaN until training-only median imputation; missingness
indicators and unknown ordinal categories are supported. Entirely missing features
are retained but cannot teach the model anything. Tree models need no scaling.

For future events, qualifying or practice must confirm entrants. The application
does not invent a future entry list. Without qualifying it selects a separately
trained pre-qualifying model. To force that information set for a completed race:

```bash
python src/predict.py --year 2026 --round 19 --phase pre
```

## Modeling and evaluation

Both information sets and all three data variants compare scikit-learn GradientBoosting, XGBoost and LightGBM.
Five targets are trained: winner, podium, top-10, non-finish and finishing-position
regression. A unique finishing order is obtained by sorting regression scores;
ties are broken by driver identifier, never actual classification order.
The most likely winner and the first driver in the regression ranking can differ.

With the downloaded seasons, splits are:

| Purpose | Seasons |
| --- | --- |
| Training | 2018–2023 |
| Hyperparameter CV | Expanding training windows validating on 2022 and 2023 |
| Family selection and early stopping | 2024 |
| Independent Platt calibration | 2025 |
| Untouched test | Available 2026 races |

Optuna tunes winner log loss for each family/phase over the requested parameter
ranges, with 5 trials in the verified run. That parameter configuration is reused
across targets to bound compute. Each target has separate chronological early
stopping and validation-based family selection. Increase `--trials` for a broader
search. Sklearn stopping explicitly checks staged predictions on the validation
season instead of using its random internal validation split. XGBoost/LightGBM use
their native stopping APIs. Binary class weighting is used for those two libraries;
the sklearn model is an unweighted baseline.

Platt calibration uses only 2025. Winner probabilities are normalized per race to
sum to one. Podium/top-10 probabilities are clipped upward to satisfy
`P(win) <= P(podium) <= P(top10)`. These adjustments can change calibration;
the reported test metrics and calibration plots evaluate the final adjusted values.
Podium/top-10 marginal probabilities are not constrained to sum to 3/10.

Metrics include race winner accuracy, winner-in-top-3 hit rate, categorical race
log loss, binary log loss/Brier, Spearman ranking correlation, regression and rank
MAE, actual pole-sitter and qualifying-proxy baselines, and a reconstructed
championship-leader baseline. Actual grid is used only in the evaluation baseline.
The test season is never used to select the “best” model. Consequently another
family may perform better on the displayed test season.

Current variant results and coverage are recorded in [models/variant_comparison.md](models/variant_comparison.md),
`models/evaluation.json`, and the Model Performance tab. The JSON includes the
feature-file SHA-256, dependency versions, training timestamp and split.
No automatic refit on test labels is performed. Prediction rejects events at or
before the calibration cutoff; earlier historical backtests need separately
trained earlier-cutoff models.

SHAP and native feature importance explain the **uncalibrated** winner estimator,
not the normalized race probabilities. XGBoost and LightGBM importance uses gain;
sklearn uses impurity reduction. Generate exported plots with:

```bash
python src/explain.py --phase post
python -m pytest -q
ruff check src app.py tests
```

## Limitations and sources

F1 outcomes are high-variance; probabilities are not guarantees. A 19-race test
sample is small. The 2026 regulation and power-unit changes are a major distribution
shift, and tree models cannot extrapolate the consequences of an unseen era label.
Driver experience is left-truncated at 2018; team rebrands are separate categories.
Historical results reflect provider revisions, not versioned snapshots as known
on race day. A live production betting or decision system would need timestamped
entry lists, confirmed grids, weather forecasts, monitoring, scheduled refresh,
official standings/countback and substantially more backtesting.

Live checks found HTTP 401 from the F1 timing endpoint and HTTP 404 from the mirror. Open-Meteo archive is reachable outside the sandbox. Practice/strategy inputs remain missing, but historical weather can now be recovered. The UI labels the data source and distinguishes retrospective archive observations from available pre-race inputs.
If the provider requires account access, configure it locally according to FastF1
documentation and rerun the resumable collector; do not put credentials in source.

- [FastF1 documentation](https://docs.fastf1.dev/) and [data availability](https://github.com/theOehrly/Fast-F1/blob/master/docs/data_reference/index.rst).
- [FastF1 results client](https://github.com/theOehrly/Fast-F1/blob/master/fastf1/ergast/interface.py), backed by Jolpica.
- [XGBoost estimator API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/stable/Python-API.html).
- [FIA 2026 regulation and power-unit overview](https://www.fia.com/news/f1s-new-era-everything-you-need-know-about-how-fia-making-formula-1-more-competitive-more).

Circuit character labels are conservative manual heuristics rather than official
classifications. Fastest-lap probability and CatBoost are not implemented; DNF and
top-10 probabilities are included. The non-finish label groups non-classified
statuses, including withdrawals/disqualifications, so it is broader than purely
mechanical retirement.
