# F1 Race Predictor

Predicts F1 race outcomes (win, podium, top 10, DNF and finishing order) using gradient boosting, with a Streamlit app to explore the results.

Data comes from the [FastF1](https://github.com/theOehrly/Fast-F1) library (race and qualifying results via Jolpica) and historical weather from Open-Meteo.

## How it works

I trained on 2018-2023 races, tuned on 2024, calibrated probabilities on 2025, and tested on the 2026 races that have already happened. The 2026 season was never used to pick models or parameters.

There are two versions of the model:
- **Pre-qualifying**: uses only history (recent form, circuit record, team reliability, championship position).
- **Post-qualifying**: adds the qualifying result, which is the biggest single signal.

I compared scikit-learn GradientBoosting, XGBoost and LightGBM for each target and picked the best one using 2024 validation loss.

## Results

19 races in the 2026 test set:

| Model | Winner correct | Winner in top 3 |
|---|---|---|
| Pre-qualifying | 36.8% | 68.4% |
| Post-qualifying | 63.2% | 94.7% |

For comparison, picking the pole sitter gets 68.4% and picking the championship leader gets 36.8%. So the model doesn't beat the pole-sitter baseline, but it puts the winner in its top 3 in 18 of 19 races and its probabilities are well calibrated. With only 19 races, a difference of one race is about 5 points, so I wouldn't read much into the gap.

## Features

- Recent finishing position, points and positions gained (last 3 and 5 races)
- Driver and team record at each circuit
- Qualifying gaps to pole and to the teammate
- Championship standings before the race
- Team reliability and DNF rate
- Regulation era (pre-2022, 2022-2025, 2026)
- Weather (air temperature, humidity, wind, rain) from Open-Meteo

## Limitations

- **No timing data.** FastF1's live timing server returned HTTP 401, so practice pace, tyre strategy and clean-air features aren't in the models. The collector supports them, so they can be added if access works.
- **Weather is retrospective.** I use archived weather for past races. For a real upcoming race you would need a forecast.
- **Small test set.** 19 races is not much, and the 2026 regulation change is a big shift from the training years.
- F1 is unpredictable, so treat the output as probabilities, not guarantees.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/data_collection.py --mode core      # results + qualifying
python src/weather_features.py --download      # weather
python src/train_model.py --trials 5
streamlit run app.py
```

On a Mac, XGBoost and LightGBM need OpenMP: `brew install libomp`.

To predict a single race:

```bash
python src/predict.py --year 2026 --round 19
```

To try the timing collector (needs access to F1's timing servers):

```bash
python src/data_collection.py --check-network
python src/data_collection.py --mode sessions --start 2026 --end 2026 --rounds 19
```

## Project layout

- `src/data_collection.py`: downloads results, schedules and timing
- `src/feature_engineering.py`: builds features from past races only
- `src/weather_features.py`: matches Open-Meteo weather to each race
- `src/train_model.py`: trains and calibrates the models
- `src/predict.py`: predicts a race
- `src/explain.py`: SHAP plots
- `app.py`: Streamlit app
- `tests/`: checks for data leakage and probability sanity

## Credits

Data from FastF1, Jolpica and Open-Meteo (CC BY 4.0).
