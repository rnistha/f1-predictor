"""Streamlit explorer for measured F1 forecasts and their limitations."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from src.data_collection import load_summaries
from src.metadata import TEAM_COLORS
from src.predict import predict_race
from src.utils import DATA, MODELS, read_frame, setup
from src.weather_features import load_race_weather_samples

st.set_page_config(page_title="F1 Race Lab", page_icon="🏎️", layout="wide")
setup()


@st.cache_data(ttl=120)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load local checkpoints; no surprise full-season network fetch in the UI."""
    return read_frame(DATA / "race.parquet"), read_frame(DATA / "quali.parquet"), load_summaries()


@st.cache_data(ttl=120)
def forecast(year: int, rnd: int, model: str, phase: str, modified: float, variant: str) -> pd.DataFrame:
    """Cache forecasts with artifact modification time as part of the key."""
    return predict_race(year, rnd, model, phase, variant)


def main() -> None:
    st.title("🏎️ F1 Race Lab")
    st.caption("Race models • chronological evaluation • explicit weather provenance")
    races, quali, summaries = load_data()
    if races.empty:
        st.info("No local data yet. Run `python src/data_collection.py --mode core`, then train the models.")
        return
    schedules = {int(p.name.split("_")[0]): read_frame(p) for p in (DATA / "raw").glob("*_schedule.parquet")}
    years = sorted(set(races.year.astype(int)) | set(schedules), reverse=True)
    year = st.sidebar.selectbox("Year", years)
    schedule = schedules.get(year, pd.DataFrame())
    if not schedule.empty:
        events = {int(r.RoundNumber): r.EventName for _, r in schedule.iterrows() if r.RoundNumber > 0}
    else:
        events = dict(
            races[races.year == year][["round", "race"]].drop_duplicates().itertuples(index=False, name=None)
        )
    completed = races[races.year == year]["round"]
    default = int(completed.max()) if len(completed) else min(events)
    rnd = st.sidebar.selectbox(
        "Grand Prix",
        list(events),
        index=list(events).index(default) if default in events else 0,
        format_func=lambda r: f"{r}. {events[r]}",
    )
    model = st.sidebar.selectbox("Model", ["best", "xgboost", "lightgbm", "sklearn"])
    phase = st.sidebar.selectbox(
        "Information available",
        ["auto", "pre", "post"],
        format_func=lambda x: {"auto": "Automatic", "pre": "Before qualifying", "post": "After qualifying"}[
            x
        ],
    )
    variant = st.sidebar.selectbox("Model variant", ["results-only", "enriched", "observed-weather"])
    if variant == "observed-weather":
        st.sidebar.warning(
            "Retrospective race-window weather: this is a hindsight analysis, not a pre-race prediction. Future races require a forecast."
        )
    key = (year, rnd, model, phase, variant)
    if st.sidebar.button("Run prediction", type="primary"):
        try:
            modified = max((p.stat().st_mtime for p in MODELS.glob("*.joblib")), default=0)
            with st.spinner("Building pre-race features and calculating probabilities…"):
                st.session_state["prediction"] = forecast(year, rnd, model, phase, modified, variant)
                st.session_state["prediction_key"] = key
        except Exception as exc:
            st.error(str(exc))
    prediction = st.session_state.get("prediction") if st.session_state.get("prediction_key") == key else None
    tabs = st.tabs(
        [
            "Race Prediction",
            "Predicted vs Actual",
            "Driver & Team Form",
            "Weather & Conditions",
            "Circuit History",
            "Qualifying & Practice Pace",
            "Model Performance",
        ]
    )
    actual = races[(races.year == year) & (races["round"] == rnd)]
    prior = races[(races.year < year) | ((races.year == year) & (races["round"] < rnd))]
    session_data = (
        summaries[(summaries.year == year) & (summaries["round"] == rnd)]
        if not summaries.empty
        else summaries
    )
    with tabs[0]:
        if prediction is None:
            st.info(
                "Select a Grand Prix and click Run prediction. Historical races must be after the model calibration cutoff."
            )
        else:
            st.subheader(events[rnd])
            st.caption(
                f"Variant: {prediction.model_variant.iloc[0]} · {prediction.phase.iloc[0]}-qualifying model. Grid: {prediction.grid_source.iloc[0]}. Ranking uses the finishing-position model; win probabilities use a separate classifier."
            )
            if prediction.model_variant.iloc[0] == "observed-weather":
                st.warning(
                    "Observed-weather model uses weather during the race. Its results are retrospective; a forecast would be needed for a real future race."
                )
            leader = prediction.loc[prediction.win_probability.idxmax()]
            a, b, c = st.columns(3)
            a.metric("Most likely winner", leader.driver.replace("_", " ").title())
            b.metric("Win probability", f"{leader.win_probability:.1%}")
            c.metric("Drivers", len(prediction))
            colors = TEAM_COLORS.copy()
            missing = {
                label: int((prediction[column] == 0).sum())
                for column, label in [
                    ("has_practice_data", "practice pace"),
                    ("has_weather_data", "pre-race weather"),
                    ("has_strategy_data", "historical strategy"),
                ]
            }
            if any(missing.values()):
                st.warning(
                    "Data availability — drivers missing: "
                    + ", ".join(f"{label}: {count}/{len(prediction)}" for label, count in missing.items())
                    + ". Results-only excludes timing features; enriched retains missing values and availability flags."
                )
            if prediction.grid.dropna().duplicated().any():
                st.caption(
                    "The provider reports duplicate qualifying positions for this event. They are retained as supplied; grid values are proxies, not a verified starting grid."
                )
            if not session_data.empty and "team_color" in session_data:
                for _, row in session_data.dropna(subset=["team_color"]).iterrows():
                    match = prediction[prediction.driver == row.driver]
                    if len(match) and len(str(row.team_color)) == 6:
                        colors[match.team.iloc[0]] = "#" + row.team_color
            cols = [
                "predicted_position",
                "driver",
                "team",
                "grid",
                "win_probability",
                "podium_probability",
                "top10_probability",
                "dnf_probability",
            ]
            table = prediction[cols].style.format(
                {c: "{:.1%}" for c in cols if "probability" in c}, na_rep="—"
            )
            table = table.apply(
                lambda r: ["background-color: #294b3e" if r.predicted_position <= 3 else ""] * len(r), axis=1
            )
            st.dataframe(table, hide_index=True, width="stretch")
            st.plotly_chart(
                px.bar(
                    prediction.sort_values("win_probability", ascending=False),
                    x="driver",
                    y="win_probability",
                    color="team",
                    color_discrete_map=colors,
                ),
                width="stretch",
            )
            st.download_button(
                "Download prediction CSV",
                prediction[cols].to_csv(index=False),
                f"f1_{year}_{rnd}.csv",
                "text/csv",
            )
    with tabs[1]:
        if prediction is None or actual.empty:
            st.info("Run a prediction for a completed held-out race to compare results.")
        else:
            compare = prediction[["driver", "predicted_position", "win_probability"]].merge(
                actual[["driver", "finish", "status"]], on="driver"
            )
            a, b = st.columns(2)
            a.metric("Rank MAE", f"{(compare.predicted_position - compare.finish).abs().mean():.2f}")
            b.metric(
                "Spearman rank correlation",
                f"{compare.predicted_position.corr(compare.finish, method='spearman'):.2f}",
            )
            st.dataframe(compare, hide_index=True)
            st.plotly_chart(
                px.scatter(
                    compare,
                    x="finish",
                    y="predicted_position",
                    hover_name="driver",
                    range_x=[0, 23],
                    range_y=[0, 23],
                ),
                width="stretch",
            )
    with tabs[2]:
        season = prior[prior.year == year].sort_values("round").copy()
        if season.empty:
            st.info("No earlier races in the selected season.")
        else:
            season["rolling_finish"] = season.groupby("driver").finish.transform(
                lambda s: s.rolling(5, min_periods=1).mean()
            )
            selected = st.multiselect(
                "Drivers",
                sorted(season.driver.unique()),
                default=list(season.groupby("driver").points.sum().nlargest(5).index),
            )
            st.plotly_chart(
                px.line(
                    season[season.driver.isin(selected)],
                    x="round",
                    y="rolling_finish",
                    color="driver",
                    markers=True,
                ),
                width="stretch",
            )
            st.caption(
                "Points below are Grand Prix points before this round; sprint points are included in model features but not these charts."
            )
            a, b = st.columns(2)
            a.plotly_chart(
                px.bar(
                    season.groupby("driver", as_index=False)
                    .points.sum()
                    .sort_values("points", ascending=False),
                    x="driver",
                    y="points",
                ),
                width="stretch",
            )
            b.plotly_chart(
                px.bar(season.groupby("team", as_index=False).points.sum(), x="team", y="points"),
                width="stretch",
            )
    with tabs[3]:
        paths = list((DATA / "sessions").glob(f"{year}_{rnd}_*/weather_data.parquet"))
        race_window = load_race_weather_samples(year, rnd)
        options = (["Observed race-hour window"] if not race_window.empty else []) + [
            p.parent.name for p in paths
        ]
        if not options:
            st.info("Weather data has not been downloaded for this event.")
        else:
            selection = st.selectbox("Session weather", options)
            weather = (
                race_window
                if selection == "Observed race-hour window"
                else read_frame(next(p for p in paths if p.parent.name == selection))
            )
            if "weather_source" in weather and weather.weather_source.eq("open-meteo-archive").any():
                st.caption(
                    "Historical reanalysis: [Open-Meteo](https://open-meteo.com/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Not a race-day forecast."
                )
            if "weather_source" in weather:
                st.caption(
                    "Weather source: "
                    + str(weather.weather_source.iloc[0])
                    + ". Archive weather is retrospective reanalysis, not a pre-race forecast."
                )
            weather["minutes"] = weather.Time.dt.total_seconds() / 60
            st.plotly_chart(
                px.line(weather, x="minutes", y=["AirTemp", "TrackTemp"], labels={"value": "°C"}),
                width="stretch",
            )
            humidity_column, rain_column = st.columns(2)
            humidity_column.plotly_chart(
                px.line(weather, x="minutes", y="Humidity", labels={"Humidity": "Relative humidity (%)"}),
                width="stretch",
            )
            rain_field = "Precipitation" if "Precipitation" in weather else "Rainfall"
            weather["rain_value"] = pd.to_numeric(weather[rain_field], errors="coerce").astype(float)
            rain_column.plotly_chart(
                px.bar(
                    weather,
                    x="minutes",
                    y="rain_value",
                    labels={
                        "rain_value": "Precipitation (mm)"
                        if rain_field == "Precipitation"
                        else "Rain detected (0/1)"
                    },
                ),
                width="stretch",
            )
            st.caption(
                "Observed race-window weather is used only by the explicitly retrospective observed-weather variant. Results-only and enriched exclude it. Real future races require a forecast."
            )
    with tabs[4]:
        circuit = actual.circuit.iloc[0] if len(actual) else None
        if circuit is None:
            st.info("Circuit identity is not available locally for this event yet.")
        else:
            history = prior[prior.circuit == circuit]
            poles = history[history.grid == 1]
            st.metric(
                "Historical pole-to-win rate", f"{(poles.finish == 1).mean():.1%}" if len(poles) else "—"
            )
            st.dataframe(
                history[history.finish == 1][["year", "race", "driver", "team", "grid"]], hide_index=True
            )
            st.dataframe(
                history.groupby("driver")
                .agg(
                    races=("finish", "size"),
                    mean_finish=("finish", "mean"),
                    wins=("finish", lambda s: (s == 1).sum()),
                )
                .sort_values("mean_finish")
            )
    with tabs[5]:
        q = quali[(quali.year == year) & (quali["round"] == rnd)] if not quali.empty else quali
        if not q.empty:
            st.plotly_chart(
                px.bar(
                    q.sort_values("quali_position"),
                    x="driver",
                    y=["q1", "q2", "q3"],
                    barmode="group",
                    labels={"value": "Lap seconds"},
                ),
                width="stretch",
            )
        fp = (
            session_data[session_data.session.str.startswith("Practice")]
            if not session_data.empty
            else session_data
        )
        if not fp.empty and "long_run" in fp and fp.long_run.notna().any():
            st.plotly_chart(
                px.bar(fp, x="driver", y="long_run", color="session", barmode="group"), width="stretch"
            )
            st.caption(
                "Long-run median of clean laps in stints with ≥5 laps. Fuel loads and run plans are unknown."
            )
        else:
            st.info("Practice timing is not downloaded for this event.")
    with tabs[6]:
        performance_tab()


def performance_tab() -> None:
    """Display measured test metrics, reliability, ranking and SHAP explanations."""
    path = MODELS / "evaluation.json"
    if not path.exists():
        st.info("Train models to generate evaluation results.")
        return
    report = json.loads(path.read_text())
    st.write("Chronological split", report["split"])
    st.caption(
        f"Training dataset: {report.get('provenance', {}).get('races', '—')} races; sessions with laps: {report.get('provenance', {}).get('timing_sessions', '—')}."
    )
    st.dataframe(pd.DataFrame(report["models"]), hide_index=True)
    if "variant_difference_enriched_minus_results_only" in report:
        st.caption(
            "Enriched minus results-only on identical test races. Positive accuracy/rank correlation is better; negative loss/MAE is better."
        )
        st.dataframe(pd.DataFrame(report["variant_difference_enriched_minus_results_only"]), hide_index=True)
        st.write("Feature availability (fraction of driver-race rows)", report.get("availability", {}))
    if "observed_difference_minus_results_only" in report:
        st.warning(report["observed_weather_warning"])
        st.dataframe(pd.DataFrame(report["observed_difference_minus_results_only"]), hide_index=True)
    variant = st.selectbox("Evaluated model variant", ["results-only", "enriched", "observed-weather"])
    phase = st.selectbox("Evaluated feature set", ["post", "pre"])
    test = read_frame(MODELS / f"{variant}_{phase}_best_test.parquet")
    if test.empty:
        return
    fraction, mean = calibration_curve(test.winner, test.winner_probability, n_bins=8, strategy="quantile")
    fig = go.Figure(
        [
            go.Scatter(x=mean, y=fraction, mode="lines+markers", name="Held-out winners"),
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal"),
        ]
    )
    fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed win fraction")
    st.plotly_chart(fig, width="stretch")
    picked = test.groupby("race_id").winner_probability.transform("max") == test.winner_probability
    st.plotly_chart(
        px.imshow(
            confusion_matrix(test.winner, picked.astype(int)),
            text_auto=True,
            labels={"x": "Predicted winner", "y": "Actual winner"},
        ),
        width="stretch",
    )
    st.plotly_chart(
        px.scatter(test, x="finish", y="predicted_position", hover_data=["driver", "race_id"], opacity=0.3),
        width="stretch",
    )
    importance = read_frame(MODELS / f"{variant}_{phase}_importance.parquet").head(20)
    st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h"), width="stretch")
    st.caption(
        "Native tree importance: gain for XGBoost/LightGBM; impurity importance for sklearn. SHAP explains the uncalibrated winner model, before race normalization."
    )
    if st.checkbox("Compute SHAP explanations (may take a moment)"):
        try:
            import shap

            bundle = joblib.load(MODELS / f"{variant}_{phase}_best.joblib")
            item = bundle["targets"]["winner"]
            sample = test.head(200)
            x = item["preprocessor"].transform(sample[bundle["columns"]])
            with st.spinner("Computing tree explanations…"):
                explanation = shap.TreeExplainer(item["model"])(x)
                if explanation.values.ndim == 3:
                    explanation = explanation[:, :, 1]
                explanation.feature_names = list(item["preprocessor"].get_feature_names_out())
                shap.plots.beeswarm(explanation, show=False, max_display=15)
                st.pyplot(plt.gcf())
                plt.close("all")
                row = st.selectbox(
                    "Driver / race explanation",
                    list(range(len(sample))),
                    format_func=lambda i: f"{sample.iloc[i].race_id} · {sample.iloc[i].driver}",
                )
                shap.plots.waterfall(explanation[row], show=False, max_display=15)
                st.pyplot(plt.gcf())
                plt.close("all")
        except Exception as exc:
            st.warning(f"SHAP unavailable for this artifact: {exc}")


if __name__ == "__main__":
    main()
