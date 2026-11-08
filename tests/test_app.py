"""Exercise the real trained app, when local artifacts are available."""

import pytest
from streamlit.testing.v1 import AppTest

from src.utils import MODELS, ROOT


@pytest.mark.skipif(
    not (MODELS / "results-only_post_best.joblib").exists(), reason="Requires trained artifacts"
)
def test_app_runs_and_predicts_latest_race():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    app.sidebar.button[0].click().run(timeout=60)
    assert not app.exception
    assert not app.error
    assert any(metric.label == "Win probability" for metric in app.metric)
    assert len(app.tabs) == 7
    app.checkbox[0].check().run(timeout=60)
    assert not app.exception
    assert all("SHAP unavailable" not in str(w.value) for w in app.warning)
    assert any("Data availability" in str(w.value) for w in app.warning)


@pytest.mark.skipif(
    not (MODELS / "observed-weather_post_best.joblib").exists(), reason="Requires retrained variants"
)
def test_variants_share_test_races_and_observed_warning():
    import json
    import numpy as np
    import pandas as pd
    from src.predict import predict_race

    report = json.loads((MODELS / "evaluation.json").read_text())
    assert report["split"]["validation"] == 2024
    assert report["split"]["calibration"] == 2025
    assert report["split"]["test"] == 2026
    event_ids = []
    for variant in ("results-only", "enriched", "observed-weather"):
        test = pd.read_parquet(MODELS / f"{variant}_post_best_test.parquet")
        event_ids.append(set(zip(test.race_id, test.driver, strict=True)))
        prediction = predict_race(2026, 19, variant=variant)
        assert prediction.model_variant.eq(variant).all()
        np.testing.assert_allclose(prediction.win_probability.sum(), 1)
    assert event_ids[0] == event_ids[1] == event_ids[2]
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    selection = next(item for item in app.sidebar.selectbox if item.label == "Model variant")
    selection.select("observed-weather").run()
    app.sidebar.button[0].click().run()
    assert not app.exception and not app.error
    assert any("retrospective" in str(w.value).lower() for w in app.warning)
