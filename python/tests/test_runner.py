"""Runs every model script through the runner on a small synthetic series."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import provlepsi_runner as runner  # noqa: E402

MODELS = ROOT / "models"


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory) -> str:
    rng = np.random.default_rng(0)
    n = 160
    t = np.arange(n)
    temp = 20 + 5 * np.sin(t / 8) + rng.normal(0, 0.5, n)
    promo = (rng.random(n) > 0.7).astype(int)
    sales = 50 + 0.3 * t + 2 * temp + 6 * promo + rng.normal(0, 1, n)
    regime = np.where(temp > 20, "high", "low")
    df = pd.DataFrame({"sales": sales, "temp": temp, "promo": promo, "regime": regime})
    path = tmp_path_factory.mktemp("data") / "data.csv"
    df.to_csv(path, index=False)
    numeric = path.with_name("numeric.csv")
    df.drop(columns=["regime"]).to_csv(numeric, index=False)
    return str(path)


def numeric_csv(csv_path: str) -> str:
    return str(Path(csv_path).with_name("numeric.csv"))


def run(name: str, cfg: dict) -> dict:
    source = (MODELS / name).read_text()
    out = json.loads(runner.run_model_json(source, json.dumps(cfg), name))
    assert out["ok"], out.get("traceback")
    return out["data"]


def assert_regression(data: dict) -> None:
    for key in ("MAE", "MSE", "RMSE"):
        assert math.isfinite(data["metrics"][key])
    assert math.isclose(data["metrics"]["RMSE"], math.sqrt(data["metrics"]["MSE"]))
    assert len(data["test"]["values"]) == len(data["test_predictions"]["values"])
    assert data["test"]["index"] == data["test_predictions"]["index"]


def test_linear(csv_path):
    data = run(
        "LinearRegression_model.py",
        {"csv_path": numeric_csv(csv_path), "response_variable": "sales", "split_fraction": 0.2},
    )
    assert_regression(data)
    assert data["feature_columns"] == ["temp", "promo"]
    assert data["n_test"] == 32
    assert isinstance(data["prediction"]["single_prediction"], float)


@pytest.mark.parametrize("name", ["RidgeRegression_model.py", "LassoRegression_model.py"])
def test_regularised(csv_path, name):
    data = run(
        name,
        {
            "csv_path": numeric_csv(csv_path),
            "response_variable": "sales",
            "split_fraction": 0.25,
            "alpha": 0.5,
            "max_lags": 2,
        },
    )
    assert_regression(data)
    assert "temp_lag_2" in data["feature_columns"]
    assert set(data["extras"]["feature_weights"]) == set(data["feature_columns"])
    if name.startswith("Lasso"):
        assert "selected_features" in data["extras"]


def test_random_forest_and_repredict(csv_path):
    data = run(
        "RandomForest_model.py",
        {
            "csv_path": numeric_csv(csv_path),
            "response_variable": "sales",
            "split_fraction": 0.2,
            "n_estimators": 20,
            "max_depth": None,
            "min_samples_split": 2,
        },
    )
    assert_regression(data)
    assert {"lag1", "lag2", "lag3"} <= set(data["feature_columns"])
    entry = dict(data["default_entry"])
    entry["lag1"] = entry["lag1"] + 30
    again = json.loads(runner.predict_json(data["session_id"], json.dumps(entry)))
    assert again["ok"]
    assert again["data"]["user_entry"]["lag1"] == pytest.approx(entry["lag1"])


def test_gradient_boosting(csv_path):
    data = run(
        "GradientBoosting_model.py",
        {
            "csv_path": numeric_csv(csv_path),
            "response_variable": "sales",
            "split_fraction": 0.2,
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "max_lags": 3,
        },
    )
    assert_regression(data)


@pytest.mark.parametrize("kind", ["logistic", "lda", "qda"])
def test_classifiers(csv_path, kind):
    data = run(
        "LogisticRegression_model.py",
        {
            "csv_path": csv_path,
            "response_variable": "regime",
            "split_fraction": 0.2,
            "classifier_type": kind,
            "C_param": 1.0,
        },
    )
    assert 0 <= data["metrics"]["accuracy"] <= 1
    assert data["prediction"]["predicted_class_label"] in {"high", "low"}
    assert set(data["prediction"]["predicted_probabilities"]) == {"high", "low"}
    assert len(data["extras"]["conf_matrix"]) == 2


def test_arima(csv_path):
    data = run(
        "ARIMA_model.py",
        {
            "csv_path": numeric_csv(csv_path),
            "response_variable": "sales",
            "split_fraction": 0.2,
            "p": 2,
            "d": 1,
            "q": 1,
            "forecast_steps": 7,
        },
    )
    assert_regression(data)
    assert len(data["prediction"]["forecast"]) == 7
    again = json.loads(runner.predict_json(data["session_id"], json.dumps({"steps": 3})))
    assert len(again["data"]["forecast"]) == 3


def test_errors_are_reported(csv_path):
    source = (MODELS / "LinearRegression_model.py").read_text()
    out = json.loads(
        runner.run_model_json(
            source,
            json.dumps({"csv_path": numeric_csv(csv_path), "response_variable": "nope", "split_fraction": 0.2}),
        )
    )
    assert not out["ok"] and "nope" in out["error"]


# SHA-256 of the five scripts exactly as Ron supplied them. The runner never edits
# them on disk; this test fails if anyone does.
ORIGINAL_HASHES = {
    "LinearRegression_model.py": "1fe69825c35c7742c319e0ecc30cc53df852e53ff2f6174fff28e19179335195",
    "RidgeRegression_model.py": "8417c6adea9c7745341a8367ec3f54228660495eec1de11e965418bca3e34b56",
    "LassoRegression_model.py": "811c8a8444a6f2ddacd58198dd44d16757c490f98fba628e808eb0d721716fa9",
    "RandomForest_model.py": "9f89f4a7d1f0a11416419175ac0f9a996887ada2c395dcb307d9d5d3d98ae8fe",
    "LogisticRegression_model.py": "b9b9c41b623a0ce9556521018bf19219ae0f48f6ad4077313490fc2c3e74a666",
}


@pytest.mark.parametrize("name", sorted(ORIGINAL_HASHES))
def test_original_files_untouched(name):
    digest = hashlib.sha256((MODELS / name).read_bytes()).hexdigest()
    assert digest == ORIGINAL_HASHES[name]
