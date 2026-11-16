"""
Provlepsi model runner.

The model files in ``python/models`` are plain scripts: they read a CSV with
``pd.read_csv("")`` and declare their inputs as top-level placeholders such as
``response_variable = ""`` or ``split_fraction = None``. This module runs those
scripts *unchanged on disk* by rewriting them in memory before executing:

* every ``pd.read_csv(...)`` call reads the CSV path chosen by the app;
* every top-level assignment to a name that the app supplies (``response_variable``,
  ``split_fraction``, ``alpha``, ``n_estimators``, ...) takes the app's value;
* the script is split at ``user_entry = {}``. Everything before it is the
  training phase, which runs once. Everything after it is the prediction
  phase, which can be re-run later for any new ``user_entry`` without
  retraining.

The same module runs in the browser under Pyodide and on a desktop Python
(see ``local_server.py``), so both engines behave identically.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import math
import traceback
import uuid
import warnings
from typing import Any

CFG_NAME = "__provlepsi_cfg__"
ENTRY_NAME = "user_entry"
MAX_SERIES_POINTS = 5000

# Fitted namespaces, kept so that predict() can reuse a trained model.
_SESSIONS: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# Source rewriting
# --------------------------------------------------------------------------- #

def _cfg_lookup(key: str) -> ast.expr:
    return ast.Subscript(
        value=ast.Name(id=CFG_NAME, ctx=ast.Load()),
        slice=ast.Constant(value=key),
        ctx=ast.Load(),
    )


def _logistic_accepts_multi_class() -> bool:
    try:
        from sklearn.linear_model import LogisticRegression

        return "multi_class" in inspect.signature(LogisticRegression).parameters
    except Exception:  # sklearn missing: nothing to adapt
        return True


class _CallRewriter(ast.NodeTransformer):
    """Points read_csv at the app's file and smooths over library upgrades."""

    def __init__(self) -> None:
        self.drop_multi_class = not _logistic_accepts_multi_class()
        self.notes: list[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "read_csv":
            node.args = [_cfg_lookup("csv_path")] + node.args[1:]
        if name == "LogisticRegression" and self.drop_multi_class:
            before = len(node.keywords)
            node.keywords = [k for k in node.keywords if k.arg != "multi_class"]
            if len(node.keywords) != before:
                # scikit-learn >= 1.8 removed this argument; lbfgs is multinomial by default,
                # so dropping it keeps the original behaviour.
                self.notes.append(
                    "Ignored LogisticRegression(multi_class=...) because this scikit-learn "
                    "version removed it; lbfgs already fits a multinomial model."
                )
        return node


def _assigned_name(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def compile_model(source: str, cfg_keys: set[str], filename: str = "<model>"):
    """Return (train_code, predict_code, info) for a model script."""
    tree = ast.parse(source, filename=filename)
    rewriter = _CallRewriter()
    tree = rewriter.visit(tree)

    injected: list[str] = []
    split_at: int | None = None
    for i, stmt in enumerate(tree.body):
        name = _assigned_name(stmt)
        if name is None:
            continue
        if name == ENTRY_NAME and split_at is None:
            split_at = i
            continue
        if name in cfg_keys and name not in (ENTRY_NAME, "csv_path"):
            stmt.value = _cfg_lookup(name)
            injected.append(name)

    if split_at is None:
        train_body, predict_body = tree.body, []
    else:
        train_body, predict_body = tree.body[:split_at], tree.body[split_at + 1 :]

    def _compile(body: list[ast.stmt], label: str):
        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)
        return compile(module, f"{filename}:{label}", "exec")

    info = {
        "injected": sorted(set(injected)),
        "has_user_entry": split_at is not None,
        "notes": rewriter.notes,
    }
    return _compile(train_body, "train"), _compile(predict_body, "predict"), info


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd
    except ImportError:  # pragma: no cover
        np = pd = None  # type: ignore

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return _jsonable(float(value))
        if isinstance(value, np.ndarray):
            return [_jsonable(v) for v in value.tolist()]
    if pd is not None:
        if isinstance(value, pd.Series):
            return [_jsonable(v) for v in value.tolist()]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _series_payload(values: Any, index: Any = None) -> dict[str, list] | None:
    if values is None:
        return None
    try:
        vals = list(values.tolist() if hasattr(values, "tolist") else values)
    except Exception:
        return None
    if index is None:
        index = getattr(values, "index", range(len(vals)))
    idx = list(index)
    if len(idx) != len(vals):
        idx = list(range(len(vals)))
    if len(vals) > MAX_SERIES_POINTS:
        step = math.ceil(len(vals) / MAX_SERIES_POINTS)
        vals, idx = vals[::step], idx[::step]
    return {"index": _jsonable(idx), "values": _jsonable(vals)}


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

METRIC_NAMES = ("MAE", "MSE", "RMSE", "AIC", "BIC", "accuracy", "precision", "recall")
EXTRA_NAMES = (
    "conf_matrix",
    "class_key_map",
    "K_classes",
    "feature_weights",
    "selected_features",
    "zeroed_out_features",
    "forecast",
)
PREDICTION_NAMES = ("single_prediction", "predicted_class_label", "predicted_probabilities")


def _default_entry(ns: dict[str, Any]) -> dict[str, Any]:
    X = ns.get("X")
    if X is not None and hasattr(X, "iloc") and len(X):
        return {str(k): v for k, v in X.iloc[-1].to_dict().items()}
    return {}


def _coerce_entry(ns: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, Any]:
    merged = _default_entry(ns)
    X = ns.get("X")
    for key, raw in (entry or {}).items():
        if raw is None or raw == "":
            continue
        value = raw
        if X is not None and key in getattr(X, "columns", []):
            if X[key].dtype.kind in "iufb":
                value = float(raw)
        merged[key] = value
    return merged


def _collect_predictions(ns: dict[str, Any]) -> dict[str, Any]:
    out = {name: _jsonable(ns[name]) for name in PREDICTION_NAMES if name in ns}
    for name in ("forecast",):
        if name in ns:
            out[name] = _jsonable(ns[name])
    out["user_entry"] = _jsonable(ns.get(ENTRY_NAME, {}))
    return out


def _run_predict(session: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, Any]:
    ns = session["ns"]
    ns[ENTRY_NAME] = _coerce_entry(ns, entry)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(session["predict_code"], ns)
    return _collect_predictions(ns)


def _relevant_notes(notes: list[str], ns: dict[str, Any]) -> list[str]:
    # The multi_class note only matters when a LogisticRegression was actually fitted
    # (the same script also builds LDA and QDA).
    fitted_logistic = any(type(v).__name__ == "LogisticRegression" for v in ns.values())
    return [n for n in notes if fitted_logistic or "LogisticRegression" not in n]


def run_model(source: str, config: dict[str, Any], filename: str = "<model>") -> dict[str, Any]:
    """Train a model script with the app's config and return everything the UI shows."""
    cfg = dict(config)
    entry = cfg.pop(ENTRY_NAME, None)
    train_code, predict_code, info = compile_model(source, set(cfg) | {"csv_path"}, filename)

    ns: dict[str, Any] = {"__name__": "__provlepsi_model__", CFG_NAME: cfg}
    stdout = io.StringIO()
    caught: list[str] = []
    with contextlib.redirect_stdout(stdout), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exec(train_code, ns)
        caught = sorted({str(x.message) for x in w})[:10]

    session_id = uuid.uuid4().hex
    session = {"ns": ns, "predict_code": predict_code}
    _SESSIONS[session_id] = session

    prediction = _run_predict(session, entry) if info["has_user_entry"] else {}

    X = ns.get("X")
    y_test = ns.get("y_test")
    y_train = ns.get("y_train")
    result: dict[str, Any] = {
        "session_id": session_id,
        "metrics": {n: _jsonable(ns[n]) for n in METRIC_NAMES if n in ns},
        "extras": {n: _jsonable(ns[n]) for n in EXTRA_NAMES if n in ns},
        "prediction": prediction,
        "feature_columns": [str(c) for c in X.columns] if X is not None and hasattr(X, "columns") else [],
        "default_entry": _jsonable(_default_entry(ns)),
        "train": _series_payload(y_train),
        "test": _series_payload(y_test),
        "test_predictions": _series_payload(ns.get("sample_predictions"), getattr(y_test, "index", None)),
        "n_train": int(len(y_train)) if y_train is not None else None,
        "n_test": int(len(y_test)) if y_test is not None else None,
        "injected": info["injected"],
        "notes": _relevant_notes(info["notes"], ns) + caught,
        "stdout": stdout.getvalue()[-4000:],
    }
    return result


def predict(session_id: str, entry: dict[str, Any] | None) -> dict[str, Any]:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError("This model is no longer in memory. Train it again to make predictions.")
    return _run_predict(session, entry)


def forget(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


# JSON-in / JSON-out wrappers used by the JavaScript side ---------------------- #

def _envelope(fn, *args) -> str:
    try:
        return json.dumps({"ok": True, "data": fn(*args)}, allow_nan=False)
    except Exception as exc:  # report the script's own error back to the UI
        return json.dumps(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-3000:],
            }
        )


def run_model_json(source: str, config_json: str, filename: str = "<model>") -> str:
    return _envelope(run_model, source, json.loads(config_json), filename)


def predict_json(session_id: str, entry_json: str) -> str:
    return _envelope(predict, session_id, json.loads(entry_json))
