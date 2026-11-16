# Prόvlepsi AI

A privacy-first time-series AutoML studio. Upload a CSV, explore it, choose how to clean it, then train and compare classical, machine-learning and deep-learning models. Parsing, preprocessing and training all happen on your device: the dataset is kept in the browser's IndexedDB and is never uploaded.

Built for the IDEAS (ISI Kolkata) summer internship, 2026.

## Workflow

1. **Upload**: drop a CSV (or try a synthetic sample). Recent datasets are saved in this browser only.
2. **Explore**: summary stats, trend and seasonality detection, line plot with rolling mean, correlation heatmap, histograms, box plots and autocorrelation.
3. **Prepare**: pick the target, time column and features; choose missing-value handling, outlier clipping, text encoding, feature scaling and the chronological train/test split. Clipping and scaling limits are learned from the training window only.
4. **Models**: choose models and tune their hyperparameters.
5. **Train**: Python models run in one Web Worker, neural networks in another, so the page stays responsive. Neural nets show live loss curves.
6. **Results**: leaderboard (RMSE / MAE / MSE / R², or accuracy / precision / recall), every model on the test window, per-model forecast charts, coefficients and feature importance, confusion matrices, a prediction playground and CSV downloads.

## Models

| Model | Runtime | Source |
| --- | --- | --- |
| Linear, Ridge, Lasso Regression | scikit-learn on Pyodide | `python/models/*_model.py` (original scripts, unchanged) |
| Random Forest | scikit-learn on Pyodide | original script, unchanged |
| Logistic Regression, LDA, QDA | scikit-learn on Pyodide | `LogisticRegression_model.py` (original, unchanged) |
| Gradient Boosting | scikit-learn on Pyodide | `GradientBoosting_model.py` (same template) |
| ARIMA | statsmodels on Pyodide | `ARIMA_model.py` (same template) |
| LSTM, GRU, Bi-LSTM, LSTM-GRU hybrid, Transformer | TensorFlow.js (WebGL or CPU) | `src/engines/tf.worker.ts` |

### How the Python scripts are fed

The model scripts are plain top-level Python with placeholders such as `df = pd.read_csv("")`, `response_variable = ""` and `user_entry = {}`. They are **not edited**. `python/provlepsi_runner.py` parses each script and, in memory only:

- points every `read_csv(...)` at the cleaned dataset;
- replaces each top-level placeholder the app supplies (`response_variable`, `split_fraction`, `alpha`, `max_lags`, `n_estimators`, `max_depth`, `min_samples_split`, `classifier_type`, `C_param`, ...) with the value chosen in the UI;
- splits the script at `user_entry = {}`: everything before it trains once, everything after it is re-run for each new prediction from the playground, without retraining.

`python/tests/test_runner.py` checks every script end to end and pins the SHA-256 of the five original files so they stay byte-for-byte unchanged.

One compatibility shim: scikit-learn 1.8 removed `LogisticRegression(multi_class=...)`. On those versions the runner drops that one argument; `lbfgs` already fits a multinomial model, so results are the same.

## Running it

```bash
npm install
npm run dev          # http://localhost:3000
npm run build        # static site in out/, host it anywhere
```

The in-browser Python runtime (Pyodide) is loaded from the jsDelivr CDN on first use and cached. Override the URL with `NEXT_PUBLIC_PYODIDE_URL` to self-host it.

### Optional: local Python engine

For large datasets you can run the same scripts with your desktop Python instead of Pyodide:

```bash
pip install -r python/requirements.txt
python python/local_server.py     # listens on 127.0.0.1:8765 only
```

Then choose **Local Python** from the engine button in the top bar. The server only runs the scripts in `python/models`, only listens on localhost, and only answers the app's origin (`PROVLEPSI_ORIGINS`, default `http://localhost:3000`).

### Tests

```bash
pip install -r python/requirements.txt -r python/requirements-dev.txt
npm run test:py
npm run typecheck && npm run lint
```

## Tech

Next.js 16 (static export), React 19, TypeScript, Tailwind CSS 4, Zustand, Plotly.js, Pyodide, TensorFlow.js, Comlink, PapaParse, idb-keyval.
