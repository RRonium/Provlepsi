import type { TaskType } from "./types";

export type Engine = "python" | "tf";
export type Family = "statistical" | "machine-learning" | "classification" | "deep-learning";

export interface ParamSpec {
  key: string;
  label: string;
  kind: "int" | "float" | "select" | "nullable-int";
  default: number | string | null;
  min?: number;
  max?: number;
  step?: number;
  log?: boolean;
  options?: { value: string; label: string }[];
  help?: string;
}

export interface ModelSpec {
  id: string;
  name: string;
  short: string;
  family: Family;
  engine: Engine;
  task: TaskType;
  /** Python script in python/models (python engine only). */
  file?: string;
  /** Fixed values passed to the script that the user doesn't edit. */
  fixed?: Record<string, string | number>;
  original?: boolean;
  needsFeatures?: boolean;
  forecasts?: boolean;
  blurb: string;
  params: ParamSpec[];
}

const lags = (def = 2): ParamSpec => ({
  key: "max_lags",
  label: "Lag steps",
  kind: "int",
  default: def,
  min: 1,
  max: 12,
  step: 1,
  help: "How many past time steps are added as extra columns.",
});

const dlParams = (units: number, extra: ParamSpec[] = []): ParamSpec[] => [
  { key: "window", label: "Look-back window", kind: "int", default: 24, min: 4, max: 200, step: 1, help: "Past steps the network reads to predict the next one." },
  { key: "units", label: "Hidden units", kind: "int", default: units, min: 4, max: 256, step: 4 },
  ...extra,
  { key: "epochs", label: "Epochs", kind: "int", default: 40, min: 1, max: 500, step: 1 },
  { key: "batchSize", label: "Batch size", kind: "int", default: 32, min: 4, max: 512, step: 4 },
  { key: "learningRate", label: "Learning rate", kind: "float", default: 0.005, min: 0.0001, max: 0.1, step: 0.0001, log: true },
  { key: "dropout", label: "Dropout", kind: "float", default: 0.1, min: 0, max: 0.6, step: 0.05 },
  { key: "forecastSteps", label: "Forecast horizon", kind: "int", default: 10, min: 1, max: 365, step: 1, help: "Steps to forecast past the end of the data." },
];

export const MODELS: ModelSpec[] = [
  {
    id: "arima",
    name: "ARIMA",
    short: "ARIMA",
    family: "statistical",
    engine: "python",
    task: "regression",
    file: "ARIMA_model.py",
    forecasts: true,
    blurb: "Classical autoregressive baseline for series with clear trend or seasonality. Fast and interpretable.",
    params: [
      { key: "p", label: "p · autoregressive order", kind: "int", default: 2, min: 0, max: 10, step: 1 },
      { key: "d", label: "d · differencing", kind: "int", default: 1, min: 0, max: 2, step: 1 },
      { key: "q", label: "q · moving-average order", kind: "int", default: 1, min: 0, max: 10, step: 1 },
      { key: "forecast_steps", label: "Forecast horizon", kind: "int", default: 10, min: 1, max: 365, step: 1 },
    ],
  },
  {
    id: "linear",
    name: "Linear Regression",
    short: "Linear",
    family: "machine-learning",
    engine: "python",
    task: "regression",
    file: "LinearRegression_model.py",
    original: true,
    needsFeatures: true,
    blurb: "Ordinary least squares on your feature columns with a chronological train/test split.",
    params: [],
  },
  {
    id: "ridge",
    name: "Ridge Regression",
    short: "Ridge",
    family: "machine-learning",
    engine: "python",
    task: "regression",
    file: "RidgeRegression_model.py",
    original: true,
    needsFeatures: true,
    blurb: "L2-regularised regression with lagged features. Shrinks correlated lags instead of dropping them.",
    params: [{ key: "alpha", label: "Alpha (L2 penalty)", kind: "float", default: 1, min: 0.001, max: 100, step: 0.001, log: true }, lags()],
  },
  {
    id: "lasso",
    name: "Lasso Regression",
    short: "Lasso",
    family: "machine-learning",
    engine: "python",
    task: "regression",
    file: "LassoRegression_model.py",
    original: true,
    needsFeatures: true,
    blurb: "L1-regularised regression with lagged features. Zeroes out weak lags, so it doubles as feature selection.",
    params: [{ key: "alpha", label: "Alpha (L1 penalty)", kind: "float", default: 0.1, min: 0.001, max: 100, step: 0.001, log: true }, lags()],
  },
  {
    id: "random_forest",
    name: "Random Forest",
    short: "Random Forest",
    family: "machine-learning",
    engine: "python",
    task: "regression",
    file: "RandomForest_model.py",
    original: true,
    blurb: "Ensemble of decorrelated trees on your features plus three lags of the target. Captures non-linear patterns.",
    params: [
      { key: "n_estimators", label: "Trees", kind: "int", default: 200, min: 10, max: 1000, step: 10 },
      { key: "max_depth", label: "Max depth", kind: "nullable-int", default: null, min: 1, max: 50, step: 1, help: "Leave empty for unlimited depth." },
      { key: "min_samples_split", label: "Min samples to split", kind: "int", default: 2, min: 2, max: 50, step: 1 },
    ],
  },
  {
    id: "gradient_boosting",
    name: "Gradient Boosting",
    short: "Gradient Boosting",
    family: "machine-learning",
    engine: "python",
    task: "regression",
    file: "GradientBoosting_model.py",
    blurb: "Boosted shallow trees on your features plus lags of the target. Strong non-linear baseline that trains quickly.",
    params: [
      { key: "n_estimators", label: "Boosting rounds", kind: "int", default: 200, min: 10, max: 1000, step: 10 },
      { key: "learning_rate", label: "Learning rate", kind: "float", default: 0.05, min: 0.001, max: 1, step: 0.001, log: true },
      { key: "max_depth", label: "Tree depth", kind: "int", default: 3, min: 1, max: 12, step: 1 },
      { key: "subsample", label: "Row subsample", kind: "float", default: 1, min: 0.3, max: 1, step: 0.05 },
      lags(3),
    ],
  },
  {
    id: "logistic",
    name: "Logistic Regression",
    short: "Logistic",
    family: "classification",
    engine: "python",
    task: "classification",
    file: "LogisticRegression_model.py",
    fixed: { classifier_type: "logistic" },
    original: true,
    needsFeatures: true,
    blurb: "Multinomial softmax classifier, L2-penalised through C. Works for two or more classes.",
    params: [{ key: "C_param", label: "C (inverse regularisation)", kind: "float", default: 1, min: 0.001, max: 100, step: 0.001, log: true }],
  },
  {
    id: "lda",
    name: "Linear Discriminant Analysis",
    short: "LDA",
    family: "classification",
    engine: "python",
    task: "classification",
    file: "LogisticRegression_model.py",
    fixed: { classifier_type: "lda" },
    original: true,
    needsFeatures: true,
    blurb: "Generative Bayesian classifier that assumes every class shares one covariance matrix.",
    params: [],
  },
  {
    id: "qda",
    name: "Quadratic Discriminant Analysis",
    short: "QDA",
    family: "classification",
    engine: "python",
    task: "classification",
    file: "LogisticRegression_model.py",
    fixed: { classifier_type: "qda" },
    original: true,
    needsFeatures: true,
    blurb: "Like LDA, but each class gets its own covariance, so decision boundaries can curve.",
    params: [],
  },
  {
    id: "lstm",
    name: "LSTM",
    short: "LSTM",
    family: "deep-learning",
    engine: "tf",
    task: "regression",
    forecasts: true,
    blurb: "Long short-term memory network that learns long-range temporal dependencies from sliding windows.",
    params: dlParams(32, [{ key: "layers", label: "Stacked layers", kind: "int", default: 1, min: 1, max: 3, step: 1 }]),
  },
  {
    id: "gru",
    name: "GRU",
    short: "GRU",
    family: "deep-learning",
    engine: "tf",
    task: "regression",
    forecasts: true,
    blurb: "Gated recurrent unit: a lighter, faster alternative to the LSTM with fewer gates.",
    params: dlParams(32, [{ key: "layers", label: "Stacked layers", kind: "int", default: 1, min: 1, max: 3, step: 1 }]),
  },
  {
    id: "bilstm",
    name: "Bi-LSTM",
    short: "Bi-LSTM",
    family: "deep-learning",
    engine: "tf",
    task: "regression",
    forecasts: true,
    blurb: "Bidirectional LSTM that reads each window forwards and backwards. Suited to offline forecasting.",
    params: dlParams(24),
  },
  {
    id: "lstm_gru",
    name: "LSTM-GRU Hybrid",
    short: "LSTM-GRU",
    family: "deep-learning",
    engine: "tf",
    task: "regression",
    forecasts: true,
    blurb: "An LSTM layer feeding a GRU layer, pairing longer memory with lighter gating.",
    params: dlParams(32),
  },
  {
    id: "transformer",
    name: "Transformer",
    short: "Transformer",
    family: "deep-learning",
    engine: "tf",
    task: "regression",
    forecasts: true,
    blurb: "Self-attention encoder with positional encoding. Attends to any step in the window directly.",
    params: [
      ...dlParams(32, [
        { key: "heads", label: "Attention heads", kind: "int", default: 4, min: 1, max: 8, step: 1 },
        { key: "blocks", label: "Encoder blocks", kind: "int", default: 2, min: 1, max: 4, step: 1 },
      ]).map((p) => (p.key === "units" ? { ...p, label: "Model width (d_model)", default: 32, min: 8, max: 128, step: 8 } : p)),
    ],
  },
];

export const FAMILY_LABEL: Record<Family, { title: string; runtime: string }> = {
  statistical: { title: "Statistical", runtime: "statsmodels · Pyodide" },
  "machine-learning": { title: "Machine learning", runtime: "scikit-learn · Pyodide" },
  classification: { title: "Classification", runtime: "scikit-learn · Pyodide" },
  "deep-learning": { title: "Deep learning", runtime: "TensorFlow.js · WebGL" },
};

export function modelById(id: string): ModelSpec {
  const m = MODELS.find((x) => x.id === id);
  if (!m) throw new Error(`Unknown model ${id}`);
  return m;
}

export function defaultParams(spec: ModelSpec): Record<string, number | string | null> {
  return Object.fromEntries(spec.params.map((p) => [p.key, p.default]));
}

export function isClassification(spec: ModelSpec) {
  return spec.task === "classification";
}
