export type Cell = number | string | null;

export type ColumnKind = "numeric" | "categorical" | "datetime";

export interface ColumnMeta {
  name: string;
  kind: ColumnKind;
  missing: number;
  unique: number;
}

export interface Dataset {
  id: string;
  name: string;
  bytes: number;
  addedAt: number;
  nRows: number;
  columns: ColumnMeta[];
  data: Record<string, Cell[]>;
}

export type TaskType = "regression" | "classification";

export type MissingStrategy = "drop" | "ffill" | "bfill" | "interpolate" | "mean" | "median" | "zero";
export type OutlierStrategy = "none" | "iqr" | "zscore" | "percentile";
export type EncodingStrategy = "onehot" | "label" | "drop";
export type ScalingStrategy = "none" | "standard" | "minmax";

export interface PrepOptions {
  target: string | null;
  dateCol: string | null;
  task: TaskType;
  features: string[];
  missing: MissingStrategy;
  outliers: OutlierStrategy;
  iqrK: number;
  zMax: number;
  clipTarget: boolean;
  encoding: EncodingStrategy;
  scaling: ScalingStrategy;
  testFraction: number;
}

export interface ProcessedData {
  columns: string[];
  rows: number;
  data: Record<string, Cell[]>;
  target: string;
  featureColumns: string[];
  dates: string[] | null;
  csv: string;
  report: string[];
  droppedRows: number;
  classes: string[] | null;
}

export interface SeriesPayload {
  index: number[];
  values: (number | null)[];
}

export type RunStatus = "queued" | "running" | "done" | "error";

export interface ModelResult {
  sessionId?: string;
  metrics: Record<string, number | null>;
  extras: Record<string, unknown>;
  prediction: Record<string, unknown>;
  featureColumns: string[];
  defaultEntry: Record<string, unknown>;
  train: SeriesPayload | null;
  test: SeriesPayload | null;
  testPredictions: SeriesPayload | null;
  forecast?: number[];
  history?: { loss: number[]; valLoss: number[] };
  notes: string[];
  seconds: number;
  backend?: string;
}

export interface RunState {
  status: RunStatus;
  startedAt?: number;
  finishedAt?: number;
  message?: string;
  error?: string;
  traceback?: string;
  progress?: number;
  liveLoss?: { loss: number[]; valLoss: number[] };
  result?: ModelResult;
}
