/// <reference lib="webworker" />
// Deep-learning forecasters (LSTM, GRU, Bi-LSTM, LSTM-GRU hybrid, Transformer)
// trained with TensorFlow.js in a Web Worker. Uses WebGL when the browser offers
// it in workers and falls back to the CPU backend otherwise.
import * as Comlink from "comlink";
import * as tf from "@tensorflow/tfjs";

export type DeepArch = "lstm" | "gru" | "bilstm" | "lstm_gru" | "transformer";

export interface DeepParams {
  window: number;
  units: number;
  layers?: number;
  heads?: number;
  blocks?: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  dropout: number;
  forecastSteps: number;
}

export interface DeepProgress {
  epoch: number;
  epochs: number;
  loss: number;
  valLoss: number | null;
}

export interface DeepResult {
  backend: string;
  metrics: Record<string, number>;
  train: { index: number[]; values: number[] };
  test: { index: number[]; values: number[] };
  testPredictions: { index: number[]; values: number[] };
  forecast: number[];
  history: { loss: number[]; valLoss: number[] };
  stopped: boolean;
}

// ----------------------------------------------------------------------------
// Transformer building blocks (tfjs has no built-in attention layer)
// ----------------------------------------------------------------------------

class PositionalEncoding extends tf.layers.Layer {
  static className = "ProvlepsiPositionalEncoding";
  computeOutputShape(shape: tf.Shape | tf.Shape[]) {
    return shape;
  }
  call(inputs: tf.Tensor | tf.Tensor[]) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const [, len, dim] = x.shape as number[];
      const pe = new Float32Array(len * dim);
      for (let pos = 0; pos < len; pos++) {
        for (let i = 0; i < dim; i++) {
          const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dim);
          pe[pos * dim + i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
        }
      }
      return tf.add(x, tf.tensor3d(pe, [1, len, dim]));
    });
  }
}

class MultiHeadSelfAttention extends tf.layers.Layer {
  static className = "ProvlepsiMultiHeadSelfAttention";
  private heads: number;
  private dim = 0;
  private wq!: tf.LayerVariable;
  private wk!: tf.LayerVariable;
  private wv!: tf.LayerVariable;
  private wo!: tf.LayerVariable;

  constructor(config: { heads: number; name?: string }) {
    super({ name: config.name });
    this.heads = config.heads;
  }

  build(inputShape: tf.Shape | tf.Shape[]) {
    const shape = (Array.isArray(inputShape[0]) ? inputShape[0] : inputShape) as number[];
    this.dim = shape[shape.length - 1];
    if (this.dim % this.heads !== 0) throw new Error(`Model width ${this.dim} must divide evenly into ${this.heads} heads.`);
    const init = tf.initializers.glorotUniform({});
    this.wq = this.addWeight("wq", [this.dim, this.dim], "float32", init);
    this.wk = this.addWeight("wk", [this.dim, this.dim], "float32", init);
    this.wv = this.addWeight("wv", [this.dim, this.dim], "float32", init);
    this.wo = this.addWeight("wo", [this.dim, this.dim], "float32", init);
    this.built = true;
  }

  computeOutputShape(shape: tf.Shape | tf.Shape[]) {
    return shape;
  }

  call(inputs: tf.Tensor | tf.Tensor[]) {
    return tf.tidy(() => {
      const x = (Array.isArray(inputs) ? inputs[0] : inputs) as tf.Tensor3D;
      const [b, len, d] = x.shape;
      const h = this.heads;
      const dk = d / h;
      const flat = x.reshape([b * len, d]);
      const split = (w: tf.LayerVariable) =>
        tf.matMul(flat, w.read()).reshape([b, len, h, dk]).transpose([0, 2, 1, 3]).reshape([b * h, len, dk]) as tf.Tensor3D;
      const q = split(this.wq);
      const k = split(this.wk);
      const v = split(this.wv);
      const scores = tf.matMul(q, k, false, true).div(Math.sqrt(dk));
      const attn = tf.softmax(scores);
      const ctx = tf.matMul(attn, v).reshape([b, h, len, dk]).transpose([0, 2, 1, 3]).reshape([b * len, d]);
      return tf.matMul(ctx, this.wo.read()).reshape([b, len, d]);
    });
  }

  getConfig() {
    return { ...super.getConfig(), heads: this.heads };
  }
}

tf.serialization.registerClass(PositionalEncoding);
tf.serialization.registerClass(MultiHeadSelfAttention);

// ----------------------------------------------------------------------------

function buildModel(arch: DeepArch, p: DeepParams): tf.LayersModel {
  const input = tf.input({ shape: [p.window, 1] });
  const drop = (x: tf.SymbolicTensor) => (p.dropout > 0 ? (tf.layers.dropout({ rate: p.dropout }).apply(x) as tf.SymbolicTensor) : x);
  let x: tf.SymbolicTensor = input;

  if (arch === "lstm" || arch === "gru") {
    const n = Math.max(1, p.layers ?? 1);
    for (let i = 0; i < n; i++) {
      const cfg = { units: p.units, returnSequences: i < n - 1 };
      x = (arch === "lstm" ? tf.layers.lstm(cfg) : tf.layers.gru(cfg)).apply(x) as tf.SymbolicTensor;
      x = drop(x);
    }
  } else if (arch === "bilstm") {
    x = tf.layers
      .bidirectional({ layer: tf.layers.lstm({ units: p.units }) as unknown as tf.RNN, mergeMode: "concat" })
      .apply(x) as tf.SymbolicTensor;
    x = drop(x);
  } else if (arch === "lstm_gru") {
    x = tf.layers.lstm({ units: p.units, returnSequences: true }).apply(x) as tf.SymbolicTensor;
    x = drop(x);
    x = tf.layers.gru({ units: Math.max(4, Math.round(p.units / 2)) }).apply(x) as tf.SymbolicTensor;
    x = drop(x);
  } else {
    const heads = Math.max(1, p.heads ?? 4);
    const dModel = Math.max(heads, Math.round(p.units / heads) * heads);
    x = tf.layers.dense({ units: dModel }).apply(x) as tf.SymbolicTensor;
    x = new PositionalEncoding({}).apply(x) as tf.SymbolicTensor;
    for (let i = 0; i < Math.max(1, p.blocks ?? 2); i++) {
      const a = drop(new MultiHeadSelfAttention({ heads }).apply(x) as tf.SymbolicTensor);
      x = tf.layers.layerNormalization().apply(tf.layers.add().apply([x, a])) as tf.SymbolicTensor;
      let f = tf.layers.dense({ units: dModel * 2, activation: "relu" }).apply(x) as tf.SymbolicTensor;
      f = drop(tf.layers.dense({ units: dModel }).apply(f) as tf.SymbolicTensor);
      x = tf.layers.layerNormalization().apply(tf.layers.add().apply([x, f])) as tf.SymbolicTensor;
    }
    x = tf.layers.globalAveragePooling1d().apply(x) as tf.SymbolicTensor;
  }

  const out = tf.layers.dense({ units: 1 }).apply(x) as tf.SymbolicTensor;
  const model = tf.model({ inputs: input, outputs: out });
  model.compile({ optimizer: tf.train.adam(p.learningRate), loss: "meanSquaredError" });
  return model;
}

interface Trained {
  model: tf.LayersModel;
  scaled: Float32Array;
  lo: number;
  span: number;
  window: number;
}

const trained = new Map<string, Trained>();
let current: tf.LayersModel | null = null;
let stopRequested = false;
let backendName: string | null = null;

export type BackendPref = "auto" | "webgl" | "cpu";

/** True when WebGL is emulated in software (e.g. SwiftShader), where the CPU backend is faster. */
function softwareWebGL(): boolean {
  try {
    const gl = new OffscreenCanvas(1, 1).getContext("webgl2") as WebGL2RenderingContext | null;
    if (!gl) return true;
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = String(info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    return /swiftshader|llvmpipe|software|basic render/i.test(renderer);
  } catch {
    return true;
  }
}

async function ensureBackend(pref: BackendPref = "auto") {
  const order = pref === "cpu" ? ["cpu"] : pref === "webgl" ? ["webgl", "cpu"] : softwareWebGL() ? ["cpu"] : ["webgl", "cpu"];
  if (backendName && backendName === order[0]) return backendName;
  for (const b of order) {
    try {
      if (await tf.setBackend(b)) {
        await tf.ready();
        backendName = tf.getBackend();
        return backendName;
      }
    } catch {
      // try the next backend
    }
  }
  backendName = tf.getBackend();
  return backendName;
}

function recursiveForecast(t: Trained, steps: number): number[] {
  return tf.tidy(() => {
    const win = Array.from(t.scaled.slice(t.scaled.length - t.window));
    const out: number[] = [];
    for (let s = 0; s < steps; s++) {
      const next = (t.model.predict(tf.tensor3d(win, [1, t.window, 1])) as tf.Tensor).dataSync()[0];
      out.push(next * t.span + t.lo);
      win.shift();
      win.push(next);
    }
    return out;
  });
}

const api = {
  async backend(pref: BackendPref = "auto") {
    return ensureBackend(pref);
  },

  async train(
    runId: string,
    arch: DeepArch,
    series: number[],
    testFraction: number,
    p: DeepParams,
    onProgress?: (p: DeepProgress) => void,
    backendPref: BackendPref = "auto",
  ): Promise<DeepResult> {
    const backend = await ensureBackend(backendPref);
    stopRequested = false;
    const n = series.length;
    const split = Math.floor(n * (1 - testFraction));
    const w = Math.round(p.window);
    if (split - w < 10) throw new Error(`The look-back window (${w}) is too long for ${split} training rows. Try a window under ${Math.max(2, split - 10)}.`);
    if (n - split < 1) throw new Error("The test window is empty. Increase the test fraction.");

    const trainPart = series.slice(0, split);
    const lo = Math.min(...trainPart);
    const span = Math.max(...trainPart) - lo || 1;
    const scaled = Float32Array.from(series, (v) => (v - lo) / span);

    const makeWindows = (from: number, to: number) => {
      const count = to - from;
      const xs = new Float32Array(count * w);
      const ys = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        const t = from + i;
        xs.set(scaled.subarray(t - w, t), i * w);
        ys[i] = scaled[t];
      }
      return { x: tf.tensor3d(xs, [count, w, 1]), y: tf.tensor2d(ys, [count, 1]), count };
    };
    const tr = makeWindows(w, split);
    const te = makeWindows(Math.max(w, split), n);

    const model = buildModel(arch, p);
    current = model;
    const history = { loss: [] as number[], valLoss: [] as number[] };
    await model.fit(tr.x, tr.y, {
      epochs: Math.round(p.epochs),
      batchSize: Math.round(p.batchSize),
      shuffle: true,
      validationData: te.count ? [te.x, te.y] : undefined,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const loss = Number(logs?.loss ?? NaN);
          const valLoss = logs?.val_loss !== undefined ? Number(logs.val_loss) : null;
          history.loss.push(loss);
          if (valLoss !== null) history.valLoss.push(valLoss);
          onProgress?.({ epoch: epoch + 1, epochs: Math.round(p.epochs), loss, valLoss });
          if (stopRequested) model.stopTraining = true;
          await tf.nextFrame?.();
        },
      },
    });

    const predScaled = (model.predict(te.x) as tf.Tensor).dataSync();
    const testIdx = Array.from({ length: te.count }, (_, i) => Math.max(w, split) + i);
    const actual = testIdx.map((t) => series[t]);
    const predicted = Array.from(predScaled, (v) => v * span + lo);
    let abs = 0;
    let sq = 0;
    actual.forEach((a, i) => {
      abs += Math.abs(a - predicted[i]);
      sq += (a - predicted[i]) ** 2;
    });
    const MAE = abs / actual.length;
    const MSE = sq / actual.length;
    tf.dispose([tr.x, tr.y, te.x, te.y]);

    const record: Trained = { model, scaled, lo, span, window: w };
    const old = trained.get(runId);
    if (old && old.model !== model) old.model.dispose();
    trained.set(runId, record);
    current = null;

    const trainIdx = Array.from({ length: split }, (_, i) => i);
    return {
      backend,
      metrics: { MAE, MSE, RMSE: Math.sqrt(MSE) },
      train: { index: trainIdx, values: trainPart },
      test: { index: testIdx, values: actual },
      testPredictions: { index: testIdx, values: predicted },
      forecast: recursiveForecast(record, Math.round(p.forecastSteps)),
      history,
      stopped: stopRequested,
    };
  },

  async forecast(runId: string, steps: number): Promise<number[]> {
    const t = trained.get(runId);
    if (!t) throw new Error("This network is no longer in memory. Train it again to forecast.");
    return recursiveForecast(t, Math.round(steps));
  },

  stop() {
    stopRequested = true;
    if (current) current.stopTraining = true;
  },

  reset() {
    for (const t of trained.values()) t.model.dispose();
    trained.clear();
  },
};

export type DeepApi = typeof api;
Comlink.expose(api);
