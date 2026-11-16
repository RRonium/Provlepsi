// Small synthetic datasets so the studio can be tried without a file.
// They are generated in the browser and clearly labelled as synthetic.

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function gauss(r: () => number) {
  const u = Math.max(r(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

const iso = (d: Date) => d.toISOString().slice(0, 10);

export interface Sample {
  id: string;
  name: string;
  description: string;
  task: "regression" | "classification";
  build: () => string;
}

export const SAMPLES: Sample[] = [
  {
    id: "retail",
    name: "Daily store sales",
    description: "730 days · trend, weekly cycle, temperature and promotions",
    task: "regression",
    build() {
      const r = rng(7);
      const rows = ["date,temperature,promotion,holiday,sales"];
      const start = new Date(Date.UTC(2024, 0, 1));
      for (let i = 0; i < 730; i++) {
        const d = new Date(start.getTime() + i * 864e5);
        const temp = 22 + 9 * Math.sin((2 * Math.PI * (i - 100)) / 365) + gauss(r) * 1.8;
        const promo = r() < 0.18 ? 1 : 0;
        const holiday = d.getUTCDay() === 0 && r() < 0.15 ? 1 : 0;
        const weekly = [0.82, 0.95, 0.97, 1, 1.06, 1.28, 1.18][d.getUTCDay()];
        const sales = (420 + i * 0.35 + temp * 4.2 + promo * 85 - holiday * 60) * weekly + gauss(r) * 22;
        const missing = r() < 0.01;
        rows.push(`${iso(d)},${temp.toFixed(2)},${promo},${holiday},${missing ? "" : sales.toFixed(1)}`);
      }
      return rows.join("\n");
    },
  },
  {
    id: "index",
    name: "Stock index (daily close)",
    description: "500 trading days · volatile random walk with volume",
    task: "regression",
    build() {
      const r = rng(42);
      const rows = ["date,open,volume,close"];
      let price = 18000;
      const d = new Date(Date.UTC(2023, 0, 2));
      let n = 0;
      while (n < 500) {
        d.setUTCDate(d.getUTCDate() + 1);
        if (d.getUTCDay() === 0 || d.getUTCDay() === 6) continue;
        const open = price * (1 + gauss(r) * 0.003);
        price = open * (1 + 0.0004 + gauss(r) * 0.011);
        const volume = Math.round(2.4e8 * (1 + Math.abs(gauss(r)) * 0.6));
        rows.push(`${iso(d)},${open.toFixed(2)},${volume},${price.toFixed(2)}`);
        n++;
      }
      return rows.join("\n");
    },
  },
  {
    id: "weather",
    name: "Weather conditions",
    description: "600 days · classify sunny, cloudy or rainy from readings",
    task: "classification",
    build() {
      const r = rng(11);
      const rows = ["date,humidity,pressure,wind_speed,temperature,condition"];
      const start = new Date(Date.UTC(2024, 0, 1));
      let humidity = 60;
      for (let i = 0; i < 600; i++) {
        const d = new Date(start.getTime() + i * 864e5);
        humidity = Math.min(100, Math.max(15, humidity * 0.7 + 18 + gauss(r) * 10));
        const pressure = 1013 - (humidity - 60) * 0.35 + gauss(r) * 4;
        const wind = Math.abs(8 + gauss(r) * 5 + (humidity > 75 ? 6 : 0));
        const temp = 24 + 8 * Math.sin((2 * Math.PI * (i - 90)) / 365) - (humidity - 60) * 0.08 + gauss(r) * 2;
        const score = (humidity - 60) / 12 + (1010 - pressure) / 6 + gauss(r) * 0.6;
        const condition = score > 1.2 ? "rainy" : score > -0.2 ? "cloudy" : "sunny";
        rows.push(`${iso(d)},${humidity.toFixed(1)},${pressure.toFixed(1)},${wind.toFixed(1)},${temp.toFixed(1)},${condition}`);
      }
      return rows.join("\n");
    },
  },
];
