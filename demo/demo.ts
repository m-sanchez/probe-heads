/** npm run demo: probe two labels off a frozen 6-D embedding, with the
 * train/val/holdout discipline the API enforces. Synthetic, seeded. */

import { fit, evaluate } from '../src/probe.ts';
import { sigmoid } from '../src/logistic.ts';
import type { Example } from '../src/probe.ts';

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z ^= z >>> 16;
    z = Math.imul(z, 0x21f0aaad);
    z ^= z >>> 15;
    z = Math.imul(z, 0x735a2d97);
    z ^= z >>> 15;
    return (z >>> 0) / 0x100000000;
  };
}

function makeData(n: number, seed: number): Example[] {
  const rand = seeded(seed);
  const gauss = () => {
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const w0 = [1.5, -1, 0.5, 0, 0.8, -0.3];
  const w1 = [-0.5, 0.2, 1.2, -1.4, 0, 0.6];
  const out: Example[] = [];
  for (let i = 0; i < n; i++) {
    const features = Array.from({ length: 6 }, () => gauss());
    const z0 = features.reduce((s, x, j) => s + w0[j] * x, 0);
    const z1 = features.reduce((s, x, j) => s + w1[j] * x, 0) - 2.2;
    out.push({ features, labels: [sigmoid(z0) > rand(), sigmoid(z1) > rand()] });
  }
  return out;
}

const model = fit(makeData(600, 1), makeData(200, 2), { epochs: 300 });
const ev = evaluate(model, makeData(400, 3));

console.log('two labels probed off a frozen 6-D embedding (balanced, rare)\n');
console.log('label   threshold   P      R      F1     support');
ev.perLabel.forEach((r, k) => {
  console.log(
    `${String(k).padEnd(7)} ${model.thresholds[k].toFixed(2).padEnd(11)} ` +
      `${r.precision.toFixed(2)}   ${r.recall.toFixed(2)}   ${r.f1.toFixed(2)}   ${r.support}`
  );
});
console.log(`\nmacro F1 ${ev.macroF1.toFixed(3)}, micro F1 ${ev.microF1.toFixed(3)}`);
console.log('thresholds were chosen on validation; this is the first look at the holdout.');
