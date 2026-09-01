import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fit, evaluate, predict } from '../src/probe.ts';
import { sigmoid, trainHead, score } from '../src/logistic.ts';
import { fitScaler, transform } from '../src/scaler.ts';
import { selectThreshold, f1At } from '../src/threshold.ts';
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

/** Two labels, each a noisy linear function of a 6-D feature vector, so a
 * linear probe can learn them. Label 0 is balanced; label 1 is rare. */
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
    const z1 = features.reduce((s, x, j) => s + w1[j] * x, 0) - 2.2; // shifted rare
    out.push({
      features,
      labels: [sigmoid(z0) > rand(), sigmoid(z1) > rand()]
    });
  }
  return out;
}

test('sigmoid is stable and bounded at the extremes', () => {
  assert.ok(sigmoid(1000) <= 1 && sigmoid(1000) > 0.999);
  assert.ok(sigmoid(-1000) >= 0 && sigmoid(-1000) < 0.001);
  assert.ok(Math.abs(sigmoid(0) - 0.5) < 1e-12);
});

test('the probe learns a separable label to high holdout F1', () => {
  const train = makeData(600, 1);
  const val = makeData(200, 2);
  const holdout = makeData(400, 3);
  const model = fit(train, val, { epochs: 300 });
  const ev = evaluate(model, holdout);
  assert.ok(ev.perLabel[0].f1 > 0.75, `label 0 F1 ${ev.perLabel[0].f1.toFixed(2)}`);
  assert.ok(ev.macroF1 > 0.55, `macro F1 ${ev.macroF1.toFixed(2)}`);
});

test('training is deterministic: same data in, bit-identical model out', () => {
  const train = makeData(300, 7);
  const val = makeData(100, 8);
  const a = fit(train, val);
  const b = fit(train, val);
  assert.deepEqual(a.heads, b.heads);
  assert.deepEqual(a.thresholds, b.thresholds);
});

/** The two leakage tests below deliberately assert on `fit`, not on
 * `fitScaler`. The old test called fitScaler on a literal two-row array and
 * never invoked the pipeline, so fitting the scaler on train-union-val - or
 * training the heads on train-union-val - left the whole suite green. These
 * two fail on either mutation. */

test('fit standardises on train only: a shifted val cannot move the scaler', () => {
  const train = makeData(300, 11);
  // val features are 1000 away from train; if val entered the scaler fit at
  // all, the means would move by hundreds
  const valShifted = makeData(120, 12).map((e) => ({
    features: e.features.map((f) => f + 1000),
    labels: e.labels
  }));
  const model = fit(train, valShifted);
  assert.deepEqual(model.scaler, fitScaler(train.map((e) => e.features)));
  assert.ok(
    model.scaler.mean.every((m) => Math.abs(m) < 10),
    `train means ${model.scaler.mean.join(', ')} - val's +1000 shift leaked in`
  );
  // and the train scaler is what val and holdout are measured against
  assert.deepEqual(transform(model.scaler, model.scaler.mean), model.scaler.mean.map(() => 0));
});

test('fit trains heads on train only: flipping every val label moves no weight', () => {
  const train = makeData(300, 13);
  const val = makeData(120, 14);
  const flipped = val.map((e) => ({ features: e.features, labels: e.labels.map((l) => !l) }));
  const a = fit(train, val);
  const b = fit(train, flipped);
  assert.deepEqual(a.heads, b.heads, 'val labels reached head training');
  assert.deepEqual(a.scaler, b.scaler);
  // the thresholds DO move, which is what proves the val split is actually
  // consumed - so the head equality above is a guarantee, not an accident of
  // val being ignored
  assert.notDeepEqual(a.thresholds, b.thresholds);
});

test('threshold selection beats a fixed 0.5 on a rare label', () => {
  // scores high for positives but not above 0.5; 0.5 would miss them all
  const scores = [0.45, 0.48, 0.44, 0.1, 0.12, 0.09, 0.11, 0.08, 0.07, 0.06];
  const labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
  const at5 = f1At(scores, labels, 0.5);
  const best = selectThreshold(scores, labels);
  assert.equal(at5.f1, 0, '0.5 catches nothing');
  assert.ok(best.f1 > 0.9, `selected F1 ${best.f1}`);
  assert.ok(best.threshold < 0.5);
});

test('a constant feature does not divide by zero', () => {
  const train = [
    { features: [5, 1], labels: [true] },
    { features: [5, 3], labels: [false] } // feature 0 is constant
  ];
  const scaler = fitScaler(train.map((e) => e.features));
  assert.ok(Number.isFinite(transform(scaler, [5, 2])[0]));
});

test('nothing in the holdout can change the trained model', () => {
  const train = makeData(300, 4);
  const val = makeData(100, 5);
  const holdoutA = makeData(200, 6);
  const holdoutB = makeData(200, 9);
  const model = fit(train, val);
  // the holdout was never an input to fit, so nothing it contains - and no
  // number of looks at it - can move the model or contaminate a later
  // report. This is what the "kept out of training" claim rests on.
  const before = JSON.stringify(model);
  const first = evaluate(model, holdoutA);
  evaluate(model, holdoutB); // a different holdout in between
  for (const ex of holdoutB) {
    ex.features = ex.features.map((f) => f * 1000 + 7);
    ex.labels = ex.labels.map((l) => !l);
  }
  evaluate(model, holdoutB); // and a wildly mutated one
  assert.equal(JSON.stringify(model), before, 'the model is unchanged by anything a holdout does');
  assert.deepEqual(evaluate(model, holdoutA), first, 'evaluate carries no state between calls');
});

test('predict thresholds each label independently at its selected cut', () => {
  const train = makeData(400, 4);
  const val = makeData(150, 5);
  const model = fit(train, val);
  const p = predict(model, makeData(1, 6)[0].features);
  assert.equal(p.scores.length, 2);
  assert.equal(p.labels.length, 2);
  assert.equal(p.labels[0], p.scores[0] >= model.thresholds[0]);
  assert.equal(p.labels[1], p.scores[1] >= model.thresholds[1]);
});

test('trainHead reduces error on a trivially separable set', () => {
  const X = [
    [2, 0],
    [2, 1],
    [-2, 0],
    [-2, -1]
  ];
  const y = [1, 1, 0, 0];
  const head = trainHead(X, y, { epochs: 500 });
  assert.ok(score(head, [2, 0]) > 0.8);
  assert.ok(score(head, [-2, 0]) < 0.2);
});
