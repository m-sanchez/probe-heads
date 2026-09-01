/** Does the head that comes back actually solve its own objective?
 *
 * The README tells a reader that a low score means "not linearly
 * readable". It can also mean the head never finished training, and until
 * `fit` reported a convergence number there was no way to tell the two
 * apart. At the demo's 6-D scale the default 300 epochs reach a gradient
 * norm of 4.2e-14; at d=64 they reach 1.1e-3, and at d=128 5.1e-3.
 *
 * Every gradient norm asserted here is recomputed in this file from the
 * public API, not read off the model - the library does not get to grade
 * its own homework. */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fit } from '../src/probe.ts';
import { transform } from '../src/scaler.ts';
import { score, trainHead } from '../src/logistic.ts';
import type { Head } from '../src/logistic.ts';
import type { Example, ProbeModel } from '../src/probe.ts';

/** Integer-only PRNG: no Math.log or Math.cos, so the fixtures below are
 * identical on every engine. */
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

const L2 = 1e-3;

/** A wide problem: at d=64 and above, 300 epochs of full-batch descent do
 * not get near stationarity, which is the realistic embedding case. */
function wide(n: number, seed: number, dim = 64): Example[] {
  const rand = seeded(seed);
  const w = Array.from({ length: dim }, (_, j) => ((j % 7) - 3) / 3);
  const out: Example[] = [];
  for (let i = 0; i < n; i++) {
    const features = Array.from({ length: dim }, () => rand() * 4 - 2);
    const z = features.reduce((s, x, j) => s + w[j] * x, 0) / 8;
    out.push({ features, labels: [1 / (1 + Math.exp(-z)) > rand()] });
  }
  return out;
}

/** The L2 norm of the gradient of the objective trainHead minimises:
 * mean log loss plus (l2/2)|w|^2, the penalty on the weights only. */
function gradNorm(head: Head, X: number[][], y: number[], l2: number): number {
  const n = X.length;
  const d = head.weights.length;
  const sumW = new Array<number>(d).fill(0);
  let sumB = 0;
  for (let i = 0; i < n; i++) {
    const err = score(head, X[i]) - y[i];
    for (let j = 0; j < d; j++) sumW[j] += err * X[i][j];
    sumB += err;
  }
  let sq = (sumB / n) * (sumB / n);
  for (let j = 0; j < d; j++) {
    const g = sumW[j] / n + l2 * head.weights[j];
    sq += g * g;
  }
  return Math.sqrt(sq);
}

function trainingMatrix(model: ProbeModel, train: Example[]): { X: number[][]; y: number[] } {
  return {
    X: train.map((e) => transform(model.scaler, e.features)),
    y: train.map((e) => (e.labels[0] ? 1 : 0))
  };
}

test('the returned head is stationary to the tolerance that was asked for', () => {
  const train = wide(200, 21);
  const val = wide(70, 22);
  const tolerance = 1e-6;
  const model = fit(train, val, { tolerance, epochs: 5000 });
  const { X, y } = trainingMatrix(model, train);
  const norm = gradNorm(model.heads[0], X, y, L2);
  assert.ok(norm <= tolerance, `gradient norm ${norm.toExponential(3)} > tolerance ${tolerance}`);
  assert.equal(model.convergence[0].converged, true);
  assert.ok(
    model.convergence[0].epochs < 5000,
    `ran the full budget (${model.convergence[0].epochs}) instead of stopping at the tolerance`
  );
  // the README says "about 1,100 epochs" for this shape
  const taken = model.convergence[0].epochs;
  assert.ok(taken > 800 && taken < 1400, `took ${taken} epochs`);
});

test('at embedding width the default budget is nowhere near stationary', () => {
  // d=768, the width of a sentence-transformer embedding. The README's
  // Honest limits section quotes this number.
  const train = wide(200, 21, 768);
  const val = wide(70, 22, 768);
  const model = fit(train, val);
  const { X, y } = trainingMatrix(model, train);
  assert.equal(gradNorm(model.heads[0], X, y, L2).toExponential(1), '2.4e-3');
});

test('fit says so when the epoch budget runs out short of the tolerance', () => {
  const train = wide(200, 21);
  const val = wide(70, 22);
  const tolerance = 1e-6;
  // the default 300 epochs at d=64: measured gradient norm 1.145e-3, which
  // used to be reported as a finished probe with no caveat attached
  const model = fit(train, val, { tolerance });
  const { X, y } = trainingMatrix(model, train);
  const norm = gradNorm(model.heads[0], X, y, L2);
  // optional access so a model with no convergence record fails with the
  // measured number rather than a TypeError
  assert.ok(
    model.convergence?.[0]?.converged === false,
    `asked for a gradient norm <= ${tolerance}, got ${norm.toExponential(3)}, ` +
      `and the model reports ${JSON.stringify(model.convergence?.[0])}`
  );
  assert.ok(norm > tolerance, 'pick a harder case: this one converges by default');
  assert.equal(model.convergence[0].epochs, 300);
  assert.equal(model.convergence[0].finalGradNorm, norm);
  // the number the README's Honest limits section quotes for d=64
  assert.equal(norm.toExponential(1), '1.1e-3');
});

test('the reported gradient norm is the real one at the returned weights', () => {
  const train = wide(200, 21);
  const val = wide(70, 22);
  const model = fit(train, val, { epochs: 300 });
  const { X, y } = trainingMatrix(model, train);
  const recomputed = gradNorm(model.heads[0], X, y, L2);
  assert.equal(model.convergence[0].finalGradNorm, recomputed);
  assert.ok(model.convergence[0].finalLoss > 0 && model.convergence[0].finalLoss < 1);
});

test('convergence is recorded per head', () => {
  const train = wide(200, 21).map((e, i) => ({ features: e.features, labels: [e.labels[0], i % 3 === 0] }));
  const val = wide(70, 22).map((e, i) => ({ features: e.features, labels: [e.labels[0], i % 3 === 0] }));
  const model = fit(train, val);
  assert.equal(model.convergence.length, 2);
  assert.notEqual(model.convergence[0].finalLoss, model.convergence[1].finalLoss);
});

test('the default tolerance of 0 leaves v1 numbers untouched', () => {
  const train = wide(200, 21);
  const val = wide(70, 22);
  // no early stop unless a tolerance is asked for, so a probe run before
  // this feature existed returns the same weights it always did
  const plain = fit(train, val, { epochs: 300 });
  const explicit = fit(train, val, { epochs: 300, tolerance: 0 });
  assert.deepEqual(plain.heads, explicit.heads);
  assert.equal(plain.convergence[0].epochs, 300);
  assert.equal(plain.convergence[0].tolerance, 0);
});

test('trainHead reports where it stopped', () => {
  const X = [
    [2, 0],
    [2, 1],
    [-2, 0],
    [-2, -1]
  ];
  const y = [1, 1, 0, 0];
  const result = trainHead(X, y, { epochs: 500 });
  assert.ok(score(result.head, [2, 0]) > 0.8);
  assert.ok(score(result.head, [-2, 0]) < 0.2);
  assert.equal(result.convergence.epochs, 500);
  assert.equal(result.convergence.finalGradNorm, gradNorm(result.head, X, y, L2));
});
