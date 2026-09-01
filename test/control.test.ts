/** What does an F1 of 0.70 mean?
 *
 * On this package's own demo data, a probe trained on PURE NOISE - labels
 * drawn independently of the features - reaches a holdout F1 of 0.671,
 * against real signal's 0.787. The mechanism is not subtle: an
 * F1-maximising threshold search on a label it cannot predict collapses to
 * the lowest cut in the grid, predicts everything positive, and collects
 * recall 1.0 at a precision equal to the base rate. A reader handed one
 * number cannot tell readability from class balance, which is the only
 * question the library exists to answer.
 *
 * So every report now carries the two numbers an F1 has to be read
 * against: the always-positive floor, and the control probe. */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { controlFit, controlPermutation, evaluate, fit } from '../src/probe.ts';
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

const W = [1.5, -1, 0.5, 0, 0.8, -0.3];

/** labels are a noisy linear function of the features: real signal */
function signal(n: number, seed: number): Example[] {
  const rand = seeded(seed);
  const out: Example[] = [];
  for (let i = 0; i < n; i++) {
    const features = Array.from({ length: 6 }, () => rand() * 4 - 2);
    const z = features.reduce((s, x, j) => s + W[j] * x, 0);
    out.push({ features, labels: [sigmoid(z) > rand()] });
  }
  return out;
}

/** labels are independent of the features: zero signal by construction */
function noise(n: number, seed: number, dim = 6): Example[] {
  const rand = seeded(seed);
  const out: Example[] = [];
  for (let i = 0; i < n; i++) {
    out.push({
      features: Array.from({ length: dim }, () => rand() * 4 - 2),
      labels: [rand() < 0.5]
    });
  }
  return out;
}

function controlF1(f1: number, selectivity: number | null): number {
  assert.notEqual(selectivity, null, 'no control was run');
  return f1 - (selectivity as number);
}

test('a probe on signal clears its control', () => {
  const train = signal(600, 1);
  const val = signal(200, 2);
  const ev = evaluate(fit(train, val), signal(400, 3), controlFit(train, val));
  const r = ev.perLabel[0];
  assert.ok(
    (r.selectivity as number) > 0.05,
    `F1 ${r.f1.toFixed(3)} against control ${controlF1(r.f1, r.selectivity).toFixed(3)}`
  );
  assert.ok((ev.macroSelectivity as number) > 0.05);
  // the numbers the README quotes
  assert.equal(r.f1.toFixed(3), '0.787');
  assert.equal(controlF1(r.f1, r.selectivity).toFixed(3), '0.669');
});

test('a probe on pure noise scores like a result and clears nothing', () => {
  const train = noise(600, 4);
  const val = noise(200, 5);
  const ev = evaluate(fit(train, val), noise(400, 6), controlFit(train, val));
  const r = ev.perLabel[0];
  // this is the number the headline used to be, with no signal behind it
  assert.ok(r.f1 > 0.6, `noise F1 ${r.f1.toFixed(3)} should still look like a result`);
  assert.ok(
    Math.abs(r.selectivity as number) < 0.05,
    `noise selectivity ${(r.selectivity as number).toFixed(3)} should be about zero`
  );
  // the numbers the README quotes
  assert.equal(r.f1.toFixed(3), '0.671');
  assert.equal((r.selectivity as number).toFixed(3), '0.000');
  // and the mechanism the README describes: with nothing to predict, the
  // threshold search collapses to the lowest cut, calls everything
  // positive, and lands exactly on the always-positive floor
  assert.equal(controlF1(r.f1, r.selectivity).toFixed(6), r.baselineF1.toFixed(6));
});

test('the always-positive floor is 2b/(1+b) on the holdout base rate', () => {
  const train = signal(300, 7);
  const val = signal(120, 8);
  const holdout = signal(200, 9);
  const ev = evaluate(fit(train, val), holdout);
  const r = ev.perLabel[0];
  const b = r.support / holdout.length;
  assert.equal(r.baselineF1, (2 * b) / (1 + b));
  assert.equal(ev.macroBaselineF1, r.baselineF1);
  // and it is a floor worth printing: on a balanced label it is over 0.6
  assert.ok(r.baselineF1 > 0.6, `floor ${r.baselineF1.toFixed(3)}`);
});

test('selectivity is null, not zero, when no control was run', () => {
  const train = signal(200, 7);
  const val = signal(80, 8);
  const ev = evaluate(fit(train, val), signal(100, 9));
  assert.equal(ev.perLabel[0].selectivity, null);
  assert.equal(ev.macroSelectivity, null);
});

test('the control introduces no randomness: two control fits are identical', () => {
  const train = signal(300, 7);
  const val = signal(120, 8);
  const a = controlFit(train, val);
  const b = controlFit(train, val);
  assert.deepEqual(a.heads, b.heads);
  assert.deepEqual(a.thresholds, b.thresholds);
  // and it is a different model from the real one, or it would prove nothing
  assert.notDeepEqual(a.heads, fit(train, val).heads);
});

test('the control permutation is a bijection and never the identity', () => {
  for (const n of [1, 2, 3, 5, 7, 8, 9, 12, 25, 49, 100, 600]) {
    const p = controlPermutation(n);
    assert.equal(p.length, n);
    assert.deepEqual(
      [...p].sort((x, y) => x - y),
      Array.from({ length: n }, (_, i) => i),
      `n=${n} is not a permutation of 0..${n - 1}`
    );
    if (n > 1) assert.ok(p.some((v, i) => v !== i), `n=${n} permutes nothing`);
  }
});

test('the control permutes rows, so no label loses or gains a positive', () => {
  // a val split with exactly one positive: resampling would lose or
  // duplicate it, and fit would then reject the control for having no
  // positives to select a threshold on
  const val: Example[] = Array.from({ length: 20 }, (_, i) => ({
    features: [i / 10, 1 - i / 10],
    labels: [i === 13]
  }));
  const train: Example[] = Array.from({ length: 40 }, (_, i) => ({
    features: [i / 20, 1 - i / 20],
    labels: [i % 4 === 0]
  }));
  assert.doesNotThrow(() => controlFit(train, val));
});

test('the control is not a restatement of the floor', () => {
  // over-parameterised: d=768 on n=120, where the head memorises the
  // permuted labels and the control stops tracking the class balance.
  // Measured: control 0.608 against a floor of 0.659.
  const train = noise(120, 31, 768);
  const val = noise(60, 32, 768);
  const ev = evaluate(fit(train, val), noise(120, 33, 768), controlFit(train, val));
  const r = ev.perLabel[0];
  const control = controlF1(r.f1, r.selectivity);
  assert.ok(
    Math.abs(control - r.baselineF1) > 0.02,
    `control ${control.toFixed(3)} and floor ${r.baselineF1.toFixed(3)} are the same number here`
  );
});

test('evaluate refuses a control probe of a different shape', () => {
  const train = signal(200, 7);
  const val = signal(80, 8);
  const model = fit(train, val);
  const wide = fit(noise(200, 31, 12), noise(80, 32, 12));
  assert.throws(() => evaluate(model, signal(100, 9), wide), /control probe is 1 labels x 12/);
});
