/** Shape and split hygiene: every one of these is a failure mode that
 * used to return a plausible number instead of an error. A probe that
 * answers "F1 0.00, not linearly readable" because the embedding width
 * changed is worse than a probe that refuses to answer. */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate, fit, predict } from '../src/probe.ts';
import { fitScaler, transform } from '../src/scaler.ts';
import { f1At, selectThreshold } from '../src/threshold.ts';
import { score, trainHead } from '../src/logistic.ts';
import type { Example } from '../src/probe.ts';

/** A tiny two-feature, one-label problem that fits fine on its own. */
function twoFeature(): { train: Example[]; val: Example[] } {
  const train: Example[] = [
    { features: [2, 0], labels: [true] },
    { features: [2, 1], labels: [true] },
    { features: [-2, 0], labels: [false] },
    { features: [-2, -1], labels: [false] }
  ];
  const val: Example[] = [
    { features: [1.5, 0.5], labels: [true] },
    { features: [-1.5, -0.5], labels: [false] }
  ];
  return { train, val };
}

test('the model records the feature width it was fit on', () => {
  const { train, val } = twoFeature();
  assert.equal(fit(train, val).dim, 2);
});

test('evaluate refuses a holdout of the wrong feature width', () => {
  const { train, val } = twoFeature();
  const model = fit(train, val);
  // measured before the fix: this returned { macroF1: 0, microF1: 0 } with no
  // error, i.e. a shape bug served as the scientific conclusion "not readable"
  const holdout: Example[] = [{ features: [2, 0, 9], labels: [true] }];
  assert.throws(() => evaluate(model, holdout), /holdout example 0 has 3 features/);
});

test('predict refuses a feature vector that is too short', () => {
  const { train, val } = twoFeature();
  const model = fit(train, val);
  // measured before the fix: returned a confident score from the first weight
  assert.throws(() => predict(model, [0]), /1 values, the scaler expects 2/);
});

test('predict refuses a feature vector that is too long', () => {
  const { train, val } = twoFeature();
  const model = fit(train, val);
  // measured before the fix: score was NaN, and NaN >= threshold is false, so
  // the extra column was reported as a confident negative
  assert.throws(() => predict(model, [0, 1, 9]), /3 values, the scaler expects 2/);
});

test('fit refuses ragged training features', () => {
  const train: Example[] = [
    { features: [1, 2], labels: [true] },
    { features: [3], labels: [false] }
  ];
  assert.throws(() => fit(train, train), /training example 1 has 1 features/);
});

test('fitScaler refuses ragged rows instead of producing NaN statistics', () => {
  // measured before the fix: { mean: [2, NaN], std: [1, NaN] }, which then
  // poisons every downstream score
  assert.throws(() => fitScaler([[1, 2], [3]]), /ragged|width/);
});

test('transform refuses a vector the scaler was not fit for', () => {
  const scaler = fitScaler([[1, 2], [3, 4]]);
  assert.throws(() => transform(scaler, [1]), /1 values, the scaler expects 2/);
  assert.throws(() => transform(scaler, [1, 2, 3]), /3 values, the scaler expects 2/);
});

test('fit refuses an empty validation split rather than defaulting the cut', () => {
  const { train } = twoFeature();
  // measured before the fix: thresholds silently came back as [0.5]
  assert.throws(() => fit(train, []), /validation split is empty/);
});

test('fit refuses a validation label with no positives', () => {
  const { train } = twoFeature();
  const val: Example[] = [
    { features: [1.5, 0.5], labels: [false] },
    { features: [-1.5, -0.5], labels: [false] }
  ];
  // measured before the fix: every grid point scores F1 0, so the search
  // returned grid[0] = 0.05 and the head fired on everything
  assert.throws(() => fit(train, val), /label 0.*positive|no positive/);
});

test('fit refuses a validation label with no negatives', () => {
  const { train } = twoFeature();
  const val: Example[] = [
    { features: [1.5, 0.5], labels: [true] },
    { features: [-1.5, -0.5], labels: [true] }
  ];
  assert.throws(() => fit(train, val), /label 0.*negative|no negative/);
});

test('selectThreshold refuses a degenerate label column', () => {
  // measured before the fix: -> { threshold: 0.05, f1: 0 }, a head that fires
  // on everything, reported as if it were a selected cut
  assert.throws(() => selectThreshold([0.1, 0.2, 0.3, 0.05], [0, 0, 0, 0]), /positive/);
  assert.throws(() => selectThreshold([], []), /empty/);
});

test('f1At refuses mismatched score and label arrays', () => {
  assert.throws(() => f1At([0.1, 0.2], [1], 0.5), /1 labels|length/);
});

test('score and trainHead refuse vectors of the wrong width', () => {
  assert.throws(() => score({ weights: [1, 2], bias: 0 }, [1]), /1 values, the head expects 2/);
  assert.throws(() => trainHead([[1, 2], [3]], [1, 0]), /ragged|width/);
  assert.throws(() => trainHead([[1, 2]], [1, 0]), /1 rows and 2 labels/);
  assert.throws(() => trainHead([], []), /no examples/);
});
