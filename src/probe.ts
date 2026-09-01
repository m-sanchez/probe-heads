/** The probe: deterministic one-vs-rest logistic heads on frozen feature
 * vectors, with train/val/holdout hygiene wired into the API.
 *
 * The shape enforces the discipline. `fit` sees train and val only - it
 * standardises on train, trains a head per label on train, and selects
 * each label's threshold on val. The holdout is not an argument to `fit`;
 * it goes to `evaluate`, once, at the end. A frozen embedding in, an
 * honest per-label report out, the same every run.
 *
 * Every split is checked against the model's feature width and label
 * count, and a validation split that cannot select a threshold (empty, or
 * a label with only one class present) is rejected rather than quietly
 * defaulted. A wrong answer that looks like a result is worse than an
 * error. */

import { fitScaler, transform } from './scaler.ts';
import type { Scaler } from './scaler.ts';
import { score, trainHead } from './logistic.ts';
import type { Convergence, Head, TrainOptions } from './logistic.ts';
import { defaultGrid, f1At, selectThreshold } from './threshold.ts';

export interface Example {
  /** the frozen feature vector (an embedding, say) */
  features: number[];
  /** one boolean per class; a multi-label, one-vs-rest target */
  labels: boolean[];
}

export interface ProbeModel {
  scaler: Scaler;
  heads: Head[];
  /** the val-selected decision threshold per label */
  thresholds: number[];
  labelCount: number;
  /** the feature width every vector must have, in `fit`, `predict` and
   * `evaluate` alike */
  dim: number;
  /** what the optimiser did for each head. A head that stopped a long way
   * from stationarity is an underfit probe, not a verdict on the
   * embedding, and this is where you find out which one you have. */
  convergence: Convergence[];
}

export interface FitOptions extends TrainOptions {
  thresholdGrid?: number[];
}

function assertConsistent(
  examples: Example[],
  labelCount: number,
  dim: number,
  split: string
): void {
  for (let i = 0; i < examples.length; i++) {
    const e = examples[i];
    if (e.labels.length !== labelCount) {
      throw new Error(
        `${split} example ${i} has ${e.labels.length} labels, the model expects ${labelCount}`
      );
    }
    if (e.features.length !== dim) {
      throw new Error(
        `${split} example ${i} has ${e.features.length} features, the model expects ${dim}`
      );
    }
  }
}

/** A validation split has to be able to answer the question it is asked:
 * which cut maximises F1 for this label. With no positives (or no
 * negatives) it cannot, and the search would return an artefact of the
 * grid. */
function assertSelectable(val: Example[], labelCount: number): void {
  if (val.length === 0) {
    throw new Error('validation split is empty: thresholds are selected on validation data');
  }
  for (let k = 0; k < labelCount; k++) {
    let positives = 0;
    for (const e of val) if (e.labels[k]) positives++;
    if (positives === 0) {
      throw new Error(`validation label ${k} has no positive examples: no threshold to select`);
    }
    if (positives === val.length) {
      throw new Error(`validation label ${k} has no negative examples: no threshold to select`);
    }
  }
}

/** Fit heads on train, choose thresholds on val. The holdout is not
 * accepted here, by design. */
export function fit(train: Example[], val: Example[], opts: FitOptions = {}): ProbeModel {
  if (train.length === 0) throw new Error('training split is empty');
  const labelCount = train[0].labels.length;
  if (labelCount === 0) throw new Error('training examples carry no labels');
  const dim = train[0].features.length;
  if (dim === 0) throw new Error('training examples carry no features');
  assertConsistent(train, labelCount, dim, 'training');
  assertConsistent(val, labelCount, dim, 'validation');
  assertSelectable(val, labelCount);

  const scaler = fitScaler(train.map((e) => e.features));
  const xTrain = train.map((e) => transform(scaler, e.features));
  const xVal = val.map((e) => transform(scaler, e.features));
  const grid = opts.thresholdGrid ?? defaultGrid();

  const heads: Head[] = [];
  const thresholds: number[] = [];
  const convergence: Convergence[] = [];
  for (let k = 0; k < labelCount; k++) {
    const yTrain = train.map((e) => (e.labels[k] ? 1 : 0));
    const trained = trainHead(xTrain, yTrain, opts);
    heads.push(trained.head);
    convergence.push(trained.convergence);
    const valScores = xVal.map((x) => score(trained.head, x));
    const yVal = val.map((e) => (e.labels[k] ? 1 : 0));
    thresholds.push(selectThreshold(valScores, yVal, grid).threshold);
  }
  return { scaler, heads, thresholds, labelCount, dim, convergence };
}

/** The permutation the control probe uses: p(i) = (stride * i + 1) mod n,
 * where stride is the smallest integer >= 7 that is coprime with n, so p
 * is a bijection and never the identity. It is arithmetic, not sampling -
 * there is no RNG anywhere in this package and a control run is exactly as
 * reproducible as the real one.
 *
 * It is a permutation of rows, so every label keeps its class balance and
 * its correlations with the other labels exactly; only the link between a
 * feature vector and its labels is broken. If your rows arrive in an order
 * that this permutation happens to line up with - sorted by label, say -
 * shuffle them once before splitting, or the control is not a null. */
export function controlPermutation(n: number): number[] {
  if (n <= 0) return [];
  let stride = 7;
  while (gcd(stride, n) !== 1) stride++;
  const p = new Array<number>(n);
  for (let i = 0; i < n; i++) p[i] = (stride * i + 1) % n;
  return p;
}

function gcd(a: number, b: number): number {
  while (b !== 0) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

function permuteLabels(examples: Example[]): Example[] {
  const p = controlPermutation(examples.length);
  return examples.map((e, i) => ({ features: e.features, labels: examples[p[i]].labels }));
}

/** The control probe: the identical pipeline on the identical features,
 * with the labels permuted so that no feature-label signal survives.
 *
 * Its holdout F1 is what this probe scores on a property that is not there
 * at all. On the demo's data a probe of pure noise still reaches F1 0.671
 * against real signal's 0.787, because an F1-maximising threshold search
 * on a balanced label collapses towards predicting everything positive.
 * Read a probe's F1 against this number, not against zero. */
export function controlFit(train: Example[], val: Example[], opts: FitOptions = {}): ProbeModel {
  return fit(permuteLabels(train), permuteLabels(val), opts);
}

export interface Prediction {
  /** the head's raw sigmoid output per label. It is a score, not a
   * calibrated probability: the head is L2-regularised and the cut is
   * chosen away from 0.5, so rank it, threshold it, but do not read it as
   * "70% likely". */
  scores: number[];
  /** thresholded decision per label */
  labels: boolean[];
}

export function predict(model: ProbeModel, features: number[]): Prediction {
  const x = transform(model.scaler, features);
  const scores = model.heads.map((h) => score(h, x));
  return { scores, labels: scores.map((s, k) => s >= model.thresholds[k]) };
}

export interface LabelReport {
  precision: number;
  recall: number;
  f1: number;
  /** number of positive examples for this label in the holdout */
  support: number;
  /** what a classifier that answers "yes" to everything scores on this
   * label: 2b/(1+b) for base rate b. The floor an F1 has to clear before
   * it means anything. */
  baselineF1: number;
  /** f1 minus the control probe's f1 on the same holdout with the labels
   * permuted - how much of this score is the property rather than the
   * class balance and the head's capacity. `null` when `evaluate` was
   * called without a control, which is itself the honest answer: not
   * measured. */
  selectivity: number | null;
}

export interface Evaluation {
  perLabel: LabelReport[];
  /** unweighted mean F1 across labels */
  macroF1: number;
  /** F1 pooled over all label decisions */
  microF1: number;
  /** unweighted mean of the always-positive floor */
  macroBaselineF1: number;
  /** macroF1 minus the control's macroF1, or null if no control was given */
  macroSelectivity: number | null;
}

/** Evaluate on the holdout, once, at the val-selected thresholds.
 *
 * Pass the model from `controlFit` as `control` and every label gets a
 * selectivity: this probe's F1 minus what the same pipeline scores when
 * the labels carry no signal. The control is scored against the control
 * task - the holdout labels under the same permutation - which is the
 * comparison that makes the difference meaningful. */
export function evaluate(model: ProbeModel, holdout: Example[], control?: ProbeModel): Evaluation {
  assertConsistent(holdout, model.labelCount, model.dim, 'holdout');
  if (holdout.length === 0) throw new Error('holdout split is empty: there is nothing to report');
  let controlEval: Evaluation | null = null;
  if (control) {
    if (control.labelCount !== model.labelCount || control.dim !== model.dim) {
      throw new Error(
        `control probe is ${control.labelCount} labels x ${control.dim} features, ` +
          `the model is ${model.labelCount} x ${model.dim}`
      );
    }
    controlEval = evaluate(control, permuteLabels(holdout));
  }
  const xs = holdout.map((e) => transform(model.scaler, e.features));
  const perLabel: LabelReport[] = [];
  let microTp = 0;
  let microFp = 0;
  let microFn = 0;
  for (let k = 0; k < model.labelCount; k++) {
    const scores = xs.map((x) => score(model.heads[k], x));
    const labels = holdout.map((e) => (e.labels[k] ? 1 : 0));
    const point = f1At(scores, labels, model.thresholds[k]);
    const support = labels.reduce<number>((a, b) => a + b, 0);
    const base = support / holdout.length;
    perLabel.push({
      precision: point.precision,
      recall: point.recall,
      f1: point.f1,
      support,
      baselineF1: (2 * base) / (1 + base),
      selectivity: controlEval === null ? null : point.f1 - controlEval.perLabel[k].f1
    });
    for (let i = 0; i < scores.length; i++) {
      const pred = scores[i] >= model.thresholds[k] ? 1 : 0;
      if (pred === 1 && labels[i] === 1) microTp++;
      else if (pred === 1 && labels[i] === 0) microFp++;
      else if (pred === 0 && labels[i] === 1) microFn++;
    }
  }
  const macroF1 = perLabel.reduce((s, r) => s + r.f1, 0) / (perLabel.length || 1);
  const microPrec = microTp + microFp === 0 ? 0 : microTp / (microTp + microFp);
  const microRec = microTp + microFn === 0 ? 0 : microTp / (microTp + microFn);
  const microF1 = microPrec + microRec === 0 ? 0 : (2 * microPrec * microRec) / (microPrec + microRec);
  const macroBaselineF1 = perLabel.reduce((s, r) => s + r.baselineF1, 0) / (perLabel.length || 1);
  return {
    perLabel,
    macroF1,
    microF1,
    macroBaselineF1,
    macroSelectivity: controlEval === null ? null : macroF1 - controlEval.macroF1
  };
}
