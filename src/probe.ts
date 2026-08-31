/** The probe: deterministic one-vs-rest logistic heads on frozen feature
 * vectors, with train/val/holdout hygiene wired into the API.
 *
 * The shape enforces the discipline. `fit` sees train and val only - it
 * standardises on train, trains a head per label on train, and selects
 * each label's threshold on val. The holdout is not an argument to `fit`;
 * it goes to `evaluate`, once, at the end. A frozen embedding in, an
 * honest per-label report out, the same every run. */

import { fitScaler, transform } from './scaler.ts';
import type { Scaler } from './scaler.ts';
import { score, trainHead } from './logistic.ts';
import type { Head, TrainOptions } from './logistic.ts';
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
}

export interface FitOptions extends TrainOptions {
  thresholdGrid?: number[];
}

function assertConsistent(examples: Example[], labelCount: number): void {
  for (const e of examples) {
    if (e.labels.length !== labelCount) {
      throw new Error(`every example needs ${labelCount} labels, got ${e.labels.length}`);
    }
  }
}

/** Fit heads on train, choose thresholds on val. The holdout is not
 * accepted here, by design. */
export function fit(train: Example[], val: Example[], opts: FitOptions = {}): ProbeModel {
  if (train.length === 0) throw new Error('training split is empty');
  const labelCount = train[0].labels.length;
  assertConsistent(train, labelCount);
  assertConsistent(val, labelCount);

  const scaler = fitScaler(train.map((e) => e.features));
  const xTrain = train.map((e) => transform(scaler, e.features));
  const xVal = val.map((e) => transform(scaler, e.features));
  const grid = opts.thresholdGrid ?? defaultGrid();

  const heads: Head[] = [];
  const thresholds: number[] = [];
  for (let k = 0; k < labelCount; k++) {
    const yTrain = train.map((e) => (e.labels[k] ? 1 : 0));
    const head = trainHead(xTrain, yTrain, opts);
    heads.push(head);
    if (val.length > 0) {
      const valScores = xVal.map((x) => score(head, x));
      const yVal = val.map((e) => (e.labels[k] ? 1 : 0));
      thresholds.push(selectThreshold(valScores, yVal, grid).threshold);
    } else {
      thresholds.push(0.5);
    }
  }
  return { scaler, heads, thresholds, labelCount };
}

export interface Prediction {
  /** probability per label */
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
}

export interface Evaluation {
  perLabel: LabelReport[];
  /** unweighted mean F1 across labels */
  macroF1: number;
  /** F1 pooled over all label decisions */
  microF1: number;
}

/** Evaluate on the holdout, once, at the val-selected thresholds. */
export function evaluate(model: ProbeModel, holdout: Example[]): Evaluation {
  assertConsistent(holdout, model.labelCount);
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
    perLabel.push({ precision: point.precision, recall: point.recall, f1: point.f1, support });
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
  return { perLabel, macroF1, microF1 };
}
