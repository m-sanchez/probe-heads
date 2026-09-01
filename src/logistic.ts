/** One binary logistic-regression head, trained by full-batch gradient
 * descent. Deterministic by construction: weights start at zero, there is
 * no shuffling and no randomness, so the same data and hyperparameters
 * produce bit-identical weights every run - which is what makes a probe's
 * numbers reproducible and its regressions real.
 *
 * Training also returns a certificate: the loss, the gradient norm of the
 * objective at the weights handed back, and how many steps it took to get
 * there. A probe that scores badly because the head never finished
 * training looks exactly like a probe that scores badly because the
 * property is not linearly readable, and only one of those is a result.
 * Measured with the default 300 epochs: gradient norm 4.2e-14 at the
 * demo's d=6, 1.1e-3 at d=64, 5.8e-3 at d=768. */

export interface Head {
  weights: number[];
  bias: number;
}

/** Numerically stable sigmoid. */
export function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

/** The linear pre-activation. Kept separate from `score` so the training
 * loop can reuse the exact same accumulation order. */
function preActivation(head: Head, x: number[]): number {
  let z = head.bias;
  for (let j = 0; j < x.length; j++) z += head.weights[j] * x[j];
  return z;
}

export function score(head: Head, x: number[]): number {
  if (x.length !== head.weights.length) {
    throw new Error(`feature vector has ${x.length} values, the head expects ${head.weights.length}`);
  }
  return sigmoid(preActivation(head, x));
}

export interface TrainOptions {
  epochs?: number;
  learningRate?: number;
  /** L2 penalty on the weights (not the bias) */
  l2?: number;
  /** stop once the gradient norm falls to or below this. Defaults to 0,
   * which never stops early, so a probe run before this option existed
   * returns the same weights it always did. */
  tolerance?: number;
}

/** What the optimiser did, reported at the weights actually returned. */
export interface Convergence {
  /** gradient steps taken; fewer than the budget if the tolerance was met */
  epochs: number;
  /** mean log loss plus (l2/2)|w|^2, the objective being minimised */
  finalLoss: number;
  /** L2 norm of that objective's gradient. 0 is a stationary point; this
   * number is how far the returned head is from solving its own problem. */
  finalGradNorm: number;
  /** the tolerance that was asked for */
  tolerance: number;
  /** finalGradNorm <= tolerance. With the default tolerance of 0 this is
   * false and the number to read is finalGradNorm itself; set a tolerance
   * to get a yes or no. */
  converged: boolean;
}

export interface TrainResult {
  head: Head;
  convergence: Convergence;
}

/** log(1 + e^z), without overflowing for large |z|. */
function softplus(z: number): number {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

interface Gradient {
  w: number[];
  b: number;
  norm: number;
}

/** Gradient of the L2-regularised mean log loss at `head`. */
function gradientAt(head: Head, features: number[][], labels: number[], l2: number): Gradient {
  const n = features.length;
  const dim = head.weights.length;
  const sumW = new Array(dim).fill(0);
  let sumB = 0;
  for (let i = 0; i < n; i++) {
    const err = sigmoid(preActivation(head, features[i])) - labels[i];
    for (let j = 0; j < dim; j++) sumW[j] += err * features[i][j];
    sumB += err;
  }
  const b = sumB / n;
  const w = new Array<number>(dim);
  let sq = b * b;
  for (let j = 0; j < dim; j++) {
    w[j] = sumW[j] / n + l2 * head.weights[j];
    sq += w[j] * w[j];
  }
  return { w, b, norm: Math.sqrt(sq) };
}

/** The objective itself, evaluated once at the end. */
function lossAt(head: Head, features: number[][], labels: number[], l2: number): number {
  let sum = 0;
  for (let i = 0; i < features.length; i++) {
    const z = preActivation(head, features[i]);
    sum += softplus(z) - labels[i] * z;
  }
  let penalty = 0;
  for (let j = 0; j < head.weights.length; j++) penalty += head.weights[j] * head.weights[j];
  return sum / features.length + (l2 / 2) * penalty;
}

/** Reject anything whose shape makes the answer meaningless. */
function assertTrainable(features: number[][], labels: number[]): number {
  if (features.length === 0) throw new Error('cannot train a head on no examples');
  if (features.length !== labels.length) {
    throw new Error(`got ${features.length} rows and ${labels.length} labels`);
  }
  const dim = features[0].length;
  if (dim === 0) throw new Error('cannot train a head on zero-width feature vectors');
  for (let i = 0; i < features.length; i++) {
    if (features[i].length !== dim) {
      throw new Error(
        `ragged features: row 0 has width ${dim}, row ${i} has width ${features[i].length}`
      );
    }
  }
  return dim;
}

/** Fit one head to standardised features and 0/1 labels, and report where
 * the optimiser stopped. */
export function trainHead(
  features: number[][],
  labels: number[],
  opts: TrainOptions = {}
): TrainResult {
  const dim = assertTrainable(features, labels);
  const epochs = opts.epochs ?? 300;
  const lr = opts.learningRate ?? 0.5;
  const l2 = opts.l2 ?? 1e-3;
  const tolerance = opts.tolerance ?? 0;
  if (!(tolerance >= 0)) throw new Error(`tolerance must be a number >= 0, got ${tolerance}`);
  const head: Head = { weights: new Array(dim).fill(0), bias: 0 };

  let taken = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const grad = gradientAt(head, features, labels, l2);
    if (grad.norm <= tolerance) break;
    for (let j = 0; j < dim; j++) head.weights[j] -= lr * grad.w[j];
    head.bias -= lr * grad.b;
    taken++;
  }

  // recomputed after the last step, so it is the gradient at the weights
  // being handed back rather than at the ones before them
  const finalGradNorm = gradientAt(head, features, labels, l2).norm;
  return {
    head,
    convergence: {
      epochs: taken,
      finalLoss: lossAt(head, features, labels, l2),
      finalGradNorm,
      tolerance,
      converged: finalGradNorm <= tolerance
    }
  };
}
