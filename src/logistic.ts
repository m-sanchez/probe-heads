/** One binary logistic-regression head, trained by full-batch gradient
 * descent. Deterministic by construction: weights start at zero, there is
 * no shuffling and no randomness, so the same data and hyperparameters
 * produce bit-identical weights every run - which is what makes a probe's
 * numbers reproducible and its regressions real. */

export interface Head {
  weights: number[];
  bias: number;
}

/** Numerically stable sigmoid. */
export function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

export function score(head: Head, x: number[]): number {
  let z = head.bias;
  for (let j = 0; j < x.length; j++) z += head.weights[j] * x[j];
  return sigmoid(z);
}

export interface TrainOptions {
  epochs?: number;
  learningRate?: number;
  /** L2 penalty on the weights (not the bias) */
  l2?: number;
}

/** Fit one head to standardised features and 0/1 labels. */
export function trainHead(
  features: number[][],
  labels: number[],
  opts: TrainOptions = {}
): Head {
  const epochs = opts.epochs ?? 300;
  const lr = opts.learningRate ?? 0.5;
  const l2 = opts.l2 ?? 1e-3;
  const n = features.length;
  const dim = n > 0 ? features[0].length : 0;
  const head: Head = { weights: new Array(dim).fill(0), bias: 0 };
  if (n === 0) return head;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const gradW = new Array(dim).fill(0);
    let gradB = 0;
    for (let i = 0; i < n; i++) {
      const p = score(head, features[i]);
      const err = p - labels[i];
      for (let j = 0; j < dim; j++) gradW[j] += err * features[i][j];
      gradB += err;
    }
    for (let j = 0; j < dim; j++) {
      head.weights[j] -= lr * (gradW[j] / n + l2 * head.weights[j]);
    }
    head.bias -= lr * (gradB / n);
  }
  return head;
}
