/**
 * Neural Network Modules (Torch7-style)
 *
 * Common neural network layers and operations
 */

import type { Tensor } from '~/lib/modules/ml/ml.m';
import type { NNModule, LinearConfig, ConvConfig } from '~/lib/modules/nn/nn.b';

/**
 * Linear (fully connected) layer
 */
export class Linear implements NNModule {
  type = 'Linear';
  private _weight: Tensor;
  private _bias?: Tensor;
  private _config: LinearConfig;
  private _lastInput: Tensor | null = null;
  private _weightGrad: Tensor | null = null;
  private _biasGrad: Tensor | null = null;

  constructor(config: LinearConfig) {
    this._config = config;
    this._weight = this._initWeight(config.inputSize, config.outputSize);

    if (config.bias) {
      this._bias = this._initBias(config.outputSize);
    }
  }

  /**
   * Forward pass: y = xW^T + b
   */
  forward(input: Tensor): Tensor {
    this._lastInput = input;

    const { data: inputData, shape: inputShape } = input;
    const outputSize = this._config.outputSize;
    const inputSize = this._config.inputSize;
    const batchSize = inputShape.length > 1 ? inputShape[0] : 1;
    const output = new Float32Array(batchSize * outputSize);
    const weightData = this._weight.data as Float32Array;
    const inputArray = inputData as Float32Array;

    for (let b = 0; b < batchSize; b++) {
      for (let o = 0; o < outputSize; o++) {
        let sum = 0;

        for (let i = 0; i < inputSize; i++) {
          sum += inputArray[b * inputSize + i] * weightData[o * inputSize + i];
        }

        if (this._bias) {
          sum += (this._bias.data as Float32Array)[o];
        }

        output[b * outputSize + o] = sum;
      }
    }

    return { shape: [batchSize, outputSize], data: output, dtype: 'float32' };
  }

  /**
   * Backward pass: compute gradients for weight and bias, return input gradient
   */
  backward(gradOutput: Tensor): Tensor {
    const inputSize = this._config.inputSize;
    const outputSize = this._config.outputSize;
    const gradData = gradOutput.data as Float32Array;
    const weightData = this._weight.data as Float32Array;

    const lastInput = this._lastInput;
    const batchSize = lastInput ? (lastInput.shape.length > 1 ? lastInput.shape[0] : 1) : 1;
    const inputData = lastInput ? (lastInput.data as Float32Array) : new Float32Array(batchSize * inputSize);

    // Weight gradient: dL/dW = input^T × gradOutput (averaged over batch)
    const weightGrad = new Float32Array(outputSize * inputSize);

    for (let o = 0; o < outputSize; o++) {
      for (let i = 0; i < inputSize; i++) {
        let g = 0;

        for (let b = 0; b < batchSize; b++) {
          g += inputData[b * inputSize + i] * gradData[b * outputSize + o];
        }

        weightGrad[o * inputSize + i] = g / batchSize;
      }
    }

    this._weightGrad = { shape: [outputSize, inputSize], data: weightGrad, dtype: 'float32' };

    // Bias gradient: dL/db = sum over batch of gradOutput
    if (this._bias) {
      const biasGrad = new Float32Array(outputSize);

      for (let o = 0; o < outputSize; o++) {
        for (let b = 0; b < batchSize; b++) {
          biasGrad[o] += gradData[b * outputSize + o];
        }

        biasGrad[o] /= batchSize;
      }

      this._biasGrad = { shape: [outputSize], data: biasGrad, dtype: 'float32' };
    }

    // Input gradient: dL/dx = gradOutput × W
    const inputGrad = new Float32Array(batchSize * inputSize);

    for (let b = 0; b < batchSize; b++) {
      for (let i = 0; i < inputSize; i++) {
        let g = 0;

        for (let o = 0; o < outputSize; o++) {
          g += gradData[b * outputSize + o] * weightData[o * inputSize + i];
        }

        inputGrad[b * inputSize + i] = g;
      }
    }

    return { shape: [batchSize, inputSize], data: inputGrad, dtype: 'float32' };
  }

  /**
   * Returns trainable parameters: [weight, bias?]
   */
  parameters(): Tensor[] {
    return this._bias ? [this._weight, this._bias] : [this._weight];
  }

  /**
   * Returns stored gradients in same order as parameters()
   */
  gradients(): Tensor[] {
    const grads: Tensor[] = [];

    if (this._weightGrad) {
      grads.push(this._weightGrad);
    }

    if (this._bias && this._biasGrad) {
      grads.push(this._biasGrad);
    }

    return grads;
  }

  private _initWeight(inputSize: number, outputSize: number): Tensor {
    const size = inputSize * outputSize;
    const data = new Float32Array(size);
    const std = Math.sqrt(2.0 / (inputSize + outputSize));

    for (let i = 0; i < size; i++) {
      data[i] = (Math.random() - 0.5) * 2 * std;
    }

    return { shape: [outputSize, inputSize], data, dtype: 'float32' };
  }

  private _initBias(size: number): Tensor {
    return { shape: [size], data: new Float32Array(size), dtype: 'float32' };
  }
}

/**
 * ReLU activation
 */
export class ReLU implements NNModule {
  type = 'ReLU';

  /**
   * Forward pass: max(0, x)
   */
  forward(input: Tensor): Tensor {
    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      output[i] = Math.max(0, data[i]);
    }

    return { shape: [...input.shape], data: output, dtype: 'float32' };
  }
}

/**
 * Tanh activation
 */
export class Tanh implements NNModule {
  type = 'Tanh';

  /**
   * Forward pass: tanh(x)
   */
  forward(input: Tensor): Tensor {
    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      output[i] = Math.tanh(data[i]);
    }

    return { shape: [...input.shape], data: output, dtype: 'float32' };
  }
}

/**
 * Sigmoid activation
 */
export class Sigmoid implements NNModule {
  type = 'Sigmoid';

  /**
   * Forward pass: 1 / (1 + exp(-x))
   */
  forward(input: Tensor): Tensor {
    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      output[i] = 1 / (1 + Math.exp(-data[i]));
    }

    return { shape: [...input.shape], data: output, dtype: 'float32' };
  }
}

/**
 * Dropout regularisation layer
 */
export class Dropout implements NNModule {
  type = 'Dropout';
  private _p: number;
  private _training: boolean;
  private _mask: Float32Array | null = null;

  constructor(p: number = 0.5) {
    this._p = p;
    this._training = true;
  }

  /**
   * Forward pass — randomly zeroes elements with probability p during training
   */
  forward(input: Tensor): Tensor {
    if (!this._training || this._p === 0) {
      this._mask = null;
      return input;
    }

    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);
    const scale = 1 / (1 - this._p);
    this._mask = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      if (Math.random() > this._p) {
        this._mask[i] = scale;
        output[i] = data[i] * scale;
      } else {
        this._mask[i] = 0;
        output[i] = 0;
      }
    }

    return { shape: [...input.shape], data: output, dtype: 'float32' };
  }

  /**
   * Backward pass — apply stored dropout mask to gradient
   */
  backward(gradOutput: Tensor): Tensor {
    if (!this._mask) {
      return gradOutput;
    }

    const gradData = gradOutput.data as Float32Array;
    const output = new Float32Array(gradData.length);

    for (let i = 0; i < gradData.length; i++) {
      output[i] = gradData[i] * this._mask[i];
    }

    return { shape: [...gradOutput.shape], data: output, dtype: 'float32' };
  }

  /**
   * Switch between training and inference mode
   */
  setTraining(training: boolean): void {
    this._training = training;
  }
}

/**
 * Batch Normalisation
 */
export class BatchNorm implements NNModule {
  type = 'BatchNorm';
  private _numFeatures: number;
  private _gamma: Tensor;
  private _beta: Tensor;
  private _eps: number;
  private _lastInput: Tensor | null = null;
  private _lastMean: number = 0;
  private _lastVariance: number = 0;
  private _gammaGrad: Tensor | null = null;
  private _betaGrad: Tensor | null = null;

  constructor(numFeatures: number, eps: number = 1e-5) {
    this._numFeatures = numFeatures;
    this._eps = eps;

    this._gamma = {
      shape: [numFeatures],
      data: new Float32Array(numFeatures).fill(1),
      dtype: 'float32',
    };

    this._beta = {
      shape: [numFeatures],
      data: new Float32Array(numFeatures),
      dtype: 'float32',
    };
  }

  /**
   * Forward pass — normalises input then applies affine transform γ·x̂ + β
   */
  forward(input: Tensor): Tensor {
    this._lastInput = input;

    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);
    const gammaData = this._gamma.data as Float32Array;
    const betaData = this._beta.data as Float32Array;

    let mean = 0;

    for (let i = 0; i < data.length; i++) {
      mean += data[i];
    }

    mean /= data.length;
    this._lastMean = mean;

    let variance = 0;

    for (let i = 0; i < data.length; i++) {
      const diff = data[i] - mean;
      variance += diff * diff;
    }

    variance /= data.length;
    this._lastVariance = variance;

    const std = Math.sqrt(variance + this._eps);

    for (let i = 0; i < data.length; i++) {
      const featureIdx = i % this._numFeatures;
      const normalized = (data[i] - mean) / std;
      output[i] = gammaData[featureIdx] * normalized + betaData[featureIdx];
    }

    return { shape: [...input.shape], data: output, dtype: 'float32' };
  }

  /**
   * Backward pass — compute gradients for γ, β and the input
   */
  backward(gradOutput: Tensor): Tensor {
    const gradData = gradOutput.data as Float32Array;
    const n = gradData.length;
    const std = Math.sqrt(this._lastVariance + this._eps);
    const gammaData = this._gamma.data as Float32Array;

    const inputData = this._lastInput ? (this._lastInput.data as Float32Array) : new Float32Array(n);

    // Normalised inputs
    const xHat = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      xHat[i] = (inputData[i] - this._lastMean) / std;
    }

    // Gradient for gamma: dL/dγ = sum(dL/dy * x̂) per feature
    const gammaGrad = new Float32Array(this._numFeatures);
    const betaGrad = new Float32Array(this._numFeatures);

    for (let i = 0; i < n; i++) {
      const fi = i % this._numFeatures;
      gammaGrad[fi] += gradData[i] * xHat[i];
      betaGrad[fi] += gradData[i];
    }

    this._gammaGrad = { shape: [this._numFeatures], data: gammaGrad, dtype: 'float32' };
    this._betaGrad = { shape: [this._numFeatures], data: betaGrad, dtype: 'float32' };

    // Input gradient (standard batch norm backward formula)
    const inputGrad = new Float32Array(n);
    let sumDy = 0;
    let sumDyXhat = 0;

    for (let i = 0; i < n; i++) {
      const fi = i % this._numFeatures;
      sumDy += gradData[i] * gammaData[fi];
      sumDyXhat += gradData[i] * gammaData[fi] * xHat[i];
    }

    for (let i = 0; i < n; i++) {
      const fi = i % this._numFeatures;
      inputGrad[i] = (gammaData[fi] / std) * (gradData[i] - sumDy / n - (xHat[i] * sumDyXhat) / n);
    }

    return { shape: [...gradOutput.shape], data: inputGrad, dtype: 'float32' };
  }

  /**
   * Returns trainable parameters: [gamma, beta]
   */
  parameters(): Tensor[] {
    return [this._gamma, this._beta];
  }

  /**
   * Returns stored gradients in same order as parameters()
   */
  gradients(): Tensor[] {
    const grads: Tensor[] = [];

    if (this._gammaGrad) {
      grads.push(this._gammaGrad);
    }

    if (this._betaGrad) {
      grads.push(this._betaGrad);
    }

    return grads;
  }
}

/**
 * 1-D Convolutional layer
 */
export class Conv1d implements NNModule {
  type = 'Conv1d';
  private _weight: Tensor;
  private _bias?: Tensor;
  private _config: ConvConfig;
  private _lastInput: Tensor | null = null;
  private _weightGrad: Tensor | null = null;
  private _biasGrad: Tensor | null = null;

  constructor(config: ConvConfig) {
    this._config = config;

    const { kernelSize, inChannels, outChannels } = config;
    const weightSize = outChannels * inChannels * kernelSize;
    const std = Math.sqrt(2.0 / (inChannels * kernelSize + outChannels));
    const weightData = new Float32Array(weightSize);

    for (let i = 0; i < weightSize; i++) {
      weightData[i] = (Math.random() - 0.5) * 2 * std;
    }

    this._weight = { shape: [outChannels, inChannels, kernelSize], data: weightData, dtype: 'float32' };
    this._bias = { shape: [outChannels], data: new Float32Array(outChannels), dtype: 'float32' };
  }

  /**
   * Forward pass — padded 1-D convolution
   */
  forward(input: Tensor): Tensor {
    this._lastInput = input;

    const { inChannels, outChannels, kernelSize, stride, padding } = this._config;
    const inputData = input.data as Float32Array;
    const weightData = this._weight.data as Float32Array;
    const rawLength = input.shape[input.shape.length - 1];

    // Apply padding
    const paddedLength = rawLength + 2 * padding;
    let paddedData: Float32Array;

    if (padding > 0) {
      paddedData = new Float32Array(inChannels * paddedLength);

      for (let ic = 0; ic < inChannels; ic++) {
        for (let pos = 0; pos < rawLength; pos++) {
          paddedData[ic * paddedLength + padding + pos] = inputData[ic * rawLength + pos];
        }
      }
    } else {
      paddedData = inputData;
    }

    // Guard: input must fit at least one kernel
    if (paddedLength < kernelSize) {
      const outputLength = 1;
      return {
        shape: [outChannels, outputLength],
        data: new Float32Array(outChannels * outputLength),
        dtype: 'float32',
      };
    }

    const outputLength = Math.floor((paddedLength - kernelSize) / stride) + 1;
    const output = new Float32Array(outChannels * outputLength);

    for (let oc = 0; oc < outChannels; oc++) {
      for (let pos = 0; pos < outputLength; pos++) {
        let sum = 0;
        const inputPos = pos * stride;

        for (let ic = 0; ic < inChannels; ic++) {
          for (let k = 0; k < kernelSize; k++) {
            const inputIdx = ic * paddedLength + inputPos + k;
            const weightIdx = (oc * inChannels + ic) * kernelSize + k;
            sum += paddedData[inputIdx] * weightData[weightIdx];
          }
        }

        if (this._bias) {
          sum += (this._bias.data as Float32Array)[oc];
        }

        output[oc * outputLength + pos] = sum;
      }
    }

    return { shape: [outChannels, outputLength], data: output, dtype: 'float32' };
  }

  /**
   * Backward pass — compute weight (and bias) gradients; return input gradient
   */
  backward(gradOutput: Tensor): Tensor {
    const { inChannels, outChannels, kernelSize, stride, padding } = this._config;
    const gradData = gradOutput.data as Float32Array;
    const weightData = this._weight.data as Float32Array;
    const rawLength = this._lastInput ? this._lastInput.shape[this._lastInput.shape.length - 1] : 0;
    const paddedLength = rawLength + 2 * padding;
    const outputLength = gradOutput.shape[gradOutput.shape.length - 1];

    const lastInputData = this._lastInput
      ? (this._lastInput.data as Float32Array)
      : new Float32Array(inChannels * rawLength);

    // Build padded input
    const paddedInput = new Float32Array(inChannels * paddedLength);

    if (padding > 0) {
      for (let ic = 0; ic < inChannels; ic++) {
        for (let pos = 0; pos < rawLength; pos++) {
          paddedInput[ic * paddedLength + padding + pos] = lastInputData[ic * rawLength + pos];
        }
      }
    } else {
      paddedInput.set(lastInputData);
    }

    // Weight gradient
    const weightGrad = new Float32Array(outChannels * inChannels * kernelSize);

    for (let oc = 0; oc < outChannels; oc++) {
      for (let ic = 0; ic < inChannels; ic++) {
        for (let k = 0; k < kernelSize; k++) {
          let g = 0;

          for (let pos = 0; pos < outputLength; pos++) {
            const inputIdx = ic * paddedLength + pos * stride + k;
            g += paddedInput[inputIdx] * gradData[oc * outputLength + pos];
          }

          weightGrad[(oc * inChannels + ic) * kernelSize + k] = g;
        }
      }
    }

    this._weightGrad = { shape: [outChannels, inChannels, kernelSize], data: weightGrad, dtype: 'float32' };

    // Bias gradient
    if (this._bias) {
      const biasGrad = new Float32Array(outChannels);

      for (let oc = 0; oc < outChannels; oc++) {
        for (let pos = 0; pos < outputLength; pos++) {
          biasGrad[oc] += gradData[oc * outputLength + pos];
        }
      }

      this._biasGrad = { shape: [outChannels], data: biasGrad, dtype: 'float32' };
    }

    // Input gradient (full convolution of gradOutput with flipped kernel)
    const inputGradPadded = new Float32Array(inChannels * paddedLength);

    for (let oc = 0; oc < outChannels; oc++) {
      for (let ic = 0; ic < inChannels; ic++) {
        for (let pos = 0; pos < outputLength; pos++) {
          const g = gradData[oc * outputLength + pos];

          for (let k = 0; k < kernelSize; k++) {
            const inputIdx = ic * paddedLength + pos * stride + k;
            inputGradPadded[inputIdx] += g * weightData[(oc * inChannels + ic) * kernelSize + k];
          }
        }
      }
    }

    // Remove padding from input gradient
    const inputGrad = new Float32Array(inChannels * rawLength);

    for (let ic = 0; ic < inChannels; ic++) {
      for (let pos = 0; pos < rawLength; pos++) {
        inputGrad[ic * rawLength + pos] = inputGradPadded[ic * paddedLength + padding + pos];
      }
    }

    return { shape: [inChannels, rawLength], data: inputGrad, dtype: 'float32' };
  }

  /**
   * Returns trainable parameters: [weight, bias?]
   */
  parameters(): Tensor[] {
    return this._bias ? [this._weight, this._bias] : [this._weight];
  }

  /**
   * Returns stored gradients in same order as parameters()
   */
  gradients(): Tensor[] {
    const grads: Tensor[] = [];

    if (this._weightGrad) {
      grads.push(this._weightGrad);
    }

    if (this._bias && this._biasGrad) {
      grads.push(this._biasGrad);
    }

    return grads;
  }
}

/**
 * Layer Normalisation
 */
export class LayerNorm implements NNModule {
  type = 'LayerNorm';
  private _normalizedShape: number[];
  private _gamma: Tensor;
  private _beta: Tensor;
  private _eps: number;
  private _lastInput: Tensor | null = null;
  private _lastMean: number = 0;
  private _lastVariance: number = 0;
  private _gammaGrad: Tensor | null = null;
  private _betaGrad: Tensor | null = null;

  constructor(normalizedShape: number[], eps: number = 1e-5) {
    this._normalizedShape = normalizedShape;
    this._eps = eps;

    const size = normalizedShape.reduce((a, b) => a * b, 1);
    this._gamma = { shape: normalizedShape, data: new Float32Array(size).fill(1), dtype: 'float32' };
    this._beta = { shape: normalizedShape, data: new Float32Array(size), dtype: 'float32' };
  }

  /**
   * Forward pass — normalise over the last dimension(s) matching normalizedShape
   */
  forward(input: Tensor): Tensor {
    this._lastInput = input;

    const data = input.data as Float32Array;
    const normSize = this._normalizedShape.reduce((a, b) => a * b, 1);
    const numGroups = data.length / normSize;
    const output = new Float32Array(data.length);
    const gammaData = this._gamma.data as Float32Array;
    const betaData = this._beta.data as Float32Array;

    // Use mean/variance of last group for caching (simplified: store global)
    let lastMean = 0;
    let lastVariance = 0;

    for (let g = 0; g < numGroups; g++) {
      const base = g * normSize;
      let mean = 0;

      for (let i = 0; i < normSize; i++) {
        mean += data[base + i];
      }

      mean /= normSize;
      lastMean = mean;

      let variance = 0;

      for (let i = 0; i < normSize; i++) {
        const d = data[base + i] - mean;
        variance += d * d;
      }

      variance /= normSize;
      lastVariance = variance;

      const std = Math.sqrt(variance + this._eps);

      for (let i = 0; i < normSize; i++) {
        const normalized = (data[base + i] - mean) / std;
        output[base + i] = gammaData[i] * normalized + betaData[i];
      }
    }

    this._lastMean = lastMean;
    this._lastVariance = lastVariance;

    return { shape: [...input.shape], data: output, dtype: 'float32' };
  }

  /**
   * Backward pass — compute gradients for γ and β, return input gradient
   */
  backward(gradOutput: Tensor): Tensor {
    const gradData = gradOutput.data as Float32Array;
    const data = this._lastInput ? (this._lastInput.data as Float32Array) : new Float32Array(gradData.length);
    const normSize = this._normalizedShape.reduce((a, b) => a * b, 1);
    const numGroups = gradData.length / normSize;
    const gammaData = this._gamma.data as Float32Array;
    const gammaGrad = new Float32Array(normSize);
    const betaGrad = new Float32Array(normSize);
    const inputGrad = new Float32Array(gradData.length);

    for (let g = 0; g < numGroups; g++) {
      const base = g * normSize;

      let mean = 0;

      for (let i = 0; i < normSize; i++) {
        mean += data[base + i];
      }

      mean /= normSize;

      let variance = 0;

      for (let i = 0; i < normSize; i++) {
        const d = data[base + i] - mean;
        variance += d * d;
      }

      variance /= normSize;

      const std = Math.sqrt(variance + this._eps);

      const xHat = new Float32Array(normSize);

      for (let i = 0; i < normSize; i++) {
        xHat[i] = (data[base + i] - mean) / std;
      }

      // Accumulate gamma/beta gradients
      for (let i = 0; i < normSize; i++) {
        gammaGrad[i] += gradData[base + i] * xHat[i];
        betaGrad[i] += gradData[base + i];
      }

      // Input gradient (standard LayerNorm backward)
      let sumDy = 0;
      let sumDyXhat = 0;

      for (let i = 0; i < normSize; i++) {
        sumDy += gradData[base + i] * gammaData[i];
        sumDyXhat += gradData[base + i] * gammaData[i] * xHat[i];
      }

      for (let i = 0; i < normSize; i++) {
        inputGrad[base + i] =
          (gammaData[i] / std) * (gradData[base + i] - sumDy / normSize - xHat[i] * (sumDyXhat / normSize));
      }
    }

    this._gammaGrad = { shape: this._normalizedShape, data: gammaGrad, dtype: 'float32' };
    this._betaGrad = { shape: this._normalizedShape, data: betaGrad, dtype: 'float32' };

    return { shape: [...gradOutput.shape], data: inputGrad, dtype: 'float32' };
  }

  /**
   * Returns trainable parameters: [gamma, beta]
   */
  parameters(): Tensor[] {
    return [this._gamma, this._beta];
  }

  /**
   * Returns stored gradients in same order as parameters()
   */
  gradients(): Tensor[] {
    const grads: Tensor[] = [];

    if (this._gammaGrad) {
      grads.push(this._gammaGrad);
    }

    if (this._betaGrad) {
      grads.push(this._betaGrad);
    }

    return grads;
  }
}

/**
 * Multi-Head Self-Attention
 */
export class MultiHeadAttention implements NNModule {
  type = 'MultiHeadAttention';
  private _embedDim: number;
  private _numHeads: number;
  private _headDim: number;
  private _wq: Tensor;
  private _wk: Tensor;
  private _wv: Tensor;
  private _wo: Tensor;

  constructor(embedDim: number, numHeads: number) {
    if (embedDim % numHeads !== 0) {
      throw new Error(`embedDim (${embedDim}) must be divisible by numHeads (${numHeads})`);
    }

    this._embedDim = embedDim;
    this._numHeads = numHeads;
    this._headDim = embedDim / numHeads;

    const std = Math.sqrt(2.0 / (embedDim + embedDim));
    this._wq = this._initMatrix(embedDim, embedDim, std);
    this._wk = this._initMatrix(embedDim, embedDim, std);
    this._wv = this._initMatrix(embedDim, embedDim, std);
    this._wo = this._initMatrix(embedDim, embedDim, std);
  }

  /**
   * Forward pass — scaled dot-product multi-head self-attention
   */
  forward(input: Tensor): Tensor {
    const inputData = input.data as Float32Array;

    // Expect input shape [seqLen, embedDim] or [embedDim]
    const seqLen = input.shape.length > 1 ? input.shape[0] : 1;
    const d = this._embedDim;
    const h = this._numHeads;
    const dh = this._headDim;

    // Project to Q, K, V via loop-based GEMM
    const q = this._gemm(inputData, this._wq.data as Float32Array, seqLen, d, d);
    const k = this._gemm(inputData, this._wk.data as Float32Array, seqLen, d, d);
    const v = this._gemm(inputData, this._wv.data as Float32Array, seqLen, d, d);

    // Multi-head attention
    const scale = 1 / Math.sqrt(dh);
    const attnOutput = new Float32Array(seqLen * d);

    for (let head = 0; head < h; head++) {
      const offset = head * dh;

      // Compute attention scores: (seqLen x dh) × (dh x seqLen) = (seqLen x seqLen)
      const scores = new Float32Array(seqLen * seqLen);

      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          let dot = 0;

          for (let dk = 0; dk < dh; dk++) {
            dot += q[i * d + offset + dk] * k[j * d + offset + dk];
          }

          scores[i * seqLen + j] = dot * scale;
        }
      }

      // Softmax over last dimension
      for (let i = 0; i < seqLen; i++) {
        let maxVal = -Infinity;

        for (let j = 0; j < seqLen; j++) {
          if (scores[i * seqLen + j] > maxVal) {
            maxVal = scores[i * seqLen + j];
          }
        }

        let expSum = 0;

        for (let j = 0; j < seqLen; j++) {
          scores[i * seqLen + j] = Math.exp(scores[i * seqLen + j] - maxVal);
          expSum += scores[i * seqLen + j];
        }

        for (let j = 0; j < seqLen; j++) {
          scores[i * seqLen + j] /= expSum;
        }
      }

      // Weighted sum of values
      for (let i = 0; i < seqLen; i++) {
        for (let dk = 0; dk < dh; dk++) {
          let weighted = 0;

          for (let j = 0; j < seqLen; j++) {
            weighted += scores[i * seqLen + j] * v[j * d + offset + dk];
          }

          attnOutput[i * d + offset + dk] = weighted;
        }
      }
    }

    // Output projection
    const outputData = this._gemm(attnOutput, this._wo.data as Float32Array, seqLen, d, d);

    return { shape: [seqLen, d], data: outputData, dtype: 'float32' };
  }

  /**
   * Returns trainable parameters: [wq, wk, wv, wo]
   */
  parameters(): Tensor[] {
    return [this._wq, this._wk, this._wv, this._wo];
  }

  /** Simple GEMM: A(m×k) × B^T(n×k) → C(m×n) where B is stored row-major [n, k] */
  private _gemm(a: Float32Array, b: Float32Array, m: number, k: number, n: number): Float32Array {
    const c = new Float32Array(m * n);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;

        for (let p = 0; p < k; p++) {
          sum += a[i * k + p] * b[j * k + p];
        }

        c[i * n + j] = sum;
      }
    }

    return c;
  }

  private _initMatrix(rows: number, cols: number, std: number): Tensor {
    const data = new Float32Array(rows * cols);

    for (let i = 0; i < data.length; i++) {
      data[i] = (Math.random() - 0.5) * 2 * std;
    }

    return { shape: [rows, cols], data, dtype: 'float32' };
  }
}

/**
 * Token Embedding lookup table
 */
export class Embedding implements NNModule {
  type = 'Embedding';
  private _numEmbeddings: number;
  private _embeddingDim: number;
  private _weight: Tensor;

  constructor(numEmbeddings: number, embeddingDim: number) {
    this._numEmbeddings = numEmbeddings;
    this._embeddingDim = embeddingDim;

    // Initialise with small random values
    const data = new Float32Array(numEmbeddings * embeddingDim);
    const std = Math.sqrt(1.0 / embeddingDim);

    for (let i = 0; i < data.length; i++) {
      data[i] = (Math.random() - 0.5) * 2 * std;
    }

    this._weight = { shape: [numEmbeddings, embeddingDim], data, dtype: 'float32' };
  }

  /**
   * Forward pass — gather embeddings for integer token IDs in input.data
   */
  forward(input: Tensor): Tensor {
    const ids = input.data as Float32Array;
    const weightData = this._weight.data as Float32Array;
    const d = this._embeddingDim;
    const output = new Float32Array(ids.length * d);

    for (let i = 0; i < ids.length; i++) {
      const idx = Math.floor(ids[i]);
      const clampedIdx = Math.max(0, Math.min(this._numEmbeddings - 1, idx));

      for (let j = 0; j < d; j++) {
        output[i * d + j] = weightData[clampedIdx * d + j];
      }
    }

    return { shape: [ids.length, d], data: output, dtype: 'float32' };
  }

  /**
   * Returns trainable parameters: [weight]
   */
  parameters(): Tensor[] {
    return [this._weight];
  }
}
