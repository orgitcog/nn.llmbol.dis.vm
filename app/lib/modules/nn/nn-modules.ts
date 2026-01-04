/**
 * Neural Network Modules (Torch7-style)
 * 
 * Common neural network layers and operations
 */

import type { Tensor } from '../ml/ml.m';
import type { NNModule, LinearConfig, ConvConfig } from './nn.b';

/**
 * Linear (fully connected) layer
 */
export class Linear implements NNModule {
  type = 'Linear';
  private weight: Tensor;
  private bias?: Tensor;
  private config: LinearConfig;

  constructor(config: LinearConfig) {
    this.config = config;
    
    // Initialize weight
    this.weight = this.initializeWeight(config.inputSize, config.outputSize);
    
    // Initialize bias if needed
    if (config.bias) {
      this.bias = this.initializeBias(config.outputSize);
    }
  }

  forward(input: Tensor): Tensor {
    // y = xW^T + b
    const { data: inputData, shape: inputShape } = input;
    const outputSize = this.config.outputSize;
    const inputSize = this.config.inputSize;
    
    const batchSize = inputShape.length > 1 ? inputShape[0] : 1;
    const output = new Float32Array(batchSize * outputSize);
    
    const weightData = this.weight.data as Float32Array;
    const inputArray = inputData as Float32Array;
    
    for (let b = 0; b < batchSize; b++) {
      for (let o = 0; o < outputSize; o++) {
        let sum = 0;
        for (let i = 0; i < inputSize; i++) {
          sum += inputArray[b * inputSize + i] * weightData[o * inputSize + i];
        }
        if (this.bias) {
          sum += (this.bias.data as Float32Array)[o];
        }
        output[b * outputSize + o] = sum;
      }
    }

    return {
      shape: [batchSize, outputSize],
      data: output,
      dtype: 'float32',
    };
  }

  parameters(): Tensor[] {
    return this.bias ? [this.weight, this.bias] : [this.weight];
  }

  private initializeWeight(inputSize: number, outputSize: number): Tensor {
    const size = inputSize * outputSize;
    const data = new Float32Array(size);
    
    // Xavier/Glorot initialization: helps maintain gradient variance across layers
    // Formula: std = sqrt(2.0 / (fan_in + fan_out))
    const std = Math.sqrt(2.0 / (inputSize + outputSize));
    for (let i = 0; i < size; i++) {
      data[i] = (Math.random() - 0.5) * 2 * std;
    }

    return {
      shape: [outputSize, inputSize],
      data,
      dtype: 'float32',
    };
  }

  private initializeBias(size: number): Tensor {
    return {
      shape: [size],
      data: new Float32Array(size),
      dtype: 'float32',
    };
  }
}

/**
 * ReLU activation
 */
export class ReLU implements NNModule {
  type = 'ReLU';

  forward(input: Tensor): Tensor {
    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);
    
    for (let i = 0; i < data.length; i++) {
      output[i] = Math.max(0, data[i]);
    }

    return {
      shape: [...input.shape],
      data: output,
      dtype: 'float32',
    };
  }
}

/**
 * Tanh activation
 */
export class Tanh implements NNModule {
  type = 'Tanh';

  forward(input: Tensor): Tensor {
    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);
    
    for (let i = 0; i < data.length; i++) {
      output[i] = Math.tanh(data[i]);
    }

    return {
      shape: [...input.shape],
      data: output,
      dtype: 'float32',
    };
  }
}

/**
 * Sigmoid activation
 */
export class Sigmoid implements NNModule {
  type = 'Sigmoid';

  forward(input: Tensor): Tensor {
    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);
    
    for (let i = 0; i < data.length; i++) {
      output[i] = 1 / (1 + Math.exp(-data[i]));
    }

    return {
      shape: [...input.shape],
      data: output,
      dtype: 'float32',
    };
  }
}

/**
 * Dropout layer
 */
export class Dropout implements NNModule {
  type = 'Dropout';
  private p: number;
  private training: boolean;

  constructor(p: number = 0.5) {
    this.p = p;
    this.training = true;
  }

  forward(input: Tensor): Tensor {
    if (!this.training || this.p === 0) {
      return input;
    }

    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);
    const scale = 1 / (1 - this.p);
    
    for (let i = 0; i < data.length; i++) {
      if (Math.random() > this.p) {
        output[i] = data[i] * scale;
      } else {
        output[i] = 0;
      }
    }

    return {
      shape: [...input.shape],
      data: output,
      dtype: 'float32',
    };
  }

  setTraining(training: boolean): void {
    this.training = training;
  }
}

/**
 * Batch Normalization
 */
export class BatchNorm implements NNModule {
  type = 'BatchNorm';
  private numFeatures: number;
  private gamma: Tensor;
  private beta: Tensor;
  private eps: number;

  constructor(numFeatures: number, eps: number = 1e-5) {
    this.numFeatures = numFeatures;
    this.eps = eps;
    
    // Initialize gamma (scale) to 1
    this.gamma = {
      shape: [numFeatures],
      data: new Float32Array(numFeatures).fill(1),
      dtype: 'float32',
    };
    
    // Initialize beta (shift) to 0
    this.beta = {
      shape: [numFeatures],
      data: new Float32Array(numFeatures),
      dtype: 'float32',
    };
  }

  forward(input: Tensor): Tensor {
    const data = input.data as Float32Array;
    const output = new Float32Array(data.length);
    const gammaData = this.gamma.data as Float32Array;
    const betaData = this.beta.data as Float32Array;
    
    // Compute mean and variance
    let mean = 0;
    for (let i = 0; i < data.length; i++) {
      mean += data[i];
    }
    mean /= data.length;
    
    let variance = 0;
    for (let i = 0; i < data.length; i++) {
      const diff = data[i] - mean;
      variance += diff * diff;
    }
    variance /= data.length;
    
    const std = Math.sqrt(variance + this.eps);
    
    // Normalize and scale
    for (let i = 0; i < data.length; i++) {
      const featureIdx = i % this.numFeatures;
      const normalized = (data[i] - mean) / std;
      output[i] = gammaData[featureIdx] * normalized + betaData[featureIdx];
    }

    return {
      shape: [...input.shape],
      data: output,
      dtype: 'float32',
    };
  }

  parameters(): Tensor[] {
    return [this.gamma, this.beta];
  }
}

/**
 * Simple convolutional layer (1D)
 */
export class Conv1d implements NNModule {
  type = 'Conv1d';
  private weight: Tensor;
  private bias?: Tensor;
  private config: ConvConfig;

  constructor(config: ConvConfig) {
    this.config = config;
    
    const kernelSize = config.kernelSize;
    const inChannels = config.inChannels;
    const outChannels = config.outChannels;
    
    // Initialize weight
    const weightSize = outChannels * inChannels * kernelSize;
    this.weight = {
      shape: [outChannels, inChannels, kernelSize],
      data: new Float32Array(weightSize),
      dtype: 'float32',
    };
    
    // Xavier initialization
    const std = Math.sqrt(2.0 / (inChannels * kernelSize + outChannels));
    const weightData = this.weight.data as Float32Array;
    for (let i = 0; i < weightSize; i++) {
      weightData[i] = (Math.random() - 0.5) * 2 * std;
    }
    
    // Initialize bias
    this.bias = {
      shape: [outChannels],
      data: new Float32Array(outChannels),
      dtype: 'float32',
    };
  }

  forward(input: Tensor): Tensor {
    // Simplified 1D convolution
    const inputData = input.data as Float32Array;
    const weightData = this.weight.data as Float32Array;
    
    const inChannels = this.config.inChannels;
    const outChannels = this.config.outChannels;
    const kernelSize = this.config.kernelSize;
    const stride = this.config.stride;
    
    // Simplified output size calculation
    const inputLength = input.shape[input.shape.length - 1];
    const outputLength = Math.floor((inputLength - kernelSize) / stride) + 1;
    
    const outputSize = outChannels * outputLength;
    const output = new Float32Array(outputSize);
    
    // Perform convolution (simplified)
    for (let oc = 0; oc < outChannels; oc++) {
      for (let pos = 0; pos < outputLength; pos++) {
        let sum = 0;
        const inputPos = pos * stride;
        
        for (let ic = 0; ic < inChannels; ic++) {
          for (let k = 0; k < kernelSize; k++) {
            const inputIdx = ic * inputLength + inputPos + k;
            const weightIdx = (oc * inChannels + ic) * kernelSize + k;
            if (inputIdx < inputData.length) {
              sum += inputData[inputIdx] * weightData[weightIdx];
            }
          }
        }
        
        if (this.bias) {
          sum += (this.bias.data as Float32Array)[oc];
        }
        
        output[oc * outputLength + pos] = sum;
      }
    }

    return {
      shape: [outChannels, outputLength],
      data: output,
      dtype: 'float32',
    };
  }

  parameters(): Tensor[] {
    return this.bias ? [this.weight, this.bias] : [this.weight];
  }
}
