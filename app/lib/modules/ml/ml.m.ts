/**
 * GGML-like ML Module Implementation
 * 
 * Provides core machine learning operations inspired by GGML (GPT-Generated Model Library)
 * with support for tensor operations, quantization, and inference.
 */

export interface Tensor {
  shape: number[];
  data: Float32Array | Uint8Array | Int8Array;
  dtype: 'float32' | 'uint8' | 'int8' | 'int4';
  // Note: 'int4' dtype uses Uint8Array storage with 2 values packed per byte
  // Use Quantization.dequantize() to properly decode int4 tensors
}

export interface MLModel {
  name: string;
  parameters: Tensor[];
  config: ModelConfig;
}

export interface ModelConfig {
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  maxSequenceLength: number;
}

export class MLModule {
  private models: Map<string, MLModel>;

  constructor() {
    this.models = new Map();
  }

  /**
   * Create a tensor with the specified shape and data type
   */
  createTensor(shape: number[], dtype: Tensor['dtype'] = 'float32'): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    
    let data: Tensor['data'];
    switch (dtype) {
      case 'float32':
        data = new Float32Array(size);
        break;
      case 'uint8':
        data = new Uint8Array(size);
        break;
      case 'int8':
        data = new Int8Array(size);
        break;
      case 'int4':
        // Note: int4 values are packed 2 per byte in a Uint8Array
        // This allows for 50% memory savings compared to storing as int8
        data = new Uint8Array(Math.ceil(size / 2));
        break;
      default:
        data = new Float32Array(size);
    }

    return { shape, data, dtype };
  }

  /**
   * Matrix multiplication (GEMM - General Matrix Multiply)
   */
  matmul(a: Tensor, b: Tensor): Tensor {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error('matmul requires 2D tensors');
    }

    if (a.shape[1] !== b.shape[0]) {
      throw new Error('Incompatible shapes for matmul');
    }

    const m = a.shape[0];
    const k = a.shape[1];
    const n = b.shape[1];

    const result = this.createTensor([m, n], 'float32');
    const resultData = result.data as Float32Array;
    const aData = a.data as Float32Array;
    const bData = b.data as Float32Array;

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let l = 0; l < k; l++) {
          sum += aData[i * k + l] * bData[l * n + j];
        }
        resultData[i * n + j] = sum;
      }
    }

    return result;
  }

  /**
   * Element-wise addition
   */
  add(a: Tensor, b: Tensor): Tensor {
    if (!this.shapesMatch(a.shape, b.shape)) {
      throw new Error('Tensor shapes must match for addition');
    }

    const result = this.createTensor(a.shape, 'float32');
    const resultData = result.data as Float32Array;
    const aData = a.data as Float32Array;
    const bData = b.data as Float32Array;

    for (let i = 0; i < resultData.length; i++) {
      resultData[i] = aData[i] + bData[i];
    }

    return result;
  }

  /**
   * ReLU activation
   */
  relu(input: Tensor): Tensor {
    const result = this.createTensor(input.shape, 'float32');
    const resultData = result.data as Float32Array;
    const inputData = input.data as Float32Array;

    for (let i = 0; i < resultData.length; i++) {
      resultData[i] = Math.max(0, inputData[i]);
    }

    return result;
  }

  /**
   * Softmax activation
   */
  softmax(input: Tensor): Tensor {
    const result = this.createTensor(input.shape, 'float32');
    const resultData = result.data as Float32Array;
    const inputData = input.data as Float32Array;

    // Find max for numerical stability
    let max = inputData[0];
    for (let i = 1; i < inputData.length; i++) {
      max = Math.max(max, inputData[i]);
    }

    // Compute exp and sum
    let sum = 0;
    for (let i = 0; i < inputData.length; i++) {
      resultData[i] = Math.exp(inputData[i] - max);
      sum += resultData[i];
    }

    // Normalize
    for (let i = 0; i < resultData.length; i++) {
      resultData[i] /= sum;
    }

    return result;
  }

  /**
   * Load a model
   */
  loadModel(name: string, config: ModelConfig): MLModel {
    const model: MLModel = {
      name,
      parameters: [],
      config,
    };

    this.models.set(name, model);
    return model;
  }

  /**
   * Get a loaded model
   */
  getModel(name: string): MLModel | undefined {
    return this.models.get(name);
  }

  /**
   * Helper: Check if shapes match
   */
  private shapesMatch(a: number[], b: number[]): boolean {
    if (a.length !== b.length) return false;
    return a.every((val, idx) => val === b[idx]);
  }
}

/**
 * Create a new ML module instance
 */
export function createMLModule(): MLModule {
  return new MLModule();
}

/**
 * Global ML module instance
 */
let globalMLModule: MLModule | null = null;

export function getGlobalMLModule(): MLModule {
  if (!globalMLModule) {
    globalMLModule = createMLModule();
  }
  return globalMLModule;
}
