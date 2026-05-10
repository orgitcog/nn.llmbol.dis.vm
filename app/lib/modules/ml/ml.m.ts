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

  /*
   * Note: 'int4' dtype uses Uint8Array storage with 2 values packed per byte
   * Use Quantization.dequantize() to properly decode int4 tensors
   */
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
  private _models: Map<string, MLModel>;

  constructor() {
    this._models = new Map();
  }

  /**
   * Create a tensor with the specified shape and data type.
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
        /*
         * int4 values are packed 2 per byte — 50 % memory saving vs int8.
         * Use Quantization.dequantize() to decode.
         */
        data = new Uint8Array(Math.ceil(size / 2));
        break;
      default:
        data = new Float32Array(size);
    }

    return { shape, data, dtype };
  }

  /**
   * Matrix multiplication (GEMM).  Both tensors must be 2-D.
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
   * Element-wise addition with NumPy-style broadcasting.
   */
  add(a: Tensor, b: Tensor): Tensor {
    const [ba, bb] = this._broadcast(a, b);
    const result = this.createTensor(ba.shape, 'float32');
    const rd = result.data as Float32Array;
    const ad = ba.data as Float32Array;
    const bd = bb.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = ad[i] + bd[i];
    }

    return result;
  }

  /**
   * Element-wise subtraction with broadcasting.
   */
  subtract(a: Tensor, b: Tensor): Tensor {
    const [ba, bb] = this._broadcast(a, b);
    const result = this.createTensor(ba.shape, 'float32');
    const rd = result.data as Float32Array;
    const ad = ba.data as Float32Array;
    const bd = bb.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = ad[i] - bd[i];
    }

    return result;
  }

  /**
   * Element-wise multiplication with broadcasting.
   */
  multiply(a: Tensor, b: Tensor): Tensor {
    const [ba, bb] = this._broadcast(a, b);
    const result = this.createTensor(ba.shape, 'float32');
    const rd = result.data as Float32Array;
    const ad = ba.data as Float32Array;
    const bd = bb.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = ad[i] * bd[i];
    }

    return result;
  }

  /**
   * Element-wise division with broadcasting.
   */
  divide(a: Tensor, b: Tensor): Tensor {
    const [ba, bb] = this._broadcast(a, b);
    const result = this.createTensor(ba.shape, 'float32');
    const rd = result.data as Float32Array;
    const ad = ba.data as Float32Array;
    const bd = bb.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = ad[i] / bd[i];
    }

    return result;
  }

  /**
   * Clamp each element of `t` to the closed interval [min, max].
   */
  clamp(t: Tensor, min: number, max: number): Tensor {
    const result = this.createTensor(t.shape, 'float32');
    const rd = result.data as Float32Array;
    const td = t.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = Math.max(min, Math.min(max, td[i]));
    }

    return result;
  }

  /**
   * Natural logarithm applied element-wise.
   */
  log(t: Tensor): Tensor {
    const result = this.createTensor(t.shape, 'float32');
    const rd = result.data as Float32Array;
    const td = t.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = Math.log(td[i]);
    }

    return result;
  }

  /**
   * Natural exponential applied element-wise.
   */
  exp(t: Tensor): Tensor {
    const result = this.createTensor(t.shape, 'float32');
    const rd = result.data as Float32Array;
    const td = t.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = Math.exp(td[i]);
    }

    return result;
  }

  /**
   * Raise each element of `t` to the given `exponent`.
   */
  pow(t: Tensor, exponent: number): Tensor {
    const result = this.createTensor(t.shape, 'float32');
    const rd = result.data as Float32Array;
    const td = t.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = Math.pow(td[i], exponent);
    }

    return result;
  }

  /**
   * Create a float32 tensor of zeros with the given shape.
   */
  zeros(shape: number[]): Tensor {
    return this.createTensor(shape, 'float32');
  }

  /**
   * Create a float32 tensor of ones with the given shape.
   */
  ones(shape: number[]): Tensor {
    const t = this.createTensor(shape, 'float32');
    (t.data as Float32Array).fill(1);

    return t;
  }

  /**
   * Create a float32 tensor filled with `value`.
   */
  fill(shape: number[], value: number): Tensor {
    const t = this.createTensor(shape, 'float32');
    (t.data as Float32Array).fill(value);

    return t;
  }

  /**
   * Create a float32 tensor of standard-normal random values (Box-Muller).
   */
  randn(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);

    for (let i = 0; i < size; i += 2) {
      const u = Math.random() + 1e-10;
      const v = Math.random();
      const r = Math.sqrt(-2 * Math.log(u));
      data[i] = r * Math.cos(2 * Math.PI * v);

      if (i + 1 < size) {
        data[i + 1] = r * Math.sin(2 * Math.PI * v);
      }
    }

    return { shape, data, dtype: 'float32' };
  }

  /**
   * Return a deep copy of a tensor.
   */
  clone(t: Tensor): Tensor {
    let dataCopy: Tensor['data'];

    if (t.data instanceof Float32Array) {
      dataCopy = new Float32Array(t.data);
    } else if (t.data instanceof Int8Array) {
      dataCopy = new Int8Array(t.data);
    } else {
      dataCopy = new Uint8Array(t.data as Uint8Array);
    }

    return { shape: [...t.shape], data: dataCopy, dtype: t.dtype };
  }

  /**
   * ReLU activation: max(0, x) applied element-wise.
   */
  relu(input: Tensor): Tensor {
    const result = this.createTensor(input.shape, 'float32');
    const rd = result.data as Float32Array;
    const id = input.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] = Math.max(0, id[i]);
    }

    return result;
  }

  /**
   * Softmax activation over the entire flat data (numerically stable).
   */
  softmax(input: Tensor): Tensor {
    const result = this.createTensor(input.shape, 'float32');
    const rd = result.data as Float32Array;
    const id = input.data as Float32Array;

    let max = id[0];

    for (let i = 1; i < id.length; i++) {
      if (id[i] > max) {
        max = id[i];
      }
    }

    let sum = 0;

    for (let i = 0; i < id.length; i++) {
      rd[i] = Math.exp(id[i] - max);
      sum += rd[i];
    }

    for (let i = 0; i < rd.length; i++) {
      rd[i] /= sum;
    }

    return result;
  }

  /**
   * Register a model under `name` and return it.
   */
  loadModel(name: string, config: ModelConfig): MLModel {
    const model: MLModel = { name, parameters: [], config };
    this._models.set(name, model);

    return model;
  }

  /**
   * Return a previously loaded model by name, or `undefined`.
   */
  getModel(name: string): MLModel | undefined {
    return this._models.get(name);
  }

  /**
   * Return the names of all currently loaded models.
   */
  getModelNames(): string[] {
    return Array.from(this._models.keys());
  }

  /**
   * Return basic runtime information about loaded models.
   */
  getStats() {
    return {
      loadedModels: this._models.size,
      modelNames: this.getModelNames(),
    };
  }

  /** Return true when `a` and `b` have identical shapes. */
  private _shapesMatch(a: number[], b: number[]): boolean {
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }

  /**
   * Broadcast `a` and `b` to a common shape (NumPy rules).
   * Shapes are right-aligned; size-1 dimensions are expanded.
   */
  private _broadcast(a: Tensor, b: Tensor): [Tensor, Tensor] {
    if (this._shapesMatch(a.shape, b.shape)) {
      return [a, b];
    }

    const ndim = Math.max(a.shape.length, b.shape.length);
    const shapeA = [...Array(ndim - a.shape.length).fill(1), ...a.shape] as number[];
    const shapeB = [...Array(ndim - b.shape.length).fill(1), ...b.shape] as number[];
    const outShape: number[] = [];

    for (let i = 0; i < ndim; i++) {
      if (shapeA[i] !== shapeB[i] && shapeA[i] !== 1 && shapeB[i] !== 1) {
        throw new Error(`Cannot broadcast shapes [${a.shape}] and [${b.shape}]`);
      }

      outShape.push(Math.max(shapeA[i], shapeB[i]));
    }

    const expandedA = this._expandBroadcast(a.data as Float32Array, shapeA, outShape);
    const expandedB = this._expandBroadcast(b.data as Float32Array, shapeB, outShape);

    return [
      { shape: outShape, data: expandedA, dtype: 'float32' },
      { shape: outShape, data: expandedB, dtype: 'float32' },
    ];
  }

  /** Materialize a broadcast by repeating size-1 dimensions. */
  private _expandBroadcast(data: Float32Array, srcShape: number[], outShape: number[]): Float32Array {
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(outSize);
    const ndim = outShape.length;

    // Strides for the (left-padded) source shape
    const srcStrides = new Array<number>(ndim);
    let s = 1;

    for (let d = ndim - 1; d >= 0; d--) {
      srcStrides[d] = s;
      s *= srcShape[d];
    }

    for (let flatOut = 0; flatOut < outSize; flatOut++) {
      let tmp = flatOut;
      let srcFlat = 0;

      for (let d = ndim - 1; d >= 0; d--) {
        const outIdx = tmp % outShape[d];
        tmp = Math.floor(tmp / outShape[d]);
        srcFlat += (srcShape[d] === 1 ? 0 : outIdx) * srcStrides[d];
      }

      result[flatOut] = data[srcFlat];
    }

    return result;
  }
}

/**
 * Create a new ML module instance.
 */
export function createMLModule(): MLModule {
  return new MLModule();
}

/** Singleton ML module instance. */
let _globalMLModule: MLModule | null = null;

/**
 * Return (lazily creating) a process-wide singleton MLModule.
 */
export function getGlobalMLModule(): MLModule {
  if (!_globalMLModule) {
    _globalMLModule = createMLModule();
  }

  return _globalMLModule;
}
