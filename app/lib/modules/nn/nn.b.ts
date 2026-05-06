/**
 * Build-a-Bear Neural Network Architecture
 *
 * Torch7-inspired modular neural network builder that allows
 * dynamic composition of neural network layers
 */

import type { Tensor } from '~/lib/modules/ml/ml.m';

export interface NNModule {
  type: string;
  forward(input: Tensor): Tensor;
  backward?(gradOutput: Tensor): Tensor;
  parameters?(): Tensor[];

  /** Returns stored gradients in same order as parameters() */
  gradients?(): Tensor[];
}

export interface LinearConfig {
  inputSize: number;
  outputSize: number;
  bias: boolean;
}

export interface ConvConfig {
  inChannels: number;
  outChannels: number;
  kernelSize: number;
  stride: number;
  padding: number;
}

/**
 * Sequential container for building networks layer by layer
 */
export class Sequential implements NNModule {
  type = 'Sequential';
  private _modules: NNModule[];

  constructor() {
    this._modules = [];
  }

  /**
   * Add a module to the sequence
   */
  add(module: NNModule): this {
    this._modules.push(module);
    return this;
  }

  /**
   * Forward pass through all modules in order
   */
  forward(input: Tensor): Tensor {
    let output = input;

    for (const module of this._modules) {
      output = module.forward(output);
    }

    return output;
  }

  /**
   * Backward pass through all modules in reverse order
   */
  backward(gradOutput: Tensor): Tensor {
    let grad = gradOutput;

    for (let i = this._modules.length - 1; i >= 0; i--) {
      if (this._modules[i].backward) {
        grad = this._modules[i].backward!(grad);
      }
    }

    return grad;
  }

  /**
   * Get all trainable parameters from all child modules
   */
  parameters(): Tensor[] {
    const params: Tensor[] = [];

    for (const module of this._modules) {
      if (module.parameters) {
        params.push(...module.parameters());
      }
    }

    return params;
  }

  /**
   * Get number of modules in the sequence
   */
  size(): number {
    return this._modules.length;
  }

  /**
   * Get module at the given index
   */
  get(index: number): NNModule | undefined {
    return this._modules[index];
  }

  /**
   * Return the underlying list of child modules (for gradient updates)
   */
  getModules(): NNModule[] {
    return this._modules;
  }
}

/**
 * Parallel container — splits input along inputDimension, feeds each chunk
 * to a separate branch, then concatenates outputs along outputDimension
 */
export class Parallel implements NNModule {
  type = 'Parallel';
  private _modules: NNModule[];
  private _inputDimension: number;
  private _outputDimension: number;

  constructor(inputDimension: number, outputDimension: number) {
    this._modules = [];
    this._inputDimension = inputDimension;
    this._outputDimension = outputDimension;
  }

  /**
   * Add a branch module to the parallel container
   */
  add(module: NNModule): this {
    this._modules.push(module);
    return this;
  }

  /**
   * Forward pass — split input along inputDimension, run each chunk through
   * the corresponding branch, concatenate outputs along outputDimension
   */
  forward(input: Tensor): Tensor {
    if (this._modules.length === 0) {
      return input;
    }

    const chunks = this._splitTensor(input, this._inputDimension, this._modules.length);
    const outputs: Tensor[] = [];

    for (let i = 0; i < this._modules.length; i++) {
      outputs.push(this._modules[i].forward(chunks[i]));
    }

    return this._concatenate(outputs, this._outputDimension);
  }

  /**
   * Backward pass — split gradOutput among branches, call each branch's
   * backward, then concatenate input gradients along inputDimension
   */
  backward(gradOutput: Tensor): Tensor {
    if (this._modules.length === 0) {
      return gradOutput;
    }

    const gradChunks = this._splitTensor(gradOutput, this._outputDimension, this._modules.length);
    const inputGrads: Tensor[] = [];

    for (let i = 0; i < this._modules.length; i++) {
      if (this._modules[i].backward) {
        inputGrads.push(this._modules[i].backward!(gradChunks[i]));
      } else {
        inputGrads.push(gradChunks[i]);
      }
    }

    return this._concatenate(inputGrads, this._inputDimension);
  }

  /**
   * Collect trainable parameters from all branches
   */
  parameters(): Tensor[] {
    const params: Tensor[] = [];

    for (const module of this._modules) {
      if (module.parameters) {
        params.push(...module.parameters());
      }
    }

    return params;
  }

  /** Split a tensor into n equal chunks along the specified dimension */
  private _splitTensor(tensor: Tensor, dim: number, n: number): Tensor[] {
    const shape = tensor.shape;
    const data = tensor.data as Float32Array;

    // Determine the size along the split dimension
    const dimSize = shape[dim] ?? data.length;
    const chunkSize = Math.floor(dimSize / n);
    const chunks: Tensor[] = [];

    if (shape.length <= 1 || dim >= shape.length) {
      // Flat split
      for (let i = 0; i < n; i++) {
        const start = i * chunkSize;
        const end = i === n - 1 ? data.length : start + chunkSize;
        const chunkData = data.slice(start, end);
        chunks.push({ shape: [chunkData.length], data: chunkData, dtype: 'float32' });
      }

      return chunks;
    }

    // Multi-dimensional split along dim
    const outerSize = shape.slice(0, dim).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(dim + 1).reduce((a, b) => a * b, 1);

    for (let i = 0; i < n; i++) {
      const start = i * chunkSize;
      const len = i === n - 1 ? dimSize - start : chunkSize;
      const chunkData = new Float32Array(outerSize * len * innerSize);
      const newShape = [...shape];
      newShape[dim] = len;

      for (let o = 0; o < outerSize; o++) {
        for (let c = 0; c < len; c++) {
          for (let inn = 0; inn < innerSize; inn++) {
            const srcIdx = (o * dimSize + (start + c)) * innerSize + inn;
            const dstIdx = (o * len + c) * innerSize + inn;
            chunkData[dstIdx] = data[srcIdx];
          }
        }
      }

      chunks.push({ shape: newShape, data: chunkData, dtype: 'float32' });
    }

    return chunks;
  }

  /** Concatenate tensors along the specified dimension */
  private _concatenate(tensors: Tensor[], dim: number): Tensor {
    if (tensors.length === 0) {
      throw new Error('No tensors to concatenate');
    }

    if (tensors.length === 1) {
      return tensors[0];
    }

    const firstShape = tensors[0].shape;

    if (firstShape.length <= 1 || dim >= firstShape.length) {
      // Flat concatenation
      const totalSize = tensors.reduce((sum, t) => sum + t.data.length, 0);
      const result = new Float32Array(totalSize);
      let offset = 0;

      for (const t of tensors) {
        result.set(t.data as Float32Array, offset);
        offset += t.data.length;
      }

      return { shape: [totalSize], data: result, dtype: 'float32' };
    }

    const outerSize = firstShape.slice(0, dim).reduce((a, b) => a * b, 1);
    const innerSize = firstShape.slice(dim + 1).reduce((a, b) => a * b, 1);
    const totalDimSize = tensors.reduce((sum, t) => sum + (t.shape[dim] ?? 0), 0);
    const newShape = [...firstShape];
    newShape[dim] = totalDimSize;

    const resultData = new Float32Array(outerSize * totalDimSize * innerSize);

    let dimOffset = 0;

    for (const t of tensors) {
      const tDimSize = t.shape[dim] ?? 0;
      const tData = t.data as Float32Array;

      for (let o = 0; o < outerSize; o++) {
        for (let c = 0; c < tDimSize; c++) {
          for (let inn = 0; inn < innerSize; inn++) {
            const srcIdx = (o * tDimSize + c) * innerSize + inn;
            const dstIdx = (o * totalDimSize + dimOffset + c) * innerSize + inn;
            resultData[dstIdx] = tData[srcIdx];
          }
        }
      }

      dimOffset += tDimSize;
    }

    return { shape: newShape, data: resultData, dtype: 'float32' };
  }
}

/**
 * ConcatTable — passes the same input to every branch module and
 * concatenates all outputs into a single tensor
 */
export class ConcatTable implements NNModule {
  type = 'ConcatTable';
  private _modules: NNModule[];

  constructor() {
    this._modules = [];
  }

  /**
   * Add a branch module
   */
  add(module: NNModule): this {
    this._modules.push(module);
    return this;
  }

  /**
   * Forward pass — each module receives the same input; outputs are
   * concatenated into a single tensor
   */
  forward(input: Tensor): Tensor {
    const outputs: Tensor[] = [];

    for (const module of this._modules) {
      outputs.push(module.forward(input));
    }

    const totalSize = outputs.reduce((sum, t) => sum + t.data.length, 0);
    const result: Tensor = {
      shape: [outputs.length, outputs[0].data.length],
      data: new Float32Array(totalSize),
      dtype: 'float32',
    };

    let offset = 0;

    for (const output of outputs) {
      (result.data as Float32Array).set(output.data as Float32Array, offset);
      offset += output.data.length;
    }

    return result;
  }

  /**
   * Backward pass — split gradOutput evenly among branches, pass each
   * chunk to its branch's backward, then sum the resulting input gradients
   */
  backward(gradOutput: Tensor): Tensor {
    if (this._modules.length === 0) {
      return gradOutput;
    }

    const gradData = gradOutput.data as Float32Array;
    const chunkSize = Math.floor(gradData.length / this._modules.length);
    const inputGrads: Float32Array[] = [];

    for (let i = 0; i < this._modules.length; i++) {
      const start = i * chunkSize;
      const end = i === this._modules.length - 1 ? gradData.length : start + chunkSize;
      const chunkData = gradData.slice(start, end);
      const chunk: Tensor = { shape: [chunkData.length], data: chunkData, dtype: 'float32' };

      if (this._modules[i].backward) {
        const grad = this._modules[i].backward!(chunk);
        inputGrads.push(grad.data as Float32Array);
      } else {
        inputGrads.push(chunkData);
      }
    }

    // Sum all input gradients (each branch received the same input)
    const inputLen = inputGrads[0].length;
    const sumGrad = new Float32Array(inputLen);

    for (const g of inputGrads) {
      for (let j = 0; j < inputLen; j++) {
        sumGrad[j] += g[j];
      }
    }

    return { shape: [inputLen], data: sumGrad, dtype: 'float32' };
  }

  /**
   * Collect trainable parameters from all branch modules
   */
  parameters(): Tensor[] {
    const params: Tensor[] = [];

    for (const module of this._modules) {
      if (module.parameters) {
        params.push(...module.parameters());
      }
    }

    return params;
  }
}

/**
 * Create a new sequential model
 */
export function nn(): Sequential {
  return new Sequential();
}

/**
 * Create a parallel model
 */
export function parallel(inputDim: number, outputDim: number): Parallel {
  return new Parallel(inputDim, outputDim);
}

/**
 * Create a concat table
 */
export function concatTable(): ConcatTable {
  return new ConcatTable();
}
