/**
 * Build-a-Bear Neural Network Architecture
 * 
 * Torch7-inspired modular neural network builder that allows
 * dynamic composition of neural network layers
 */

import type { Tensor } from '../ml/ml.m';

export interface NNModule {
  type: string;
  forward(input: Tensor): Tensor;
  backward?(gradOutput: Tensor): Tensor;
  parameters?(): Tensor[];
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
  private modules: NNModule[];

  constructor() {
    this.modules = [];
  }

  /**
   * Add a module to the sequence
   */
  add(module: NNModule): this {
    this.modules.push(module);
    return this;
  }

  /**
   * Forward pass through all modules
   */
  forward(input: Tensor): Tensor {
    let output = input;
    for (const module of this.modules) {
      output = module.forward(output);
    }
    return output;
  }

  /**
   * Backward pass through all modules (in reverse)
   */
  backward(gradOutput: Tensor): Tensor {
    let grad = gradOutput;
    for (let i = this.modules.length - 1; i >= 0; i--) {
      if (this.modules[i].backward) {
        grad = this.modules[i].backward!(grad);
      }
    }
    return grad;
  }

  /**
   * Get all parameters
   */
  parameters(): Tensor[] {
    const params: Tensor[] = [];
    for (const module of this.modules) {
      if (module.parameters) {
        params.push(...module.parameters());
      }
    }
    return params;
  }

  /**
   * Get number of modules
   */
  size(): number {
    return this.modules.length;
  }

  /**
   * Get module at index
   */
  get(index: number): NNModule | undefined {
    return this.modules[index];
  }
}

/**
 * Parallel container for branching networks
 */
export class Parallel implements NNModule {
  type = 'Parallel';
  private modules: NNModule[];
  private inputDimension: number;
  private outputDimension: number;

  constructor(inputDimension: number, outputDimension: number) {
    this.modules = [];
    this.inputDimension = inputDimension;
    this.outputDimension = outputDimension;
  }

  /**
   * Add a module to the parallel container
   */
  add(module: NNModule): this {
    this.modules.push(module);
    return this;
  }

  /**
   * Forward pass - concatenate outputs
   */
  forward(input: Tensor): Tensor {
    const outputs: Tensor[] = [];
    
    for (const module of this.modules) {
      outputs.push(module.forward(input));
    }

    // Concatenate along the output dimension
    return this.concatenate(outputs);
  }

  /**
   * Helper to concatenate tensors
   */
  private concatenate(tensors: Tensor[]): Tensor {
    if (tensors.length === 0) {
      throw new Error('No tensors to concatenate');
    }

    // Simplified concatenation along first dimension
    const totalSize = tensors.reduce((sum, t) => {
      const size = t.data.length;
      return sum + size;
    }, 0);

    const result: Tensor = {
      shape: [tensors.length, tensors[0].data.length],
      data: new Float32Array(totalSize),
      dtype: 'float32',
    };

    let offset = 0;
    for (const tensor of tensors) {
      (result.data as Float32Array).set(tensor.data as Float32Array, offset);
      offset += tensor.data.length;
    }

    return result;
  }

  parameters(): Tensor[] {
    const params: Tensor[] = [];
    for (const module of this.modules) {
      if (module.parameters) {
        params.push(...module.parameters());
      }
    }
    return params;
  }
}

/**
 * Concat table for joining multiple inputs
 */
export class ConcatTable implements NNModule {
  type = 'ConcatTable';
  private modules: NNModule[];

  constructor() {
    this.modules = [];
  }

  /**
   * Add a module
   */
  add(module: NNModule): this {
    this.modules.push(module);
    return this;
  }

  /**
   * Forward pass - each module gets same input, outputs concatenated
   */
  forward(input: Tensor): Tensor {
    const outputs: Tensor[] = [];
    
    for (const module of this.modules) {
      outputs.push(module.forward(input));
    }

    // Return as concatenated output
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

  parameters(): Tensor[] {
    const params: Tensor[] = [];
    for (const module of this.modules) {
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
