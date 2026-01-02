/**
 * Model Builder
 * 
 * High-level API for building neural network models dynamically
 */

import { Sequential, Parallel, ConcatTable } from './nn.b';
import type { NNModule } from './nn.b';
import { LayerFactory, type LayerConfig } from './layer-factory';
import type { Tensor } from '../ml/ml.m';

export interface ModelArchitecture {
  name: string;
  layers: LayerConfig[];
  inputShape: number[];
  outputShape: number[];
}

export class ModelBuilder {
  private model: Sequential;
  private architecture: ModelArchitecture;

  constructor(name: string, inputShape: number[]) {
    this.model = new Sequential();
    this.architecture = {
      name,
      layers: [],
      inputShape,
      outputShape: inputShape,
    };
  }

  /**
   * Add a layer to the model
   */
  addLayer(config: LayerConfig): this {
    const layer = LayerFactory.create(config);
    this.model.add(layer);
    this.architecture.layers.push(config);
    return this;
  }

  /**
   * Add multiple layers
   */
  addLayers(configs: LayerConfig[]): this {
    for (const config of configs) {
      this.addLayer(config);
    }
    return this;
  }

  /**
   * Add a linear layer
   */
  linear(inputSize: number, outputSize: number, bias: boolean = true): this {
    return this.addLayer({
      type: 'linear',
      params: { inputSize, outputSize, bias },
    });
  }

  /**
   * Add ReLU activation
   */
  relu(): this {
    return this.addLayer({ type: 'relu' });
  }

  /**
   * Add Tanh activation
   */
  tanh(): this {
    return this.addLayer({ type: 'tanh' });
  }

  /**
   * Add Sigmoid activation
   */
  sigmoid(): this {
    return this.addLayer({ type: 'sigmoid' });
  }

  /**
   * Add Dropout
   */
  dropout(p: number = 0.5): this {
    return this.addLayer({ type: 'dropout', params: { p } });
  }

  /**
   * Add Batch Normalization
   */
  batchNorm(numFeatures: number, eps?: number): this {
    return this.addLayer({
      type: 'batchnorm',
      params: { numFeatures, eps },
    });
  }

  /**
   * Add 1D Convolution
   */
  conv1d(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    stride: number = 1,
    padding: number = 0
  ): this {
    return this.addLayer({
      type: 'conv1d',
      params: { inChannels, outChannels, kernelSize, stride, padding },
    });
  }

  /**
   * Build a feedforward block
   */
  feedforward(
    inputSize: number,
    hiddenSizes: number[],
    outputSize: number,
    activation: 'relu' | 'tanh' | 'sigmoid' = 'relu',
    dropout: number = 0
  ): this {
    const configs = LayerFactory.createFeedforward(
      inputSize,
      hiddenSizes,
      outputSize,
      activation,
      dropout
    );
    return this.addLayers(configs);
  }

  /**
   * Build a convolutional block
   */
  convBlock(
    inputChannels: number,
    channels: number[],
    kernelSizes: number[],
    activation: 'relu' | 'tanh' | 'sigmoid' = 'relu'
  ): this {
    const configs = LayerFactory.createConvNet(
      inputChannels,
      channels,
      kernelSizes,
      activation
    );
    return this.addLayers(configs);
  }

  /**
   * Build the model
   */
  build(): Sequential {
    return this.model;
  }

  /**
   * Get model architecture
   */
  getArchitecture(): ModelArchitecture {
    return this.architecture;
  }

  /**
   * Forward pass through the model
   */
  forward(input: Tensor): Tensor {
    return this.model.forward(input);
  }

  /**
   * Get model parameters
   */
  parameters(): Tensor[] {
    return this.model.parameters();
  }

  /**
   * Get model summary
   */
  summary(): string {
    const lines: string[] = [];
    lines.push(`Model: ${this.architecture.name}`);
    lines.push('=' .repeat(60));
    lines.push('Layer (type)'.padEnd(30) + 'Output Shape');
    lines.push('=' .repeat(60));
    
    for (let i = 0; i < this.architecture.layers.length; i++) {
      const layer = this.architecture.layers[i];
      const layerName = `${layer.type}_${i}`;
      lines.push(layerName.padEnd(30) + 'Dynamic');
    }
    
    lines.push('=' .repeat(60));
    
    const params = this.parameters();
    const totalParams = params.reduce((sum, p) => sum + p.data.length, 0);
    lines.push(`Total parameters: ${totalParams.toLocaleString()}`);
    lines.push('=' .repeat(60));
    
    return lines.join('\n');
  }

  /**
   * Save model architecture to JSON
   */
  toJSON(): string {
    return JSON.stringify(this.architecture, null, 2);
  }

  /**
   * Load model architecture from JSON
   */
  static fromJSON(json: string): ModelBuilder {
    const arch: ModelArchitecture = JSON.parse(json);
    const builder = new ModelBuilder(arch.name, arch.inputShape);
    builder.addLayers(arch.layers);
    builder.architecture.outputShape = arch.outputShape;
    return builder;
  }
}

/**
 * Create a new model builder
 */
export function buildModel(name: string, inputShape: number[]): ModelBuilder {
  return new ModelBuilder(name, inputShape);
}

/**
 * Create a simple feedforward network
 */
export function createFeedforwardModel(
  name: string,
  inputSize: number,
  hiddenSizes: number[],
  outputSize: number,
  activation: 'relu' | 'tanh' | 'sigmoid' = 'relu',
  dropout: number = 0
): Sequential {
  return new ModelBuilder(name, [inputSize])
    .feedforward(inputSize, hiddenSizes, outputSize, activation, dropout)
    .build();
}

/**
 * Create a simple convolutional network
 */
export function createConvModel(
  name: string,
  inputChannels: number,
  channels: number[],
  kernelSizes: number[],
  activation: 'relu' | 'tanh' | 'sigmoid' = 'relu'
): Sequential {
  return new ModelBuilder(name, [inputChannels])
    .convBlock(inputChannels, channels, kernelSizes, activation)
    .build();
}
