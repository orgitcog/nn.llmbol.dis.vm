/**
 * Model Builder
 *
 * High-level API for building neural network models dynamically
 */

import { Sequential } from '~/lib/modules/nn/nn.b';
import { LayerFactory, type LayerConfig } from '~/lib/modules/nn/layer-factory';
import type { Tensor } from '~/lib/modules/ml/ml.m';

export interface ModelArchitecture {
  name: string;
  layers: LayerConfig[];
  inputShape: number[];
  outputShape: number[];
}

export class ModelBuilder {
  private _model: Sequential;
  private _architecture: ModelArchitecture;

  constructor(name: string, inputShape: number[]) {
    this._model = new Sequential();
    this._architecture = {
      name,
      layers: [],
      inputShape,
      outputShape: inputShape,
    };
  }

  /**
   * Add a single layer to the model
   */
  addLayer(config: LayerConfig): this {
    const layer = LayerFactory.create(config);
    this._model.add(layer);
    this._architecture.layers.push(config);

    return this;
  }

  /**
   * Add multiple layers at once
   */
  addLayers(configs: LayerConfig[]): this {
    for (const config of configs) {
      this.addLayer(config);
    }

    return this;
  }

  /**
   * Add a Linear (fully connected) layer
   */
  linear(inputSize: number, outputSize: number, bias: boolean = true): this {
    return this.addLayer({ type: 'linear', params: { inputSize, outputSize, bias } });
  }

  /**
   * Add a ReLU activation layer
   */
  relu(): this {
    return this.addLayer({ type: 'relu' });
  }

  /**
   * Add a Tanh activation layer
   */
  tanh(): this {
    return this.addLayer({ type: 'tanh' });
  }

  /**
   * Add a Sigmoid activation layer
   */
  sigmoid(): this {
    return this.addLayer({ type: 'sigmoid' });
  }

  /**
   * Add a Dropout layer
   */
  dropout(p: number = 0.5): this {
    return this.addLayer({ type: 'dropout', params: { p } });
  }

  /**
   * Add a Batch Normalisation layer
   */
  batchNorm(numFeatures: number, eps?: number): this {
    return this.addLayer({ type: 'batchnorm', params: { numFeatures, eps } });
  }

  /**
   * Add a 1-D Convolutional layer
   */
  conv1d(inChannels: number, outChannels: number, kernelSize: number, stride: number = 1, padding: number = 0): this {
    return this.addLayer({ type: 'conv1d', params: { inChannels, outChannels, kernelSize, stride, padding } });
  }

  /**
   * Add a Layer Normalisation layer
   */
  layerNorm(normalizedShape: number[], eps?: number): this {
    return this.addLayer({ type: 'layernorm', params: { normalizedShape, eps } });
  }

  /**
   * Add a Multi-Head Attention layer
   */
  multiHeadAttention(embedDim: number, numHeads: number): this {
    return this.addLayer({ type: 'multiheadattention', params: { embedDim, numHeads } });
  }

  /**
   * Add a token Embedding layer
   */
  embedding(numEmbeddings: number, embeddingDim: number): this {
    return this.addLayer({ type: 'embedding', params: { numEmbeddings, embeddingDim } });
  }

  /**
   * Build a complete feedforward block
   */
  feedforward(
    inputSize: number,
    hiddenSizes: number[],
    outputSize: number,
    activation: 'relu' | 'tanh' | 'sigmoid' = 'relu',
    dropout: number = 0,
  ): this {
    const configs = LayerFactory.createFeedforward(inputSize, hiddenSizes, outputSize, activation, dropout);
    return this.addLayers(configs);
  }

  /**
   * Build a convolutional block
   */
  convBlock(
    inputChannels: number,
    channels: number[],
    kernelSizes: number[],
    activation: 'relu' | 'tanh' | 'sigmoid' = 'relu',
  ): this {
    const configs = LayerFactory.createConvNet(inputChannels, channels, kernelSizes, activation);
    return this.addLayers(configs);
  }

  /**
   * Finalise and return the underlying Sequential model
   */
  build(): Sequential {
    return this._model;
  }

  /**
   * Return the recorded model architecture
   */
  getArchitecture(): ModelArchitecture {
    return this._architecture;
  }

  /**
   * Run a forward pass through the model
   */
  forward(input: Tensor): Tensor {
    return this._model.forward(input);
  }

  /**
   * Return all trainable parameters from the model
   */
  parameters(): Tensor[] {
    return this._model.parameters();
  }

  /**
   * Simple SGD training loop.
   *
   * For each epoch and each (input, target) pair:
   *   1. Forward pass through the model
   *   2. Compute loss via lossFn
   *   3. Backward pass using MSE-style gradient (output − target)
   *   4. Update parameters: param -= lr * grad
   *
   * @returns Array of average loss values, one per epoch
   */
  train(
    inputs: Tensor[],
    targets: Tensor[],
    lossFn: (output: Tensor, target: Tensor) => Tensor,
    lr: number = 0.01,
    epochs: number = 1,
  ): number[] {
    const losses: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;

      for (let i = 0; i < inputs.length; i++) {
        // Forward pass
        const output = this._model.forward(inputs[i]);
        const loss = lossFn(output, targets[i]);
        epochLoss += (loss.data as Float32Array)[0] ?? 0;

        // Compute initial gradient: (output − target) / n  (MSE derivative)
        const outData = output.data as Float32Array;
        const targetData = targets[i].data as Float32Array;
        const n = outData.length;
        const gradData = new Float32Array(n);

        for (let j = 0; j < n; j++) {
          gradData[j] = (outData[j] - targetData[j]) / n;
        }

        const grad: Tensor = { shape: output.shape, data: gradData, dtype: 'float32' };

        // Backward pass
        this._model.backward(grad);

        // Parameter update — iterate through modules that expose gradients
        for (const mod of this._model.getModules()) {
          if (!mod.parameters || !mod.gradients) {
            continue;
          }

          const params = mod.parameters();
          const grads = mod.gradients();

          for (let p = 0; p < params.length && p < grads.length; p++) {
            const pData = params[p].data as Float32Array;
            const gData = grads[p].data as Float32Array;

            for (let k = 0; k < pData.length; k++) {
              pData[k] -= lr * gData[k];
            }
          }
        }
      }

      losses.push(epochLoss / inputs.length);
    }

    return losses;
  }

  /**
   * Return a human-readable model summary
   */
  summary(): string {
    const lines: string[] = [];
    lines.push(`Model: ${this._architecture.name}`);
    lines.push('='.repeat(60));
    lines.push('Layer (type)'.padEnd(30) + 'Output Shape');
    lines.push('='.repeat(60));

    for (let i = 0; i < this._architecture.layers.length; i++) {
      const layer = this._architecture.layers[i];
      const layerName = `${layer.type}_${i}`;
      lines.push(layerName.padEnd(30) + 'Dynamic');
    }

    lines.push('='.repeat(60));

    const params = this.parameters();
    const totalParams = params.reduce((sum, p) => sum + p.data.length, 0);
    lines.push(`Total parameters: ${totalParams.toLocaleString()}`);
    lines.push('='.repeat(60));

    return lines.join('\n');
  }

  /**
   * Serialise the model architecture to JSON
   */
  toJSON(): string {
    return JSON.stringify(this._architecture, null, 2);
  }

  /**
   * Reconstruct a ModelBuilder from a JSON architecture string
   */
  static fromJSON(json: string): ModelBuilder {
    const arch: ModelArchitecture = JSON.parse(json);
    const builder = new ModelBuilder(arch.name, arch.inputShape);
    builder.addLayers(arch.layers);
    builder._architecture.outputShape = arch.outputShape;

    return builder;
  }
}

/**
 * Create a new ModelBuilder
 */
export function buildModel(name: string, inputShape: number[]): ModelBuilder {
  return new ModelBuilder(name, inputShape);
}

/**
 * Create a simple feedforward network and return the built Sequential model
 */
export function createFeedforwardModel(
  name: string,
  inputSize: number,
  hiddenSizes: number[],
  outputSize: number,
  activation: 'relu' | 'tanh' | 'sigmoid' = 'relu',
  dropout: number = 0,
): Sequential {
  return new ModelBuilder(name, [inputSize])
    .feedforward(inputSize, hiddenSizes, outputSize, activation, dropout)
    .build();
}

/**
 * Create a simple convolutional network and return the built Sequential model
 */
export function createConvModel(
  name: string,
  inputChannels: number,
  channels: number[],
  kernelSizes: number[],
  activation: 'relu' | 'tanh' | 'sigmoid' = 'relu',
): Sequential {
  return new ModelBuilder(name, [inputChannels]).convBlock(inputChannels, channels, kernelSizes, activation).build();
}
