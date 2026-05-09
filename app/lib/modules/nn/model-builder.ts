/**
 * Model Builder
 *
 * High-level API for building neural network models dynamically
 */

import { Sequential } from '~/lib/modules/nn/nn.b';
import { LayerFactory, type LayerConfig } from '~/lib/modules/nn/layer-factory';
import type { Tensor } from '~/lib/modules/ml/ml.m';

export type OptimizerType = 'sgd' | 'adam' | 'adamw';

/** Optional hyper-parameters for Adam / AdamW. */
export interface AdamOptions {
  /** Exponential decay rate for the first moment (default 0.9). */
  beta1?: number;

  /** Exponential decay rate for the second moment (default 0.999). */
  beta2?: number;

  /** Numerical stability epsilon (default 1e-8). */
  epsilon?: number;

  /** Weight-decay coefficient for AdamW (default 0.01). */
  weightDecay?: number;
}

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
   * Training loop supporting SGD, Adam, and AdamW optimisers.
   *
   * For each epoch and each (input, target) pair:
   *   1. Forward pass through the model
   *   2. Compute loss via lossFn
   *   3. Backward pass using MSE-style gradient (output − target) / n
   *   4. Parameter update according to the chosen optimiser
   *
   * @returns Array of average loss values, one per epoch.
   */
  train(
    inputs: Tensor[],
    targets: Tensor[],
    lossFn: (output: Tensor, target: Tensor) => Tensor,
    lr: number = 0.01,
    epochs: number = 1,
    optimizer: OptimizerType = 'sgd',
    adamOptions: AdamOptions = {},
  ): number[] {
    const beta1 = adamOptions.beta1 ?? 0.9;
    const beta2 = adamOptions.beta2 ?? 0.999;
    const epsilon = adamOptions.epsilon ?? 1e-8;
    const weightDecay = adamOptions.weightDecay ?? 0.01;

    /*
     * Adam / AdamW moment accumulators.
     * Indexed as moments[paramIndex] = { m: Float32Array, v: Float32Array }.
     * Allocated lazily on the first pass so we know each parameter's size.
     */
    type MomentPair = { m: Float32Array; v: Float32Array };

    const moments: MomentPair[] = [];
    let adamStep = 0;

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

        if (optimizer === 'sgd') {
          // ── SGD ─────────────────────────────────────────────────────────
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
        } else {
          // ── Adam / AdamW ─────────────────────────────────────────────────
          adamStep++;

          // Bias-correction denominators
          const bc1 = 1 - Math.pow(beta1, adamStep);
          const bc2 = 1 - Math.pow(beta2, adamStep);

          let paramIdx = 0;

          for (const mod of this._model.getModules()) {
            if (!mod.parameters || !mod.gradients) {
              continue;
            }

            const params = mod.parameters();
            const grads = mod.gradients();

            for (let p = 0; p < params.length && p < grads.length; p++) {
              const pData = params[p].data as Float32Array;
              const gData = grads[p].data as Float32Array;

              // Lazy initialisation of moment buffers
              if (!moments[paramIdx]) {
                moments[paramIdx] = {
                  m: new Float32Array(pData.length),
                  v: new Float32Array(pData.length),
                };
              }

              const { m, v } = moments[paramIdx];

              for (let k = 0; k < pData.length; k++) {
                const g = gData[k];

                // Update biased first and second moment estimates
                m[k] = beta1 * m[k] + (1 - beta1) * g;
                v[k] = beta2 * v[k] + (1 - beta2) * g * g;

                // Compute bias-corrected estimates
                const mHat = m[k] / bc1;
                const vHat = v[k] / bc2;

                if (optimizer === 'adamw') {
                  // AdamW: apply decoupled weight decay before the gradient step
                  pData[k] *= 1 - lr * weightDecay;
                }

                pData[k] -= (lr * mHat) / (Math.sqrt(vHat) + epsilon);
              }

              paramIdx++;
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
   * Serialise the model architecture to JSON (without trained weights).
   * Use `save()` to include weight values.
   */
  toJSON(): string {
    return JSON.stringify(this._architecture, null, 2);
  }

  /**
   * Reconstruct a ModelBuilder from a JSON architecture string.
   * Weights are re-initialised randomly; use `load()` to restore trained weights.
   */
  static fromJSON(json: string): ModelBuilder {
    const arch: ModelArchitecture = JSON.parse(json);
    const builder = new ModelBuilder(arch.name, arch.inputShape);
    builder.addLayers(arch.layers);
    builder._architecture.outputShape = arch.outputShape;

    return builder;
  }

  /**
   * Serialise the model architecture AND all trained parameter values to JSON.
   * The resulting string can be restored (including weights) via `ModelBuilder.load()`.
   */
  save(): string {
    const layers = this._model.getModules();
    const weights = layers.map((layer) =>
      layer.parameters ? layer.parameters().map((p) => Array.from(p.data as Float32Array)) : [],
    );

    return JSON.stringify({ architecture: this._architecture, weights }, null, 2);
  }

  /**
   * Reconstruct a ModelBuilder from a JSON string produced by `save()`.
   * Both the architecture and the trained parameter values are restored.
   */
  static load(json: string): ModelBuilder {
    const parsed: { architecture: ModelArchitecture; weights?: number[][][] } = JSON.parse(json);
    const { architecture, weights } = parsed;

    const builder = new ModelBuilder(architecture.name, architecture.inputShape);
    builder.addLayers(architecture.layers);
    builder._architecture.outputShape = architecture.outputShape;

    // Restore parameter values when present (backwards-compatible with toJSON output)
    if (weights) {
      const layers = builder._model.getModules();

      for (let i = 0; i < layers.length && i < weights.length; i++) {
        const layer = layers[i];

        if (!layer.parameters || weights[i].length === 0) {
          continue;
        }

        const params = layer.parameters();

        for (let p = 0; p < params.length && p < weights[i].length; p++) {
          const pData = params[p].data as Float32Array;
          const saved = weights[i][p];

          for (let k = 0; k < pData.length && k < saved.length; k++) {
            pData[k] = saved[k];
          }
        }
      }
    }

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
