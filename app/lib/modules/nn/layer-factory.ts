/**
 * Layer Factory
 *
 * Factory for creating neural network layers with various configurations
 */

import type { NNModule, LinearConfig, ConvConfig } from '~/lib/modules/nn/nn.b';
import {
  Linear,
  ReLU,
  Tanh,
  Sigmoid,
  Dropout,
  BatchNorm,
  Conv1d,
  LayerNorm,
  MultiHeadAttention,
  Embedding,
} from '~/lib/modules/nn/nn-modules';

export type LayerType =
  | 'linear'
  | 'relu'
  | 'tanh'
  | 'sigmoid'
  | 'dropout'
  | 'batchnorm'
  | 'conv1d'
  | 'layernorm'
  | 'multiheadattention'
  | 'embedding';

export interface LayerConfig {
  type: LayerType;
  params?: any;
}

export class LayerFactory {
  /**
   * Create a layer instance from a config descriptor
   */
  static create(config: LayerConfig): NNModule {
    const availableTypes = [
      'linear',
      'relu',
      'tanh',
      'sigmoid',
      'dropout',
      'batchnorm',
      'conv1d',
      'layernorm',
      'multiheadattention',
      'embedding',
    ];

    switch (config.type) {
      case 'linear':
        return this._createLinear(config.params);
      case 'relu':
        return new ReLU();
      case 'tanh':
        return new Tanh();
      case 'sigmoid':
        return new Sigmoid();
      case 'dropout':
        return new Dropout(config.params?.p || 0.5);
      case 'batchnorm':
        return new BatchNorm(config.params?.numFeatures || 128, config.params?.eps);
      case 'conv1d':
        return this._createConv1d(config.params);
      case 'layernorm':
        return new LayerNorm(config.params?.normalizedShape || [128], config.params?.eps);
      case 'multiheadattention':
        return new MultiHeadAttention(config.params?.embedDim || 128, config.params?.numHeads || 4);
      case 'embedding':
        return new Embedding(config.params?.numEmbeddings || 1000, config.params?.embeddingDim || 128);
      default:
        throw new Error(`Unknown layer type: ${config.type}. Available types: ${availableTypes.join(', ')}`);
    }
  }

  /** Create a linear layer from raw params */
  private static _createLinear(params: any): Linear {
    const config: LinearConfig = {
      inputSize: params.inputSize || params.in || 128,
      outputSize: params.outputSize || params.out || 128,
      bias: params.bias !== false,
    };
    return new Linear(config);
  }

  /** Create a Conv1d layer from raw params */
  private static _createConv1d(params: any): Conv1d {
    const config: ConvConfig = {
      inChannels: params.inChannels || params.in || 1,
      outChannels: params.outChannels || params.out || 1,
      kernelSize: params.kernelSize || params.kernel || 3,
      stride: params.stride || 1,
      padding: params.padding || 0,
    };
    return new Conv1d(config);
  }

  /**
   * Create multiple layers from an array of config descriptors
   */
  static createMany(configs: LayerConfig[]): NNModule[] {
    return configs.map((config) => this.create(config));
  }

  /**
   * Generate configs for a simple feedforward network
   */
  static createFeedforward(
    inputSize: number,
    hiddenSizes: number[],
    outputSize: number,
    activation: LayerType = 'relu',
    dropout: number = 0,
  ): LayerConfig[] {
    const configs: LayerConfig[] = [];

    let prevSize = inputSize;

    for (const hiddenSize of hiddenSizes) {
      configs.push({ type: 'linear', params: { inputSize: prevSize, outputSize: hiddenSize } });
      configs.push({ type: activation });

      if (dropout > 0) {
        configs.push({ type: 'dropout', params: { p: dropout } });
      }

      prevSize = hiddenSize;
    }

    configs.push({ type: 'linear', params: { inputSize: prevSize, outputSize } });

    return configs;
  }

  /**
   * Generate configs for a convolutional network
   */
  static createConvNet(
    inputChannels: number,
    channels: number[],
    kernelSizes: number[],
    activation: LayerType = 'relu',
  ): LayerConfig[] {
    if (channels.length !== kernelSizes.length) {
      throw new Error('channels and kernelSizes must have same length');
    }

    const configs: LayerConfig[] = [];
    let prevChannels = inputChannels;

    for (let i = 0; i < channels.length; i++) {
      configs.push({
        type: 'conv1d',
        params: {
          inChannels: prevChannels,
          outChannels: channels[i],
          kernelSize: kernelSizes[i],
          stride: 1,
          padding: 0,
        },
      });
      configs.push({ type: 'batchnorm', params: { numFeatures: channels[i] } });
      configs.push({ type: activation });
      prevChannels = channels[i];
    }

    return configs;
  }

  /**
   * Generate configs for a residual block
   */
  static createResidualBlock(size: number, activation: LayerType = 'relu'): LayerConfig[] {
    return [
      { type: 'linear', params: { inputSize: size, outputSize: size } },
      { type: activation },
      { type: 'linear', params: { inputSize: size, outputSize: size } },
      { type: activation },
    ];
  }
}

/**
 * Helper functions for common layer patterns
 */

/** Create a Linear layer */
export function linear(inputSize: number, outputSize: number, bias: boolean = true): NNModule {
  return LayerFactory.create({ type: 'linear', params: { inputSize, outputSize, bias } });
}

/** Create a ReLU activation */
export function relu(): NNModule {
  return new ReLU();
}

/** Create a Tanh activation */
export function tanh(): NNModule {
  return new Tanh();
}

/** Create a Sigmoid activation */
export function sigmoid(): NNModule {
  return new Sigmoid();
}

/** Create a Dropout layer */
export function dropout(p: number = 0.5): NNModule {
  return new Dropout(p);
}

/** Create a BatchNorm layer */
export function batchNorm(numFeatures: number, eps?: number): NNModule {
  return new BatchNorm(numFeatures, eps);
}

/** Create a Conv1d layer */
export function conv1d(
  inChannels: number,
  outChannels: number,
  kernelSize: number,
  stride: number = 1,
  padding: number = 0,
): NNModule {
  return LayerFactory.create({ type: 'conv1d', params: { inChannels, outChannels, kernelSize, stride, padding } });
}

/** Create a LayerNorm layer */
export function layerNorm(normalizedShape: number[], eps?: number): NNModule {
  return new LayerNorm(normalizedShape, eps);
}

/** Create a MultiHeadAttention layer */
export function multiHeadAttention(embedDim: number, numHeads: number): NNModule {
  return new MultiHeadAttention(embedDim, numHeads);
}

/** Create an Embedding layer */
export function embedding(numEmbeddings: number, embeddingDim: number): NNModule {
  return new Embedding(numEmbeddings, embeddingDim);
}
