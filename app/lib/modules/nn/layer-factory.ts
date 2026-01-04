/**
 * Layer Factory
 * 
 * Factory for creating neural network layers with various configurations
 */

import type { NNModule, LinearConfig, ConvConfig } from './nn.b';
import { 
  Linear, 
  ReLU, 
  Tanh, 
  Sigmoid, 
  Dropout, 
  BatchNorm, 
  Conv1d 
} from './nn-modules';

export type LayerType = 
  | 'linear'
  | 'relu'
  | 'tanh'
  | 'sigmoid'
  | 'dropout'
  | 'batchnorm'
  | 'conv1d';

export interface LayerConfig {
  type: LayerType;
  params?: any;
}

export class LayerFactory {
  /**
   * Create a layer based on type and configuration
   */
  static create(config: LayerConfig): NNModule {
    const availableTypes = ['linear', 'relu', 'tanh', 'sigmoid', 'dropout', 'batchnorm', 'conv1d'];
    
    switch (config.type) {
      case 'linear':
        return this.createLinear(config.params);
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
        return this.createConv1d(config.params);
      default:
        throw new Error(
          `Unknown layer type: ${config.type}. Available types: ${availableTypes.join(', ')}`
        );
    }
  }

  /**
   * Create a linear layer
   */
  private static createLinear(params: any): Linear {
    const config: LinearConfig = {
      inputSize: params.inputSize || params.in || 128,
      outputSize: params.outputSize || params.out || 128,
      bias: params.bias !== false,
    };
    return new Linear(config);
  }

  /**
   * Create a convolutional layer
   */
  private static createConv1d(params: any): Conv1d {
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
   * Create multiple layers from configurations
   */
  static createMany(configs: LayerConfig[]): NNModule[] {
    return configs.map(config => this.create(config));
  }

  /**
   * Create a simple feedforward network
   */
  static createFeedforward(
    inputSize: number,
    hiddenSizes: number[],
    outputSize: number,
    activation: LayerType = 'relu',
    dropout: number = 0
  ): LayerConfig[] {
    const configs: LayerConfig[] = [];
    
    let prevSize = inputSize;
    
    for (const hiddenSize of hiddenSizes) {
      // Linear layer
      configs.push({
        type: 'linear',
        params: { inputSize: prevSize, outputSize: hiddenSize },
      });
      
      // Activation
      configs.push({ type: activation });
      
      // Dropout
      if (dropout > 0) {
        configs.push({ type: 'dropout', params: { p: dropout } });
      }
      
      prevSize = hiddenSize;
    }
    
    // Output layer
    configs.push({
      type: 'linear',
      params: { inputSize: prevSize, outputSize },
    });
    
    return configs;
  }

  /**
   * Create a convolutional network
   */
  static createConvNet(
    inputChannels: number,
    channels: number[],
    kernelSizes: number[],
    activation: LayerType = 'relu'
  ): LayerConfig[] {
    if (channels.length !== kernelSizes.length) {
      throw new Error('channels and kernelSizes must have same length');
    }

    const configs: LayerConfig[] = [];
    let prevChannels = inputChannels;
    
    for (let i = 0; i < channels.length; i++) {
      // Conv layer
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
      
      // Batch norm
      configs.push({
        type: 'batchnorm',
        params: { numFeatures: channels[i] },
      });
      
      // Activation
      configs.push({ type: activation });
      
      prevChannels = channels[i];
    }
    
    return configs;
  }

  /**
   * Create a residual block configuration
   */
  static createResidualBlock(
    size: number,
    activation: LayerType = 'relu'
  ): LayerConfig[] {
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

export function linear(inputSize: number, outputSize: number, bias: boolean = true): NNModule {
  return LayerFactory.create({
    type: 'linear',
    params: { inputSize, outputSize, bias },
  });
}

export function relu(): NNModule {
  return new ReLU();
}

export function tanh(): NNModule {
  return new Tanh();
}

export function sigmoid(): NNModule {
  return new Sigmoid();
}

export function dropout(p: number = 0.5): NNModule {
  return new Dropout(p);
}

export function batchNorm(numFeatures: number, eps?: number): NNModule {
  return new BatchNorm(numFeatures, eps);
}

export function conv1d(
  inChannels: number,
  outChannels: number,
  kernelSize: number,
  stride: number = 1,
  padding: number = 0
): NNModule {
  return LayerFactory.create({
    type: 'conv1d',
    params: { inChannels, outChannels, kernelSize, stride, padding },
  });
}
