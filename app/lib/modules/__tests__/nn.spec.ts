import { describe, expect, it, beforeEach } from 'vitest';
import { nn, Sequential, Parallel } from '../nn/nn.b';
import { Linear, ReLU, Dropout } from '../nn/nn-modules';
import { LayerFactory, linear, relu } from '../nn/layer-factory';
import { ModelBuilder, buildModel, createFeedforwardModel } from '../nn/model-builder';
import type { Tensor } from '../ml/ml.m';

describe('Sequential', () => {
  it('should create an empty sequential model', () => {
    const model = nn();
    expect(model).toBeInstanceOf(Sequential);
    expect(model.size()).toBe(0);
  });

  it('should add layers to sequential model', () => {
    const model = nn()
      .add(new Linear({ inputSize: 10, outputSize: 5, bias: true }))
      .add(new ReLU());
    
    expect(model.size()).toBe(2);
  });

  it('should perform forward pass', () => {
    const model = nn()
      .add(new Linear({ inputSize: 3, outputSize: 2, bias: true }))
      .add(new ReLU());
    
    const input: Tensor = {
      shape: [1, 3],
      data: new Float32Array([1, 2, 3]),
      dtype: 'float32',
    };
    
    const output = model.forward(input);
    expect(output).toBeDefined();
    expect(output.shape).toEqual([1, 2]);
  });

  it('should get parameters from all layers', () => {
    const model = nn()
      .add(new Linear({ inputSize: 3, outputSize: 2, bias: true }))
      .add(new ReLU())
      .add(new Linear({ inputSize: 2, outputSize: 1, bias: true }));
    
    const params = model.parameters();
    // First Linear: weight + bias, Second Linear: weight + bias
    expect(params.length).toBe(4);
  });
});

describe('Neural Network Modules', () => {
  describe('Linear Layer', () => {
    it('should create a linear layer', () => {
      const layer = new Linear({
        inputSize: 10,
        outputSize: 5,
        bias: true,
      });
      
      expect(layer.type).toBe('Linear');
    });

    it('should forward propagate', () => {
      const layer = new Linear({
        inputSize: 3,
        outputSize: 2,
        bias: true,
      });
      
      const input: Tensor = {
        shape: [1, 3],
        data: new Float32Array([1, 2, 3]),
        dtype: 'float32',
      };
      
      const output = layer.forward(input);
      expect(output.shape).toEqual([1, 2]);
    });

    it('should have parameters', () => {
      const layer = new Linear({
        inputSize: 3,
        outputSize: 2,
        bias: true,
      });
      
      const params = layer.parameters!();
      expect(params.length).toBe(2); // weight and bias
    });
  });

  describe('ReLU', () => {
    it('should apply ReLU activation', () => {
      const layer = new ReLU();
      const input: Tensor = {
        shape: [4],
        data: new Float32Array([-2, -1, 0, 1]),
        dtype: 'float32',
      };
      
      const output = layer.forward(input);
      const expected = [0, 0, 0, 1];
      
      expect(Array.from(output.data as Float32Array)).toEqual(expected);
    });
  });

  describe('Dropout', () => {
    it('should create a dropout layer', () => {
      const layer = new Dropout(0.5);
      expect(layer.type).toBe('Dropout');
    });

    it('should forward propagate', () => {
      const layer = new Dropout(0.5);
      const input: Tensor = {
        shape: [4],
        data: new Float32Array([1, 2, 3, 4]),
        dtype: 'float32',
      };
      
      const output = layer.forward(input);
      expect(output.shape).toEqual(input.shape);
    });
  });
});

describe('LayerFactory', () => {
  it('should create a linear layer', () => {
    const layer = LayerFactory.create({
      type: 'linear',
      params: { inputSize: 10, outputSize: 5 },
    });
    
    expect(layer.type).toBe('Linear');
  });

  it('should create activation layers', () => {
    const relu = LayerFactory.create({ type: 'relu' });
    const tanh = LayerFactory.create({ type: 'tanh' });
    const sigmoid = LayerFactory.create({ type: 'sigmoid' });
    
    expect(relu.type).toBe('ReLU');
    expect(tanh.type).toBe('Tanh');
    expect(sigmoid.type).toBe('Sigmoid');
  });

  it('should create feedforward network config', () => {
    const config = LayerFactory.createFeedforward(
      10,     // input size
      [64, 32], // hidden sizes
      5,      // output size
      'relu',
      0.2     // dropout
    );
    
    // Should have: linear + relu + dropout + linear + relu + dropout + linear
    expect(config.length).toBe(7);
  });

  it('should use helper functions', () => {
    const l = linear(10, 5);
    const r = relu();
    
    expect(l.type).toBe('Linear');
    expect(r.type).toBe('ReLU');
  });
});

describe('ModelBuilder', () => {
  it('should build a simple model', () => {
    const model = buildModel('test-model', [10])
      .linear(10, 5)
      .relu()
      .linear(5, 2)
      .build();
    
    expect(model).toBeInstanceOf(Sequential);
    expect(model.size()).toBe(3);
  });

  it('should build with method chaining', () => {
    const builder = buildModel('test-model', [10])
      .linear(10, 64)
      .relu()
      .dropout(0.5)
      .linear(64, 32)
      .relu()
      .linear(32, 10);
    
    const model = builder.build();
    expect(model.size()).toBe(6);
  });

  it('should build feedforward network', () => {
    const builder = buildModel('feedforward', [10])
      .feedforward(10, [64, 32], 5, 'relu', 0.2);
    
    const model = builder.build();
    expect(model.size()).toBeGreaterThan(0);
  });

  it('should provide model summary', () => {
    const builder = buildModel('test-model', [10])
      .linear(10, 5)
      .relu()
      .linear(5, 2);
    
    const summary = builder.summary();
    expect(summary).toContain('test-model');
    expect(summary).toContain('Total parameters');
  });

  it('should serialize to JSON', () => {
    const builder = buildModel('test-model', [10])
      .linear(10, 5)
      .relu();
    
    const json = builder.toJSON();
    const parsed = JSON.parse(json);
    
    expect(parsed.name).toBe('test-model');
    expect(parsed.layers.length).toBe(2);
  });

  it('should create feedforward model helper', () => {
    const model = createFeedforwardModel(
      'simple-ff',
      10,
      [64, 32],
      5,
      'relu',
      0.2
    );
    
    expect(model).toBeInstanceOf(Sequential);
  });
});

describe('Parallel', () => {
  it('should create a parallel container', () => {
    const parallel = new Parallel(0, 0);
    expect(parallel.type).toBe('Parallel');
  });

  it('should add modules to parallel', () => {
    const parallel = new Parallel(0, 0)
      .add(new Linear({ inputSize: 10, outputSize: 5, bias: true }))
      .add(new Linear({ inputSize: 10, outputSize: 3, bias: true }));
    
    expect(parallel).toBeDefined();
  });
});
