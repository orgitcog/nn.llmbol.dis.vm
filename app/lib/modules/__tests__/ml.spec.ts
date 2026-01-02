import { describe, expect, it, beforeEach } from 'vitest';
import { MLModule, createMLModule, getGlobalMLModule } from '../ml/ml.m';
import { TensorOps } from '../ml/tensor-ops';
import { Quantization } from '../ml/quantization';
import type { Tensor } from '../ml/ml.m';

describe('MLModule', () => {
  let ml: MLModule;

  beforeEach(() => {
    ml = createMLModule();
  });

  describe('Tensor Creation', () => {
    it('should create a float32 tensor', () => {
      const tensor = ml.createTensor([2, 3], 'float32');
      
      expect(tensor.shape).toEqual([2, 3]);
      expect(tensor.dtype).toBe('float32');
      expect(tensor.data).toBeInstanceOf(Float32Array);
      expect(tensor.data.length).toBe(6);
    });

    it('should create tensors with different dtypes', () => {
      const float32 = ml.createTensor([2, 2], 'float32');
      const uint8 = ml.createTensor([2, 2], 'uint8');
      const int8 = ml.createTensor([2, 2], 'int8');
      
      expect(float32.data).toBeInstanceOf(Float32Array);
      expect(uint8.data).toBeInstanceOf(Uint8Array);
      expect(int8.data).toBeInstanceOf(Int8Array);
    });
  });

  describe('Matrix Operations', () => {
    it('should perform matrix multiplication', () => {
      const a = ml.createTensor([2, 3], 'float32');
      const b = ml.createTensor([3, 2], 'float32');
      
      // Fill with test data
      (a.data as Float32Array).set([1, 2, 3, 4, 5, 6]);
      (b.data as Float32Array).set([7, 8, 9, 10, 11, 12]);
      
      const result = ml.matmul(a, b);
      
      expect(result.shape).toEqual([2, 2]);
      expect(result.data.length).toBe(4);
    });

    it('should throw error for incompatible shapes', () => {
      const a = ml.createTensor([2, 3], 'float32');
      const b = ml.createTensor([2, 2], 'float32');
      
      expect(() => ml.matmul(a, b)).toThrow();
    });
  });

  describe('Element-wise Operations', () => {
    it('should perform element-wise addition', () => {
      const a = ml.createTensor([2, 2], 'float32');
      const b = ml.createTensor([2, 2], 'float32');
      
      (a.data as Float32Array).set([1, 2, 3, 4]);
      (b.data as Float32Array).set([5, 6, 7, 8]);
      
      const result = ml.add(a, b);
      const expected = [6, 8, 10, 12];
      
      expect(Array.from(result.data as Float32Array)).toEqual(expected);
    });
  });

  describe('Activation Functions', () => {
    it('should apply ReLU', () => {
      const input = ml.createTensor([4], 'float32');
      (input.data as Float32Array).set([-2, -1, 0, 1]);
      
      const result = ml.relu(input);
      const expected = [0, 0, 0, 1];
      
      expect(Array.from(result.data as Float32Array)).toEqual(expected);
    });

    it('should apply softmax', () => {
      const input = ml.createTensor([3], 'float32');
      (input.data as Float32Array).set([1, 2, 3]);
      
      const result = ml.softmax(input);
      const sum = Array.from(result.data as Float32Array).reduce((a, b) => a + b, 0);
      
      // Softmax should sum to 1
      expect(sum).toBeCloseTo(1.0, 5);
    });
  });

  describe('Model Management', () => {
    it('should load a model', () => {
      const config = {
        vocabSize: 1000,
        hiddenSize: 256,
        numLayers: 4,
        numHeads: 8,
        maxSequenceLength: 512,
      };
      
      const model = ml.loadModel('test-model', config);
      
      expect(model.name).toBe('test-model');
      expect(model.config).toEqual(config);
    });

    it('should retrieve a loaded model', () => {
      const config = {
        vocabSize: 1000,
        hiddenSize: 256,
        numLayers: 4,
        numHeads: 8,
        maxSequenceLength: 512,
      };
      
      ml.loadModel('test-model', config);
      const retrieved = ml.getModel('test-model');
      
      expect(retrieved).toBeDefined();
      expect(retrieved?.name).toBe('test-model');
    });
  });
});

describe('TensorOps', () => {
  describe('Reshape', () => {
    it('should reshape a tensor', () => {
      const tensor: Tensor = {
        shape: [2, 3],
        data: new Float32Array([1, 2, 3, 4, 5, 6]),
        dtype: 'float32',
      };
      
      const reshaped = TensorOps.reshape(tensor, [3, 2]);
      
      expect(reshaped.shape).toEqual([3, 2]);
      expect(reshaped.data).toBe(tensor.data);
    });

    it('should throw error for size mismatch', () => {
      const tensor: Tensor = {
        shape: [2, 3],
        data: new Float32Array(6),
        dtype: 'float32',
      };
      
      expect(() => TensorOps.reshape(tensor, [2, 2])).toThrow();
    });
  });

  describe('Transpose', () => {
    it('should transpose a 2D tensor', () => {
      const tensor: Tensor = {
        shape: [2, 3],
        data: new Float32Array([1, 2, 3, 4, 5, 6]),
        dtype: 'float32',
      };
      
      const transposed = TensorOps.transpose(tensor);
      
      expect(transposed.shape).toEqual([3, 2]);
    });
  });

  describe('Statistics', () => {
    it('should compute tensor statistics', () => {
      const tensor: Tensor = {
        shape: [5],
        data: new Float32Array([1, 2, 3, 4, 5]),
        dtype: 'float32',
      };
      
      const stats = TensorOps.stats(tensor);
      
      expect(stats.min).toBe(1);
      expect(stats.max).toBe(5);
      expect(stats.mean).toBe(3);
      expect(stats.std).toBeGreaterThan(0);
    });
  });
});

describe('Quantization', () => {
  describe('Q8_0 Quantization', () => {
    it('should quantize to Q8_0', () => {
      const tensor: Tensor = {
        shape: [64],
        data: new Float32Array(64).fill(0.5),
        dtype: 'float32',
      };
      
      const quantized = Quantization.quantize(tensor, {
        type: 'q8_0',
        blockSize: 32,
      });
      
      expect(quantized.dtype).toBe('int8');
      expect(quantized.shape).toEqual(tensor.shape);
    });

    it('should dequantize Q8_0', () => {
      const tensor: Tensor = {
        shape: [64],
        data: new Float32Array(64).fill(0.5),
        dtype: 'float32',
      };
      
      const quantized = Quantization.quantize(tensor, {
        type: 'q8_0',
        blockSize: 32,
      });
      
      const dequantized = Quantization.dequantize(quantized, {
        type: 'q8_0',
        blockSize: 32,
      });
      
      expect(dequantized.dtype).toBe('float32');
      expect(dequantized.shape).toEqual(tensor.shape);
    });
  });

  describe('Q4_0 Quantization', () => {
    it('should quantize to Q4_0', () => {
      const tensor: Tensor = {
        shape: [64],
        data: new Float32Array(64).fill(0.3),
        dtype: 'float32',
      };
      
      const quantized = Quantization.quantize(tensor, {
        type: 'q4_0',
        blockSize: 32,
      });
      
      expect(quantized.dtype).toBe('int4');
      expect(quantized.data.length).toBeLessThan(tensor.data.length);
    });
  });
});
