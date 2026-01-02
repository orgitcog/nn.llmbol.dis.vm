/**
 * Tensor Operations (GGML-style)
 * 
 * Low-level tensor operations optimized for performance
 */

import type { Tensor } from './ml.m';

export class TensorOps {
  /**
   * Reshape a tensor
   */
  static reshape(tensor: Tensor, newShape: number[]): Tensor {
    const oldSize = tensor.shape.reduce((a, b) => a * b, 1);
    const newSize = newShape.reduce((a, b) => a * b, 1);

    if (oldSize !== newSize) {
      throw new Error('Cannot reshape: size mismatch');
    }

    return {
      shape: newShape,
      data: tensor.data,
      dtype: tensor.dtype,
    };
  }

  /**
   * Transpose a 2D tensor
   */
  static transpose(tensor: Tensor): Tensor {
    if (tensor.shape.length !== 2) {
      throw new Error('transpose only supports 2D tensors');
    }

    const [rows, cols] = tensor.shape;
    const result: Tensor = {
      shape: [cols, rows],
      data: new Float32Array(rows * cols),
      dtype: 'float32',
    };

    const inputData = tensor.data as Float32Array;
    const outputData = result.data as Float32Array;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        outputData[j * rows + i] = inputData[i * cols + j];
      }
    }

    return result;
  }

  /**
   * Element-wise multiplication
   */
  static multiply(a: Tensor, b: Tensor): Tensor {
    if (a.shape.length !== b.shape.length || 
        !a.shape.every((val, idx) => val === b.shape[idx])) {
      throw new Error('Tensor shapes must match');
    }

    const result: Tensor = {
      shape: [...a.shape],
      data: new Float32Array(a.data.length),
      dtype: 'float32',
    };

    const aData = a.data as Float32Array;
    const bData = b.data as Float32Array;
    const resultData = result.data as Float32Array;

    for (let i = 0; i < aData.length; i++) {
      resultData[i] = aData[i] * bData[i];
    }

    return result;
  }

  /**
   * Scalar multiplication
   */
  static scale(tensor: Tensor, scalar: number): Tensor {
    const result: Tensor = {
      shape: [...tensor.shape],
      data: new Float32Array(tensor.data.length),
      dtype: 'float32',
    };

    const inputData = tensor.data as Float32Array;
    const outputData = result.data as Float32Array;

    for (let i = 0; i < inputData.length; i++) {
      outputData[i] = inputData[i] * scalar;
    }

    return result;
  }

  /**
   * Concatenate tensors along an axis
   */
  static concat(tensors: Tensor[], axis: number = 0): Tensor {
    if (tensors.length === 0) {
      throw new Error('Need at least one tensor to concatenate');
    }

    // Validate shapes
    const firstShape = tensors[0].shape;
    for (let i = 1; i < tensors.length; i++) {
      for (let j = 0; j < firstShape.length; j++) {
        if (j !== axis && tensors[i].shape[j] !== firstShape[j]) {
          throw new Error('Tensor shapes must match except on concat axis');
        }
      }
    }

    // Calculate output shape
    const outputShape = [...firstShape];
    outputShape[axis] = tensors.reduce((sum, t) => sum + t.shape[axis], 0);

    // Calculate total size
    const totalSize = outputShape.reduce((a, b) => a * b, 1);
    
    const result: Tensor = {
      shape: outputShape,
      data: new Float32Array(totalSize),
      dtype: 'float32',
    };

    // Copy data (simplified for axis 0)
    if (axis === 0) {
      let offset = 0;
      for (const tensor of tensors) {
        const data = tensor.data as Float32Array;
        (result.data as Float32Array).set(data, offset);
        offset += data.length;
      }
    }

    return result;
  }

  /**
   * Reduce sum along an axis
   */
  static sum(tensor: Tensor, axis?: number): Tensor {
    if (axis === undefined) {
      // Sum all elements
      const data = tensor.data as Float32Array;
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        sum += data[i];
      }
      return {
        shape: [1],
        data: Float32Array.from([sum]),
        dtype: 'float32',
      };
    }

    // Sum along specific axis (simplified implementation)
    throw new Error('Axis-specific sum not yet implemented');
  }

  /**
   * Get tensor statistics
   */
  static stats(tensor: Tensor): { min: number; max: number; mean: number; std: number } {
    const data = tensor.data as Float32Array;
    let min = data[0];
    let max = data[0];
    let sum = 0;

    for (let i = 0; i < data.length; i++) {
      const val = data[i];
      min = Math.min(min, val);
      max = Math.max(max, val);
      sum += val;
    }

    const mean = sum / data.length;
    
    let sqSum = 0;
    for (let i = 0; i < data.length; i++) {
      const diff = data[i] - mean;
      sqSum += diff * diff;
    }
    const std = Math.sqrt(sqSum / data.length);

    return { min, max, mean, std };
  }
}
