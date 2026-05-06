/**
 * Tensor Operations (GGML-style)
 *
 * Low-level tensor operations optimised for performance.
 * All methods are static; no instance state is required.
 */

import type { Tensor } from './ml.m';

export class TensorOps {
  /**
   * Reshape a tensor to `newShape` (must have the same total element count).
   */
  static reshape(tensor: Tensor, newShape: number[]): Tensor {
    const oldSize = tensor.shape.reduce((a, b) => a * b, 1);
    const newSize = newShape.reduce((a, b) => a * b, 1);

    if (oldSize !== newSize) {
      throw new Error('Cannot reshape: size mismatch');
    }

    return { shape: newShape, data: tensor.data, dtype: tensor.dtype };
  }

  /**
   * Transpose a tensor.
   *
   * - For 2-D tensors the default behaviour swaps rows and columns.
   * - For N-D tensors supply an explicit `axes` permutation array.
   * - When `axes` is omitted on an N-D tensor, all axes are reversed.
   */
  static transpose(tensor: Tensor, axes?: number[]): Tensor {
    const ndim = tensor.shape.length;

    let perm: number[];

    if (axes !== undefined) {
      if (axes.length !== ndim) {
        throw new Error('axes length must equal tensor rank');
      }

      perm = axes;
    } else {
      // Default: reverse all axes (matches NumPy behaviour)
      perm = Array.from({ length: ndim }, (_, i) => ndim - 1 - i);
    }

    const newShape = perm.map((p) => tensor.shape[p]);
    const totalSize = tensor.shape.reduce((a, b) => a * b, 1);

    const inputData = tensor.data as Float32Array;
    const outputData = new Float32Array(totalSize);

    // Strides for the *input* tensor
    const srcStrides = new Array<number>(ndim);
    let s = 1;

    for (let d = ndim - 1; d >= 0; d--) {
      srcStrides[d] = s;
      s *= tensor.shape[d];
    }

    // Strides for the *output* tensor
    const dstStrides = new Array<number>(ndim);
    let ds = 1;

    for (let d = ndim - 1; d >= 0; d--) {
      dstStrides[d] = ds;
      ds *= newShape[d];
    }

    for (let flatOut = 0; flatOut < totalSize; flatOut++) {
      // Multi-dim index in output space
      let tmp = flatOut;
      const multiOut = new Array<number>(ndim);

      for (let d = ndim - 1; d >= 0; d--) {
        multiOut[d] = tmp % newShape[d];
        tmp = Math.floor(tmp / newShape[d]);
      }

      // Map back to input: output dim d came from input dim perm[d]
      let srcFlat = 0;

      for (let d = 0; d < ndim; d++) {
        srcFlat += multiOut[d] * srcStrides[perm[d]];
      }

      outputData[flatOut] = inputData[srcFlat];
    }

    return { shape: newShape, data: outputData, dtype: 'float32' };
  }

  /**
   * Element-wise multiplication of two tensors (shapes must match exactly).
   */
  static multiply(a: Tensor, b: Tensor): Tensor {
    if (a.shape.length !== b.shape.length || !a.shape.every((v, i) => v === b.shape[i])) {
      throw new Error('Tensor shapes must match for element-wise multiply');
    }

    const resultData = new Float32Array(a.data.length);
    const aData = a.data as Float32Array;
    const bData = b.data as Float32Array;

    for (let i = 0; i < aData.length; i++) {
      resultData[i] = aData[i] * bData[i];
    }

    return { shape: [...a.shape], data: resultData, dtype: 'float32' };
  }

  /**
   * Multiply every element by a scalar.
   */
  static scale(tensor: Tensor, scalar: number): Tensor {
    const outputData = new Float32Array(tensor.data.length);
    const inputData = tensor.data as Float32Array;

    for (let i = 0; i < inputData.length; i++) {
      outputData[i] = inputData[i] * scalar;
    }

    return { shape: [...tensor.shape], data: outputData, dtype: 'float32' };
  }

  /**
   * Concatenate tensors along `axis`.  All axes are supported for N-D tensors.
   * Every tensor must have the same rank, and shapes must match on every axis
   * except the concat axis.
   */
  static concat(tensors: Tensor[], axis: number = 0): Tensor {
    if (tensors.length === 0) {
      throw new Error('Need at least one tensor to concatenate');
    }

    const firstShape = tensors[0].shape;
    const ndim = firstShape.length;

    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of range for ${ndim}-D tensors`);
    }

    for (let i = 1; i < tensors.length; i++) {
      if (tensors[i].shape.length !== ndim) {
        throw new Error('All tensors must have the same number of dimensions');
      }

      for (let d = 0; d < ndim; d++) {
        if (d !== axis && tensors[i].shape[d] !== firstShape[d]) {
          throw new Error('Tensor shapes must match except on the concat axis');
        }
      }
    }

    const outputShape = [...firstShape];
    outputShape[axis] = tensors.reduce((sum, t) => sum + t.shape[axis], 0);

    const totalSize = outputShape.reduce((a, b) => a * b, 1);
    const resultData = new Float32Array(totalSize);

    // Cumulative axis-dim offsets per tensor
    const cumSizes = [0];

    for (const t of tensors) {
      cumSizes.push(cumSizes[cumSizes.length - 1] + t.shape[axis]);
    }

    for (let flatOut = 0; flatOut < totalSize; flatOut++) {
      // Decompose flat output index into per-dimension indices
      let tmp = flatOut;
      const multiIdx = new Array<number>(ndim);

      for (let d = ndim - 1; d >= 0; d--) {
        multiIdx[d] = tmp % outputShape[d];
        tmp = Math.floor(tmp / outputShape[d]);
      }

      // Determine which source tensor owns the current axis position
      const axisIdx = multiIdx[axis];
      let ti = 0;

      while (ti < tensors.length - 1 && axisIdx >= cumSizes[ti + 1]) {
        ti++;
      }

      const srcTensor = tensors[ti];
      const srcMultiIdx = [...multiIdx];
      srcMultiIdx[axis] = axisIdx - cumSizes[ti];

      // Convert source multi-dim index to flat
      let srcFlat = 0;
      let stride = 1;

      for (let d = ndim - 1; d >= 0; d--) {
        srcFlat += srcMultiIdx[d] * stride;
        stride *= srcTensor.shape[d];
      }

      resultData[flatOut] = (srcTensor.data as Float32Array)[srcFlat];
    }

    return { shape: outputShape, data: resultData, dtype: 'float32' };
  }

  /**
   * Sum elements.
   *
   * - No `axis`: sum all elements → shape [1].
   * - With `axis`: reduce along that dimension (dimension is removed from shape).
   */
  static sum(tensor: Tensor, axis?: number): Tensor {
    const data = tensor.data as Float32Array;

    if (axis === undefined) {
      let total = 0;

      for (let i = 0; i < data.length; i++) {
        total += data[i];
      }

      return { shape: [1], data: Float32Array.from([total]), dtype: 'float32' };
    }

    return TensorOps._axisReduce(tensor, axis, (acc, v) => acc + v, 0, false);
  }

  /**
   * Mean of elements.
   *
   * - No `axis`: mean of all elements → shape [1].
   * - With `axis`: reduce along that dimension.
   */
  static mean(tensor: Tensor, axis?: number): Tensor {
    const data = tensor.data as Float32Array;

    if (axis === undefined) {
      let total = 0;

      for (let i = 0; i < data.length; i++) {
        total += data[i];
      }

      return { shape: [1], data: Float32Array.from([total / data.length]), dtype: 'float32' };
    }

    const sumTensor = TensorOps._axisReduce(tensor, axis, (acc, v) => acc + v, 0, false);
    const axisSize = tensor.shape[axis];
    const rd = sumTensor.data as Float32Array;

    for (let i = 0; i < rd.length; i++) {
      rd[i] /= axisSize;
    }

    return sumTensor;
  }

  /**
   * Maximum of elements.
   *
   * - No `axis`: global max → shape [1].
   * - With `axis`: per-slice max along that dimension.
   */
  static max(tensor: Tensor, axis?: number): Tensor {
    const data = tensor.data as Float32Array;

    if (axis === undefined) {
      let m = data[0];

      for (let i = 1; i < data.length; i++) {
        if (data[i] > m) {
          m = data[i];
        }
      }

      return { shape: [1], data: Float32Array.from([m]), dtype: 'float32' };
    }

    return TensorOps._axisReduce(tensor, axis, (acc, v) => Math.max(acc, v), -Infinity, false);
  }

  /**
   * Minimum of elements.
   *
   * - No `axis`: global min → shape [1].
   * - With `axis`: per-slice min along that dimension.
   */
  static min(tensor: Tensor, axis?: number): Tensor {
    const data = tensor.data as Float32Array;

    if (axis === undefined) {
      let m = data[0];

      for (let i = 1; i < data.length; i++) {
        if (data[i] < m) {
          m = data[i];
        }
      }

      return { shape: [1], data: Float32Array.from([m]), dtype: 'float32' };
    }

    return TensorOps._axisReduce(tensor, axis, (acc, v) => Math.min(acc, v), Infinity, false);
  }

  /**
   * Extract a sub-tensor.
   *
   * @param starts - Start index for each dimension.
   * @param sizes  - Number of elements to take in each dimension (-1 means "to end").
   */
  static slice(tensor: Tensor, starts: number[], sizes: number[]): Tensor {
    const ndim = tensor.shape.length;

    if (starts.length !== ndim || sizes.length !== ndim) {
      throw new Error('starts and sizes must have the same length as tensor rank');
    }

    const outShape = sizes.map((sz, d) => (sz === -1 ? tensor.shape[d] - starts[d] : sz));
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const resultData = new Float32Array(outSize);
    const srcData = tensor.data as Float32Array;

    // Strides for source tensor
    const srcStrides = new Array<number>(ndim);
    let s = 1;

    for (let d = ndim - 1; d >= 0; d--) {
      srcStrides[d] = s;
      s *= tensor.shape[d];
    }

    for (let flatOut = 0; flatOut < outSize; flatOut++) {
      let tmp = flatOut;
      let srcFlat = 0;

      for (let d = ndim - 1; d >= 0; d--) {
        const localIdx = tmp % outShape[d];
        tmp = Math.floor(tmp / outShape[d]);
        srcFlat += (starts[d] + localIdx) * srcStrides[d];
      }

      resultData[flatOut] = srcData[srcFlat];
    }

    return { shape: outShape, data: resultData, dtype: 'float32' };
  }

  /**
   * Gather slices from `tensor` along `axis` using an index array.
   *
   * Each entry in `indices` selects one slice along `axis`.
   * Result shape replaces the axis dimension with `indices.length`.
   *
   * @example
   * // 2-D tensor shape [4, 5], axis=0, indices=[1,3] → shape [2, 5]
   */
  static gather(tensor: Tensor, indices: number[], axis: number = 0): Tensor {
    const ndim = tensor.shape.length;

    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of range for ${ndim}-D tensor`);
    }

    const outShape = [...tensor.shape];
    outShape[axis] = indices.length;

    const outSize = outShape.reduce((a, b) => a * b, 1);
    const resultData = new Float32Array(outSize);
    const srcData = tensor.data as Float32Array;

    const srcStrides = new Array<number>(ndim);
    let s = 1;

    for (let d = ndim - 1; d >= 0; d--) {
      srcStrides[d] = s;
      s *= tensor.shape[d];
    }

    for (let flatOut = 0; flatOut < outSize; flatOut++) {
      let tmp = flatOut;
      const multiOut = new Array<number>(ndim);

      for (let d = ndim - 1; d >= 0; d--) {
        multiOut[d] = tmp % outShape[d];
        tmp = Math.floor(tmp / outShape[d]);
      }

      // Replace axis index with the gathered index
      const srcAxisIdx = indices[multiOut[axis]];
      let srcFlat = 0;

      for (let d = 0; d < ndim; d++) {
        srcFlat += (d === axis ? srcAxisIdx : multiOut[d]) * srcStrides[d];
      }

      resultData[flatOut] = srcData[srcFlat];
    }

    return { shape: outShape, data: resultData, dtype: 'float32' };
  }

  /**
   * Return min, max, mean, and standard deviation of the tensor's elements.
   */
  static stats(tensor: Tensor): { min: number; max: number; mean: number; std: number } {
    const data = tensor.data as Float32Array;
    let min = data[0];
    let max = data[0];
    let sum = 0;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];

      if (v < min) {
        min = v;
      }

      if (v > max) {
        max = v;
      }

      sum += v;
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

  /*
   * ---------------------------------------------------------------------------
   * Private helpers
   * ---------------------------------------------------------------------------
   */

  /**
   * Generic axis-wise reduction.
   *
   * @param tensor      - Input tensor.
   * @param axis        - Axis to reduce over (removed from output shape).
   * @param reduceFn    - `(accumulator, value) => newAccumulator`.
   * @param initial     - Seed value for the accumulator.
   * @param _unused     - Reserved (kept for future overloads).
   */
  private static _axisReduce(
    tensor: Tensor,
    axis: number,
    reduceFn: (acc: number, val: number) => number,
    initial: number,
    _unused: boolean,
  ): Tensor {
    const shape = tensor.shape;
    const ndim = shape.length;
    const data = tensor.data as Float32Array;

    const outShape = shape.filter((_, i) => i !== axis);
    const outSize = outShape.length > 0 ? outShape.reduce((a, b) => a * b, 1) : 1;
    const axisSize = shape[axis];
    const resultData = new Float32Array(outSize);

    // Source strides
    const srcStrides = new Array<number>(ndim);
    let s = 1;

    for (let d = ndim - 1; d >= 0; d--) {
      srcStrides[d] = s;
      s *= shape[d];
    }

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      // Recover multi-dim index in the *reduced* shape (axis dim omitted)
      let tmp = outFlat;
      const outMulti = new Array<number>(outShape.length);

      for (let d = outShape.length - 1; d >= 0; d--) {
        outMulti[d] = tmp % outShape[d];
        tmp = Math.floor(tmp / outShape[d]);
      }

      let acc = initial;

      for (let k = 0; k < axisSize; k++) {
        // Rebuild full multi-dim index with k on the reduced axis
        let srcFlat = 0;
        let outDim = 0;

        for (let d = 0; d < ndim; d++) {
          srcFlat += (d === axis ? k : outMulti[outDim++]) * srcStrides[d];
        }

        acc = reduceFn(acc, data[srcFlat]);
      }

      resultData[outFlat] = acc;
    }

    return {
      shape: outShape.length > 0 ? outShape : [1],
      data: resultData,
      dtype: 'float32',
    };
  }
}
