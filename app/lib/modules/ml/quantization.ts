/**
 * Model Quantization Support
 *
 * Implements GGML-style quantization for reducing model size and
 * improving inference speed.
 */

import type { Tensor } from './ml.m';

export type QuantizationType = 'q4_0' | 'q4_1' | 'q8_0' | 'f16' | 'f32';

export interface QuantizationConfig {
  type: QuantizationType;
  blockSize: number;
}

export class Quantization {
  /**
   * Quantize a float32 tensor to lower precision.
   */
  static quantize(tensor: Tensor, config: QuantizationConfig): Tensor {
    switch (config.type) {
      case 'q8_0':
        return Quantization._quantizeQ80(tensor);
      case 'q4_0':
        return Quantization._quantizeQ40(tensor);
      case 'q4_1':
        return Quantization._quantizeQ41(tensor);
      default:
        return tensor;
    }
  }

  /**
   * Dequantize a tensor back to float32.
   */
  static dequantize(tensor: Tensor, config: QuantizationConfig): Tensor {
    switch (config.type) {
      case 'q8_0':
        return Quantization._dequantizeQ80(tensor);
      case 'q4_0':
        return Quantization._dequantizeQ40(tensor);
      case 'q4_1':
        return Quantization._dequantizeQ41(tensor);
      default:
        return tensor;
    }
  }

  /*
   * ---------------------------------------------------------------------------
   * Q8_0 — 8-bit symmetric, scale per 32-element block
   * Block layout: [float32 scale (4 B)] [32 × int8 (32 B)] = 36 B/block
   * ---------------------------------------------------------------------------
   */

  private static _quantizeQ80(tensor: Tensor): Tensor {
    const data = tensor.data as Float32Array;
    const blockSize = 32;
    const numBlocks = Math.ceil(data.length / blockSize);
    const quantized = new Uint8Array(numBlocks * 36);

    let offset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const start = block * blockSize;
      const end = Math.min(start + blockSize, data.length);

      let maxAbs = 0;

      for (let i = start; i < end; i++) {
        const a = Math.abs(data[i]);

        if (a > maxAbs) {
          maxAbs = a;
        }
      }

      const scale = maxAbs / 127;

      const view = new DataView(quantized.buffer, quantized.byteOffset + offset);
      view.setFloat32(0, scale, true);
      offset += 4;

      for (let i = start; i < end; i++) {
        quantized[offset++] = Math.round(data[i] / (scale || 1)) & 0xff;
      }

      // Pad partial last block
      for (let i = end; i < start + blockSize; i++) {
        quantized[offset++] = 0;
      }
    }

    return { shape: tensor.shape, data: quantized, dtype: 'int8' };
  }

  private static _dequantizeQ80(tensor: Tensor): Tensor {
    const quantized = tensor.data as Uint8Array;
    const blockSize = 32;
    const numBlocks = Math.floor(quantized.length / 36);
    const data = new Float32Array(numBlocks * blockSize);

    // Use byteOffset-aware DataView so sliced TypedArrays work correctly
    const view = new DataView(quantized.buffer, quantized.byteOffset);
    let srcOffset = 0;
    let dstOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const scale = view.getFloat32(srcOffset, true);
      srcOffset += 4;

      for (let i = 0; i < blockSize; i++) {
        // Reinterpret as signed int8
        const raw = quantized[srcOffset++];
        const quantizedVal = raw >= 128 ? raw - 256 : raw;
        data[dstOffset++] = quantizedVal * scale;
      }
    }

    return { shape: tensor.shape, data, dtype: 'float32' };
  }

  /*
   * ---------------------------------------------------------------------------
   * Q4_0 — 4-bit symmetric (unsigned with offset 8), scale per 32-element block
   * Block layout: [float32 scale (4 B)] [16 B packed nibbles] = 20 B/block
   * Encoding:  nibble = clamp(round(x / scale) + 8, 0, 15)
   * Decoding:  x = (nibble - 8) * scale
   * ---------------------------------------------------------------------------
   */

  private static _quantizeQ40(tensor: Tensor): Tensor {
    const data = tensor.data as Float32Array;
    const blockSize = 32;
    const numBlocks = Math.ceil(data.length / blockSize);
    const quantized = new Uint8Array(numBlocks * 20);

    let offset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const start = block * blockSize;
      const end = Math.min(start + blockSize, data.length);

      let maxAbs = 0;

      for (let i = start; i < end; i++) {
        const a = Math.abs(data[i]);

        if (a > maxAbs) {
          maxAbs = a;
        }
      }

      // scale maps ±maxAbs to ±7 (range -8..7 after offset)
      const scale = maxAbs / 7;

      const view = new DataView(quantized.buffer, quantized.byteOffset + offset);
      view.setFloat32(0, scale, true);
      offset += 4;

      for (let i = start; i < end; i += 2) {
        const q1 = Math.max(0, Math.min(15, Math.round(data[i] / (scale || 1)) + 8));
        const q2 = i + 1 < end ? Math.max(0, Math.min(15, Math.round(data[i + 1] / (scale || 1)) + 8)) : 8;
        quantized[offset++] = (q1 << 4) | q2;
      }
    }

    return { shape: tensor.shape, data: quantized, dtype: 'int4' };
  }

  private static _dequantizeQ40(tensor: Tensor): Tensor {
    const quantized = tensor.data as Uint8Array;
    const blockSize = 32;
    const numBlocks = Math.floor(quantized.length / 20);
    const data = new Float32Array(numBlocks * blockSize);

    // byteOffset-aware view so sliced TypedArrays are handled correctly
    const view = new DataView(quantized.buffer, quantized.byteOffset);
    let srcOffset = 0;
    let dstOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const scale = view.getFloat32(srcOffset, true);
      srcOffset += 4;

      for (let i = 0; i < blockSize / 2; i++) {
        const byte = quantized[srcOffset++];

        // Symmetric 4-bit: subtract 8 to recover signed value in [-8, 7]
        const val1 = ((byte >> 4) & 0x0f) - 8;
        const val2 = (byte & 0x0f) - 8;
        data[dstOffset++] = val1 * scale;
        data[dstOffset++] = val2 * scale;
      }
    }

    return { shape: tensor.shape, data, dtype: 'float32' };
  }

  /*
   * ---------------------------------------------------------------------------
   * Q4_1 — 4-bit unsigned, min + scale per 32-element block
   * Block layout: [float32 min (4 B)] [float32 scale (4 B)] [16 B packed nibbles] = 24 B/block
   * Encoding:  nibble = clamp(round((x - min) / scale), 0, 15)
   * Decoding:  x = nibble * scale + min
   * ---------------------------------------------------------------------------
   */

  private static _quantizeQ41(tensor: Tensor): Tensor {
    const data = tensor.data as Float32Array;
    const blockSize = 32;
    const numBlocks = Math.ceil(data.length / blockSize);

    // 4 (min) + 4 (scale) + 16 (nibbles) = 24 bytes per block
    const quantized = new Uint8Array(numBlocks * 24);

    let offset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const start = block * blockSize;
      const end = Math.min(start + blockSize, data.length);

      let minVal = data[start];
      let maxVal = data[start];

      for (let i = start + 1; i < end; i++) {
        if (data[i] < minVal) {
          minVal = data[i];
        }

        if (data[i] > maxVal) {
          maxVal = data[i];
        }
      }

      const range = maxVal - minVal;
      const scale = range / 15; // 4-bit unsigned: 0..15

      const view = new DataView(quantized.buffer, quantized.byteOffset + offset);
      view.setFloat32(0, minVal, true);
      view.setFloat32(4, scale, true);
      offset += 8;

      for (let i = start; i < end; i += 2) {
        const q1 = Math.max(0, Math.min(15, Math.round((data[i] - minVal) / (scale || 1))));
        const q2 = i + 1 < end ? Math.max(0, Math.min(15, Math.round((data[i + 1] - minVal) / (scale || 1)))) : 0;
        quantized[offset++] = (q1 << 4) | q2;
      }
    }

    return { shape: tensor.shape, data: quantized, dtype: 'int4' };
  }

  private static _dequantizeQ41(tensor: Tensor): Tensor {
    const quantized = tensor.data as Uint8Array;
    const blockSize = 32;
    const numBlocks = Math.floor(quantized.length / 24);
    const data = new Float32Array(numBlocks * blockSize);

    const view = new DataView(quantized.buffer, quantized.byteOffset);
    let srcOffset = 0;
    let dstOffset = 0;

    for (let block = 0; block < numBlocks; block++) {
      const minVal = view.getFloat32(srcOffset, true);
      srcOffset += 4;

      const scale = view.getFloat32(srcOffset, true);
      srcOffset += 4;

      for (let i = 0; i < blockSize / 2; i++) {
        const byte = quantized[srcOffset++];
        const q1 = (byte >> 4) & 0x0f;
        const q2 = byte & 0x0f;
        data[dstOffset++] = q1 * scale + minVal;
        data[dstOffset++] = q2 * scale + minVal;
      }
    }

    return { shape: tensor.shape, data, dtype: 'float32' };
  }
}
