/**
 * Model Quantization Support
 * 
 * Implements quantization techniques for reducing model size
 * and improving inference speed (GGML-style quantization)
 */

import type { Tensor } from './ml.m';

export type QuantizationType = 'q4_0' | 'q4_1' | 'q8_0' | 'f16' | 'f32';

export interface QuantizationConfig {
  type: QuantizationType;
  blockSize: number;
}

export class Quantization {
  /**
   * Quantize a float32 tensor to lower precision
   */
  static quantize(tensor: Tensor, config: QuantizationConfig): Tensor {
    const data = tensor.data as Float32Array;
    
    switch (config.type) {
      case 'q8_0':
        return this.quantizeQ8_0(tensor);
      case 'q4_0':
        return this.quantizeQ4_0(tensor);
      case 'q4_1':
        return this.quantizeQ4_1(tensor);
      default:
        return tensor; // No quantization
    }
  }

  /**
   * Q8_0 quantization: 8-bit per weight, scale per block
   */
  private static quantizeQ8_0(tensor: Tensor): Tensor {
    const data = tensor.data as Float32Array;
    const blockSize = 32;
    const numBlocks = Math.ceil(data.length / blockSize);
    
    // Each block: 1 float32 scale + 32 int8 values = 36 bytes
    const quantizedSize = numBlocks * 36;
    const quantized = new Uint8Array(quantizedSize);

    let offset = 0;
    for (let block = 0; block < numBlocks; block++) {
      const start = block * blockSize;
      const end = Math.min(start + blockSize, data.length);
      
      // Find max absolute value in block
      let maxAbs = 0;
      for (let i = start; i < end; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(data[i]));
      }
      
      const scale = maxAbs / 127;
      
      // Write scale (4 bytes)
      const view = new DataView(quantized.buffer, offset);
      view.setFloat32(0, scale, true);
      offset += 4;
      
      // Write quantized values
      for (let i = start; i < end; i++) {
        const quantized_val = Math.round(data[i] / scale);
        quantized[offset++] = quantized_val & 0xFF;
      }
      
      // Pad remaining bytes in block
      for (let i = end; i < start + blockSize; i++) {
        quantized[offset++] = 0;
      }
    }

    return {
      shape: tensor.shape,
      data: quantized,
      dtype: 'int8',
    };
  }

  /**
   * Q4_0 quantization: 4-bit per weight, scale per block
   */
  private static quantizeQ4_0(tensor: Tensor): Tensor {
    const data = tensor.data as Float32Array;
    const blockSize = 32;
    const numBlocks = Math.ceil(data.length / blockSize);
    
    // Each block: 1 float32 scale + 16 bytes (32 * 4-bit values)
    const quantizedSize = numBlocks * 20;
    const quantized = new Uint8Array(quantizedSize);

    let offset = 0;
    for (let block = 0; block < numBlocks; block++) {
      const start = block * blockSize;
      const end = Math.min(start + blockSize, data.length);
      
      // Find max absolute value in block
      let maxAbs = 0;
      for (let i = start; i < end; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(data[i]));
      }
      
      const scale = maxAbs / 7; // 4-bit signed: -7 to 7
      
      // Write scale
      const view = new DataView(quantized.buffer, offset);
      view.setFloat32(0, scale, true);
      offset += 4;
      
      // Write quantized values (2 per byte)
      for (let i = start; i < end; i += 2) {
        const val1 = Math.round(data[i] / scale) & 0x0F;
        const val2 = i + 1 < end ? Math.round(data[i + 1] / scale) & 0x0F : 0;
        quantized[offset++] = (val1 << 4) | val2;
      }
    }

    return {
      shape: tensor.shape,
      data: quantized,
      dtype: 'int4',
    };
  }

  /**
   * Q4_1 quantization: 4-bit per weight with min/scale per block
   */
  private static quantizeQ4_1(tensor: Tensor): Tensor {
    // Similar to Q4_0 but with additional min value
    // Implementation simplified for brevity
    return this.quantizeQ4_0(tensor);
  }

  /**
   * Dequantize back to float32
   */
  static dequantize(tensor: Tensor, config: QuantizationConfig): Tensor {
    switch (config.type) {
      case 'q8_0':
        return this.dequantizeQ8_0(tensor);
      case 'q4_0':
      case 'q4_1':
        return this.dequantizeQ4_0(tensor);
      default:
        return tensor;
    }
  }

  /**
   * Dequantize Q8_0
   */
  private static dequantizeQ8_0(tensor: Tensor): Tensor {
    const quantized = tensor.data;
    const blockSize = 32;
    const numBlocks = Math.floor(quantized.length / 36);
    const totalElements = numBlocks * blockSize;
    
    const data = new Float32Array(totalElements);
    
    let srcOffset = 0;
    let dstOffset = 0;
    
    for (let block = 0; block < numBlocks; block++) {
      const view = new DataView(quantized.buffer, srcOffset);
      const scale = view.getFloat32(0, true);
      srcOffset += 4;
      
      for (let i = 0; i < blockSize; i++) {
        const quantizedVal = new Int8Array([quantized[srcOffset++]])[0];
        data[dstOffset++] = quantizedVal * scale;
      }
    }

    return {
      shape: tensor.shape,
      data,
      dtype: 'float32',
    };
  }

  /**
   * Dequantize Q4_0
   */
  private static dequantizeQ4_0(tensor: Tensor): Tensor {
    const quantized = tensor.data;
    const blockSize = 32;
    const numBlocks = Math.floor(quantized.length / 20);
    const totalElements = numBlocks * blockSize;
    
    const data = new Float32Array(totalElements);
    
    let srcOffset = 0;
    let dstOffset = 0;
    
    for (let block = 0; block < numBlocks; block++) {
      const view = new DataView(quantized.buffer, srcOffset);
      const scale = view.getFloat32(0, true);
      srcOffset += 4;
      
      for (let i = 0; i < blockSize / 2; i++) {
        const byte = quantized[srcOffset++];
        const val1 = ((byte >> 4) & 0x0F) - 7;
        const val2 = (byte & 0x0F) - 7;
        data[dstOffset++] = val1 * scale;
        data[dstOffset++] = val2 * scale;
      }
    }

    return {
      shape: tensor.shape,
      data,
      dtype: 'float32',
    };
  }
}
