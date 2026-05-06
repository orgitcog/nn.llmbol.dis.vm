/**
 * Bridges between the NN/ML modules and the VM module.
 * NNVMBridge: compiles a Sequential model to InfernoVM bytecode modules.
 * MLVMBridge: serialises Tensor data into VM memory segments.
 */

import { InfernoVM } from '~/lib/modules/vm/diy.dis';
import { VMRuntime, type VMProcess } from '~/lib/modules/vm/vm-runtime';
import { Sequential } from '~/lib/modules/nn/nn.b';
import { BytecodeLoader } from '~/lib/modules/vm/bytecode-loader';
import { type Tensor } from '~/lib/modules/ml/ml.m';

export interface MemorySegment {
  address: number;
  length: number;
  dtype: string;
}

export class NNVMBridge {
  private _vm: InfernoVM;

  constructor(vm: InfernoVM) {
    this._vm = vm;
  }

  /**
   * Compile a Sequential model to InfernoVM bytecode modules.
   * One module per layer. Returns array of loaded module names.
   */
  async compileModel(modelName: string, model: Sequential): Promise<string[]> {
    const layers = model.getModules();
    const moduleNames: string[] = [];

    for (let i = 0; i < layers.length; i++) {
      // Simple bytecode: PUSH_INT 0 (placeholder), HALT
      const code = [0x03, 0, 0xff];
      const bytecode = BytecodeLoader.createModuleWithExports(code, [{ name: 'forward', address: 0 }]);
      const name = `${modelName}_layer_${i}`;
      await this._vm.loadModule(name, bytecode);
      moduleNames.push(name);
    }

    return moduleNames;
  }

  /**
   * Execute a compiled model by running each layer module in sequence.
   * Returns the final result.
   */
  async executeModel(modelName: string, layerCount: number, input: unknown): Promise<unknown> {
    const results: unknown[] = [input];

    for (let i = 0; i < layerCount; i++) {
      const result = await this._vm.execute(`${modelName}_layer_${i}`, 'forward', [results[results.length - 1]]);
      results.push(result);
    }

    return results[results.length - 1];
  }
}

export class MLVMBridge {
  private _runtime: VMRuntime;

  constructor(runtime: VMRuntime) {
    this._runtime = runtime;
  }

  /**
   * Serialise tensor data into a VMProcess memory segment.
   * Returns the segment descriptor (address and length).
   */
  writeTensor(process: VMProcess, tensor: Tensor, baseAddress: number = 0): MemorySegment {
    const floatData = tensor.data instanceof Float32Array ? tensor.data : new Float32Array(tensor.data);
    const shapeBytes = tensor.shape.length * 4;
    const dataBytes = floatData.length * 4;
    const view = new DataView(process.memory.buffer);

    // Write ndim
    view.setUint32(baseAddress, tensor.shape.length, true);

    // Write shape dims
    tensor.shape.forEach((dim, i) => view.setUint32(baseAddress + 4 + i * 4, dim, true));

    // Write float32 data
    const dataStart = baseAddress + 4 + shapeBytes;
    floatData.forEach((v, i) => view.setFloat32(dataStart + i * 4, v, true));

    return { address: baseAddress, length: 4 + shapeBytes + dataBytes, dtype: tensor.dtype };
  }

  /**
   * Read a tensor from a VMProcess memory segment.
   */
  readTensor(process: VMProcess, segment: MemorySegment): Tensor {
    const view = new DataView(process.memory.buffer);
    const ndim = view.getUint32(segment.address, true);
    const shape: number[] = [];

    for (let i = 0; i < ndim; i++) {
      shape.push(view.getUint32(segment.address + 4 + i * 4, true));
    }

    const dataStart = segment.address + 4 + ndim * 4;
    const totalElements = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(totalElements);

    for (let i = 0; i < totalElements; i++) {
      data[i] = view.getFloat32(dataStart + i * 4, true);
    }

    return { shape, data, dtype: 'float32' };
  }
}
