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

  /** Size in bytes of the bytecode file header (magic, version, entry, data, code, checksum). */
  private static readonly _HEADER_SIZE = 24;

  constructor(vm: InfernoVM) {
    this._vm = vm;
  }

  /**
   * Compile a Sequential model to InfernoVM bytecode modules.
   * One module per layer with layer-type-specific bytecode.
   * Returns array of loaded module names.
   *
   * Each exported 'forward' function expects its first numeric argument to
   * have been written to process memory address 0 as a little-endian float32
   * (the InfernoVM.loadModule wrapper handles this automatically).
   */
  async compileModel(modelName: string, model: Sequential): Promise<string[]> {
    const layers = model.getModules();
    const moduleNames: string[] = [];

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const name = `${modelName}_layer_${i}`;

      // Export table: one entry 'forward' whose address is the code-section start.
      const exportTable = [{ name: 'forward', address: 0 }];
      const codeStart = this._computeCodeStart(exportTable);
      exportTable[0].address = codeStart;

      let code: number[];

      switch (layer.type) {
        case 'ReLU':
          code = this._encodeReLU(codeStart);
          break;
        case 'Tanh':
          code = this._encodeTanh(codeStart);
          break;
        case 'Sigmoid':
          code = this._encodeSigmoid(codeStart);
          break;
        default:
          // For parametric layers (Linear, BatchNorm, etc.) the full forward
          // pass is performed in TypeScript; the VM module acts as a pass-through.
          code = this._encodePassthrough();
      }

      const bytecode = BytecodeLoader.createModuleWithExports(code, exportTable);
      await this._vm.loadModule(name, bytecode);
      moduleNames.push(name);
    }

    return moduleNames;
  }

  // ── private bytecode encoders ─────────────────────────────────────────────

  /**
   * Compute the absolute byte offset where the code section starts in the
   * final buffer produced by BytecodeLoader.createModuleWithExports.
   *
   * Data segment layout:
   *   uint16  – number of exports
   *   per export: uint8 name-length + N×uint8 name + uint16 address
   */
  private _computeCodeStart(exportTable: Array<{ name: string; address: number }>): number {
    const dataSize = 2 + exportTable.reduce((acc, e) => acc + 1 + e.name.length + 2, 0);
    return NNVMBridge._HEADER_SIZE + dataSize;
  }

  /** Encode a 32-bit IEEE 754 float as four little-endian bytes. */
  private _floatLE(value: number): number[] {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    return Array.from(new Uint8Array(buf));
  }

  /**
   * ReLU: max(0, x)
   *
   * Stack machine algorithm:
   *   LOAD x from memory[0]
   *   DUP               ; [x, x]
   *   PUSH_INT 0        ; [x, x, 0]
   *   GT                ; [x, (x>0)?1:0]
   *   JZ  zero_case     ; pop flag; if x≤0 jump
   *   JMP end           ; x>0 path: x already on stack
   * zero_case:
   *   POP               ; discard x
   *   PUSH_INT 0        ; result is 0
   * end:
   *   HALT
   */
  private _encodeReLU(codeStart: number): number[] {
    // Byte offsets within the code section:
    //  0  PUSH_INT 0 (2 bytes)
    //  2  LOAD       (1 byte)
    //  3  DUP        (1 byte)
    //  4  PUSH_INT 0 (2 bytes)
    //  6  GT         (1 byte)
    //  7  JZ <abs>   (3 bytes)
    // 10  JMP <abs>  (3 bytes)
    // 13  POP        (1 byte)  ← zero_case
    // 14  PUSH_INT 0 (2 bytes)
    // 16  HALT               ← end
    const zeroCase = codeStart + 13;
    const end = codeStart + 16;
    const zeroLo = zeroCase & 0xff;
    const zeroHi = (zeroCase >> 8) & 0xff;
    const endLo = end & 0xff;
    const endHi = (end >> 8) & 0xff;

    return [
      0x03, 0x00, // PUSH_INT 0  (memory address 0)
      0x40, // LOAD
      0x50, // DUP
      0x03, 0x00, // PUSH_INT 0  (threshold)
      0x22, // GT
      0x31, zeroLo, zeroHi, // JZ zero_case
      0x30, endLo, endHi, // JMP end
      0x02, // POP           (zero_case)
      0x03, 0x00, // PUSH_INT 0
      0xff, // HALT          (end)
    ];
  }

  /**
   * Hard-Tanh: clamp(x, −1, 1)
   *
   * Stack machine algorithm:
   *   LOAD x from memory[0]
   *   DUP               ; [x, x]
   *   PUSH -1.0         ; [x, x, -1]
   *   LT                ; [x, (x<-1)?1:0]
   *   JZ  not_below     ; not below -1
   *   POP               ; discard x
   *   PUSH -1.0         ; result = -1
   *   JMP end
   * not_below:
   *   DUP               ; [x, x]
   *   PUSH 1.0          ; [x, x, 1]
   *   GT                ; [x, (x>1)?1:0]
   *   JZ  end           ; x ≤ 1: x is already the result
   *   POP               ; discard x
   *   PUSH 1.0          ; result = 1
   * end:
   *   HALT
   */
  private _encodeTanh(codeStart: number): number[] {
    // Byte offsets within the code section:
    //  0  PUSH_INT 0    (2 bytes)
    //  2  LOAD          (1 byte)
    //  3  DUP           (1 byte)
    //  4  PUSH -1.0f    (5 bytes)
    //  9  LT            (1 byte)
    // 10  JZ not_below  (3 bytes)
    // 13  POP           (1 byte)
    // 14  PUSH -1.0f    (5 bytes)
    // 19  JMP end       (3 bytes)
    // 22  DUP           (1 byte)  ← not_below
    // 23  PUSH 1.0f     (5 bytes)
    // 28  GT            (1 byte)
    // 29  JZ end        (3 bytes)
    // 32  POP           (1 byte)
    // 33  PUSH 1.0f     (5 bytes)
    // 38  HALT                    ← end
    const notBelow = codeStart + 22;
    const end = codeStart + 38;
    const notBelowLo = notBelow & 0xff;
    const notBelowHi = (notBelow >> 8) & 0xff;
    const endLo = end & 0xff;
    const endHi = (end >> 8) & 0xff;
    const neg1 = this._floatLE(-1.0);
    const pos1 = this._floatLE(1.0);

    return [
      0x03, 0x00, // PUSH_INT 0
      0x40, // LOAD
      0x50, // DUP
      0x01, ...neg1, // PUSH -1.0
      0x21, // LT
      0x31, notBelowLo, notBelowHi, // JZ not_below
      0x02, // POP
      0x01, ...neg1, // PUSH -1.0
      0x30, endLo, endHi, // JMP end
      0x50, // DUP               (not_below)
      0x01, ...pos1, // PUSH 1.0
      0x22, // GT
      0x31, endLo, endHi, // JZ end
      0x02, // POP
      0x01, ...pos1, // PUSH 1.0
      0xff, // HALT              (end)
    ];
  }

  /**
   * Hard-Sigmoid: clamp(0, 1, x/4 + 0.5)
   *
   * Stack machine algorithm:
   *   LOAD x from memory[0]
   *   PUSH 0.25         ; [x, 0.25]
   *   MUL               ; [x/4]
   *   PUSH 0.5          ; [x/4, 0.5]
   *   ADD               ; [y = x/4 + 0.5]
   *   DUP               ; [y, y]
   *   PUSH_INT 0        ; [y, y, 0]
   *   GT                ; [y, (y>0)?1:0]
   *   JNZ check_upper   ; y > 0 → check upper bound
   *   POP               ; y ≤ 0: discard y
   *   PUSH_INT 0        ; result = 0
   *   JMP end
   * check_upper:
   *   DUP               ; [y, y]
   *   PUSH 1.0          ; [y, y, 1]
   *   GT                ; [y, (y>1)?1:0]
   *   JZ  end           ; y ≤ 1: y is already the result
   *   POP               ; discard y
   *   PUSH 1.0          ; result = 1
   * end:
   *   HALT
   */
  private _encodeSigmoid(codeStart: number): number[] {
    // Byte offsets within the code section:
    //  0  PUSH_INT 0      (2 bytes)
    //  2  LOAD            (1 byte)
    //  3  PUSH 0.25f      (5 bytes)
    //  8  MUL             (1 byte)
    //  9  PUSH 0.5f       (5 bytes)
    // 14  ADD             (1 byte)
    // 15  DUP             (1 byte)
    // 16  PUSH_INT 0      (2 bytes)
    // 18  GT              (1 byte)
    // 19  JNZ check_upper (3 bytes)
    // 22  POP             (1 byte)
    // 23  PUSH_INT 0      (2 bytes)
    // 25  JMP end         (3 bytes)
    // 28  DUP             (1 byte)  ← check_upper
    // 29  PUSH 1.0f       (5 bytes)
    // 34  GT              (1 byte)
    // 35  JZ  end         (3 bytes)
    // 38  POP             (1 byte)
    // 39  PUSH 1.0f       (5 bytes)
    // 44  HALT                      ← end
    const checkUpper = codeStart + 28;
    const end = codeStart + 44;
    const checkUpperLo = checkUpper & 0xff;
    const checkUpperHi = (checkUpper >> 8) & 0xff;
    const endLo = end & 0xff;
    const endHi = (end >> 8) & 0xff;
    const quarter = this._floatLE(0.25);
    const half = this._floatLE(0.5);
    const one = this._floatLE(1.0);

    return [
      0x03, 0x00, // PUSH_INT 0
      0x40, // LOAD
      0x01, ...quarter, // PUSH 0.25
      0x12, // MUL
      0x01, ...half, // PUSH 0.5
      0x10, // ADD
      0x50, // DUP
      0x03, 0x00, // PUSH_INT 0
      0x22, // GT
      0x32, checkUpperLo, checkUpperHi, // JNZ check_upper
      0x02, // POP
      0x03, 0x00, // PUSH_INT 0
      0x30, endLo, endHi, // JMP end
      0x50, // DUP               (check_upper)
      0x01, ...one, // PUSH 1.0
      0x22, // GT
      0x31, endLo, endHi, // JZ end
      0x02, // POP
      0x01, ...one, // PUSH 1.0
      0xff, // HALT              (end)
    ];
  }

  /**
   * Pass-through: load scalar from memory[0] and return it unchanged.
   * Used for parametric layers (Linear, BatchNorm, Conv1d, …) where the
   * full forward pass is computed in TypeScript.
   */
  private _encodePassthrough(): number[] {
    return [
      0x03, 0x00, // PUSH_INT 0  (memory address)
      0x40, // LOAD
      0xff, // HALT
    ];
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
