/**
 * Bytecode Loader and Validator
 * 
 * Handles loading, validation, and verification of .dis bytecode modules
 */

export interface BytecodeHeader {
  magic: number;
  version: number;
  entryPoint: number;
  dataSize: number;
  codeSize: number;
}

export class BytecodeLoader {
  private static readonly MAGIC_NUMBER = 0x44495356; // "DISV" in hex

  /**
   * Load bytecode from a buffer
   */
  static load(buffer: ArrayBuffer): Uint8Array {
    const view = new DataView(buffer);
    const header = this.parseHeader(view);
    
    if (!this.validateHeader(header)) {
      throw new Error('Invalid bytecode header');
    }

    return new Uint8Array(buffer);
  }

  /**
   * Parse bytecode header
   */
  private static parseHeader(view: DataView): BytecodeHeader {
    return {
      magic: view.getUint32(0, true),
      version: view.getUint32(4, true),
      entryPoint: view.getUint32(8, true),
      dataSize: view.getUint32(12, true),
      codeSize: view.getUint32(16, true),
    };
  }

  /**
   * Validate bytecode header
   */
  private static validateHeader(header: BytecodeHeader): boolean {
    if (header.magic !== this.MAGIC_NUMBER) {
      return false;
    }
    
    if (header.version > 1) {
      return false;
    }

    return true;
  }

  /**
   * Create a simple bytecode module
   */
  static createModule(code: number[]): Uint8Array {
    const headerSize = 20;
    const codeSize = code.length;
    const totalSize = headerSize + codeSize;
    
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    
    // Write header
    view.setUint32(0, this.MAGIC_NUMBER, true);
    view.setUint32(4, 1, true); // version
    view.setUint32(8, headerSize, true); // entry point
    view.setUint32(12, 0, true); // data size
    view.setUint32(16, codeSize, true); // code size
    
    // Write code
    const array = new Uint8Array(buffer);
    for (let i = 0; i < code.length; i++) {
      array[headerSize + i] = code[i];
    }
    
    return array;
  }

  /**
   * Verify bytecode integrity
   */
  static verify(bytecode: Uint8Array): boolean {
    if (bytecode.length < 20) {
      return false;
    }

    const view = new DataView(bytecode.buffer);
    const header = this.parseHeader(view);
    
    return this.validateHeader(header);
  }

  /**
   * Extract bytecode metadata
   */
  static extractMetadata(bytecode: Uint8Array): BytecodeHeader {
    if (!this.verify(bytecode)) {
      throw new Error('Invalid bytecode');
    }

    const view = new DataView(bytecode.buffer);
    return this.parseHeader(view);
  }
}
