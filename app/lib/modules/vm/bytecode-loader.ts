/**
 * Bytecode Loader and Validator
 *
 * Handles loading, validation, and verification of .dis bytecode modules.
 * Header layout (24 bytes):
 *   offset  0 – magic      (uint32 LE)
 *   offset  4 – version    (uint32 LE, valid range 1–2)
 *   offset  8 – entryPoint (uint32 LE)
 *   offset 12 – dataSize   (uint32 LE)
 *   offset 16 – codeSize   (uint32 LE)
 *   offset 20 – checksum   (uint32 LE, XOR of all bytes in code section)
 */

export interface BytecodeHeader {
  magic: number;
  version: number;
  entryPoint: number;
  dataSize: number;
  codeSize: number;
  checksum: number;
}

export class BytecodeLoader {
  static readonly MAGIC_NUMBER = 0x44495356; // "DISV" in hex
  private static readonly _headerSize = 24;

  // ── private helpers ────────────────────────────────────────────────────────

  /** Parse the 24-byte header from a DataView whose offset 0 is the bytecode start. */
  private static _parseHeader(view: DataView): BytecodeHeader {
    return {
      magic: view.getUint32(0, true),
      version: view.getUint32(4, true),
      entryPoint: view.getUint32(8, true),
      dataSize: view.getUint32(12, true),
      codeSize: view.getUint32(16, true),
      checksum: view.getUint32(20, true),
    };
  }

  /** Return true only when magic and version (1–2) are valid. */
  private static _validateHeader(header: BytecodeHeader): boolean {
    if (header.magic !== this.MAGIC_NUMBER) {
      return false;
    }

    if (header.version < 1 || header.version > 2) {
      return false;
    }

    return true;
  }

  /** Compute the XOR checksum of all bytes in the code array. */
  private static _computeChecksum(code: number[]): number {
    let checksum = 0;

    for (const byte of code) {
      checksum ^= byte;
    }

    return checksum;
  }

  // ── public API ─────────────────────────────────────────────────────────────

  /**
   * Load bytecode from a raw ArrayBuffer.
   * Validates the header and checksum before returning.
   */
  static load(buffer: ArrayBuffer): Uint8Array {
    const bytes = new Uint8Array(buffer);

    if (!this.verify(bytes)) {
      throw new Error('Invalid bytecode header');
    }

    return bytes;
  }

  /**
   * Verify bytecode integrity: checks minimum length, header validity, and checksum.
   */
  static verify(bytecode: Uint8Array): boolean {
    if (bytecode.length < this._headerSize) {
      return false;
    }

    const view = new DataView(bytecode.buffer, bytecode.byteOffset, bytecode.byteLength);
    const header = this._parseHeader(view);

    if (!this._validateHeader(header)) {
      return false;
    }

    const codeStart = this._headerSize + header.dataSize;

    if (bytecode.length < codeStart + header.codeSize) {
      return false;
    }

    let computed = 0;

    for (let i = codeStart; i < codeStart + header.codeSize; i++) {
      computed ^= bytecode[i]!;
    }

    return computed === header.checksum;
  }

  /**
   * Validate bytecode, throwing an error if invalid.
   * Called before loading a module into the VM.
   */
  static validate(bytecode: Uint8Array): void {
    if (!this.verify(bytecode)) {
      throw new Error('Invalid bytecode');
    }
  }

  /**
   * Extract and return the bytecode header metadata.
   * Throws if the bytecode fails verification.
   */
  static extractMetadata(bytecode: Uint8Array): BytecodeHeader {
    if (!this.verify(bytecode)) {
      throw new Error('Invalid bytecode');
    }

    const view = new DataView(bytecode.buffer, bytecode.byteOffset, bytecode.byteLength);

    return this._parseHeader(view);
  }

  /**
   * Create a minimal bytecode module with no exports and an empty data segment.
   * The 24-byte header includes a checksum of the code section.
   */
  static createModule(code: number[]): Uint8Array {
    const H = this._headerSize;
    const dataSize = 0;
    const codeSize = code.length;
    const entryPoint = H + dataSize;
    const checksum = this._computeChecksum(code);
    const buffer = new ArrayBuffer(H + codeSize);
    const view = new DataView(buffer);

    view.setUint32(0, this.MAGIC_NUMBER, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, entryPoint, true);
    view.setUint32(12, dataSize, true);
    view.setUint32(16, codeSize, true);
    view.setUint32(20, checksum, true);

    const array = new Uint8Array(buffer);

    for (let i = 0; i < codeSize; i++) {
      array[H + i] = code[i]!;
    }

    return array;
  }

  /**
   * Create a bytecode module with an export table embedded in the data segment.
   *
   * Data segment layout:
   *   uint16  – number of exports
   *   per export: uint8 name-length, N×uint8 ASCII name, uint16 function address
   *
   * The function addresses in `exportTable` must be absolute byte offsets into
   * the final bytecode buffer (i.e. relative to the very start of the returned
   * Uint8Array).  Code starts at offset `headerSize + dataSize`, so a function
   * at the beginning of the code section has address `headerSize + dataSize`.
   */
  static createModuleWithExports(code: number[], exportTable: Array<{ name: string; address: number }>): Uint8Array {
    const H = this._headerSize;

    // Build data segment
    const dataBytes: number[] = [];
    dataBytes.push(exportTable.length & 0xff);
    dataBytes.push((exportTable.length >> 8) & 0xff);

    for (const exp of exportTable) {
      const nameBytes = Array.from(exp.name).map((c) => c.charCodeAt(0));
      dataBytes.push(nameBytes.length & 0xff);
      dataBytes.push(...nameBytes);
      dataBytes.push(exp.address & 0xff);
      dataBytes.push((exp.address >> 8) & 0xff);
    }

    const dataSize = dataBytes.length;
    const codeSize = code.length;
    const entryPoint = H + dataSize;
    const checksum = this._computeChecksum(code);
    const buffer = new ArrayBuffer(H + dataSize + codeSize);
    const view = new DataView(buffer);

    view.setUint32(0, this.MAGIC_NUMBER, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, entryPoint, true);
    view.setUint32(12, dataSize, true);
    view.setUint32(16, codeSize, true);
    view.setUint32(20, checksum, true);

    const array = new Uint8Array(buffer);

    for (let i = 0; i < dataSize; i++) {
      array[H + i] = dataBytes[i]!;
    }

    for (let i = 0; i < codeSize; i++) {
      array[H + dataSize + i] = code[i]!;
    }

    return array;
  }
}
