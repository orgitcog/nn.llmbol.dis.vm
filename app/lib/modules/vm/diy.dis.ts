/**
 * Inferno-inspired VM Deployment Engine
 *
 * This module implements a VM deployment system inspired by Inferno OS,
 * with support for bytecode execution (.dis files) and dynamic module loading.
 */

import { VMRuntime } from '~/lib/modules/vm/vm-runtime';

export interface DisModule {
  name: string;
  version: string;
  bytecode: Uint8Array;
  dependencies: string[];
  exports: Record<string, (...args: unknown[]) => unknown>;
  processId?: string;
}

export interface VMConfig {
  maxMemory: number;
  maxThreads: number;
  enableDistributed: boolean;
  securityLevel: 'strict' | 'relaxed';
}

export class InfernoVM {
  private _modules: Map<string, DisModule>;
  private _config: VMConfig;
  private _runtime: VMRuntime;

  constructor(config: Partial<VMConfig> = {}) {
    this._modules = new Map();
    this._config = {
      maxMemory: config.maxMemory ?? 1024 * 1024 * 512, // 512 MB default
      maxThreads: config.maxThreads ?? 4,
      enableDistributed: config.enableDistributed ?? false,
      securityLevel: config.securityLevel ?? 'strict',
    };
    this._runtime = new VMRuntime();
  }

  /**
   * Load a .dis module into the VM.
   *
   * Exported functions are parsed from the data segment and registered as
   * callable wrappers that execute the bytecode starting at each function's
   * address.
   */
  async loadModule(name: string, bytecode: Uint8Array): Promise<DisModule> {
    const exportAddresses = this._parseExports(bytecode);
    const proc = this._runtime.createProcess();

    const module: DisModule = {
      name,
      version: '1.0.0',
      bytecode,
      dependencies: [],
      exports: {},
      processId: proc.id,
    };

    for (const [exportName, address] of Object.entries(exportAddresses)) {
      module.exports[exportName] = (...args: unknown[]) => {
        /*
         * Write numeric arguments into the process's flat memory so that
         * bytecode can retrieve them via LOAD.  Argument i is written as a
         * little-endian float32 at memory offset i * 4.
         */
        if (args.length > 0) {
          const memView = new DataView(proc.memory.buffer);

          for (let i = 0; i < args.length; i++) {
            if (typeof args[i] === 'number') {
              memView.setFloat32(i * 4, args[i] as number, true);
            }
          }
        }

        proc.registers.set('__startPC', address);

        return this._runtime.execute(proc.id, bytecode);
      };
    }

    this._modules.set(name, module);

    return module;
  }

  /**
   * Execute an exported function in a loaded module.
   * Throws if the module or entry point is not found, or if a required
   * dependency has not yet been loaded.
   */
  async execute(moduleName: string, entryPoint: string, args: unknown[] = []): Promise<unknown> {
    const module = this._modules.get(moduleName);

    if (!module) {
      throw new Error(`Module ${moduleName} not found`);
    }

    for (const dep of module.dependencies) {
      if (!this._modules.has(dep)) {
        throw new Error(`Dependency ${dep} required by ${moduleName} is not loaded`);
      }
    }

    const fn = module.exports[entryPoint];

    if (!fn) {
      throw new Error(`Entry point ${entryPoint} not found in module ${moduleName}`);
    }

    return fn(...args);
  }

  /**
   * Return the names of all currently loaded modules.
   */
  getModules(): string[] {
    return Array.from(this._modules.keys());
  }

  /**
   * Unload a module by name.  Returns true if it was found and removed.
   */
  unloadModule(name: string): boolean {
    return this._modules.delete(name);
  }

  /**
   * Return runtime statistics including loaded module count, heap usage, and
   * the active VM configuration.
   */
  getStats() {
    const runtimeProcess =
      typeof globalThis !== 'undefined' ? (globalThis as unknown as Record<string, unknown>).process : undefined;
    const memUsage =
      runtimeProcess !== null &&
      runtimeProcess !== undefined &&
      typeof (runtimeProcess as Record<string, unknown>).memoryUsage === 'function'
        ? (runtimeProcess as { memoryUsage: () => { heapUsed: number } }).memoryUsage()
        : { heapUsed: 0 };
    const processes = this._runtime.getAllProcesses();

    return {
      loadedModules: this._modules.size,
      memoryUsage: memUsage,
      processes: processes.length,
      processList: processes.map((proc) => ({ id: proc.id, state: proc.state })),
      config: this._config,
    };
  }

  // ── private helpers ────────────────────────────────────────────────────────

  /**
   * Parse the export table from the data segment of a validated bytecode buffer.
   * Returns a map of export name → absolute bytecode offset.
   */
  private _parseExports(bytecode: Uint8Array): Record<string, number> {
    const result: Record<string, number> = {};
    const HEADER_SIZE = 24;

    if (bytecode.length < HEADER_SIZE + 2) {
      return result;
    }

    const view = new DataView(bytecode.buffer, bytecode.byteOffset, bytecode.byteLength);
    const numExports = view.getUint16(HEADER_SIZE, true);
    let offset = HEADER_SIZE + 2;

    for (let i = 0; i < numExports; i++) {
      if (offset >= bytecode.length) {
        break;
      }

      const nameLen = bytecode[offset++]!;

      if (offset + nameLen + 2 > bytecode.length) {
        break;
      }

      let exportName = '';

      for (let j = 0; j < nameLen; j++) {
        exportName += String.fromCharCode(bytecode[offset++]!);
      }

      const address = view.getUint16(offset, true);
      offset += 2;
      result[exportName] = address;
    }

    return result;
  }
}

/**
 * Create a new Inferno VM instance
 */
export function createVM(config?: Partial<VMConfig>): InfernoVM {
  return new InfernoVM(config);
}

/**
 * Global VM instance (singleton pattern)
 */
let globalVM: InfernoVM | null = null;

export function getGlobalVM(): InfernoVM {
  if (!globalVM) {
    globalVM = createVM();
  }

  return globalVM;
}
