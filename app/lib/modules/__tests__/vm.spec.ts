import { describe, expect, it, beforeEach } from 'vitest';
import { InfernoVM, createVM, getGlobalVM } from '../vm/diy.dis';
import { BytecodeLoader } from '../vm/bytecode-loader';
import { runtime } from '../vm/vm-runtime';

describe('InfernoVM', () => {
  describe('VM Creation', () => {
    it('should create a VM with default config', () => {
      const vm = createVM();
      expect(vm).toBeDefined();
      expect(vm).toBeInstanceOf(InfernoVM);
    });

    it('should create a VM with custom config', () => {
      const vm = createVM({
        maxMemory: 1024 * 1024,
        maxThreads: 2,
        enableDistributed: true,
        securityLevel: 'relaxed',
      });
      
      const stats = vm.getStats();
      expect(stats.config.maxMemory).toBe(1024 * 1024);
      expect(stats.config.maxThreads).toBe(2);
      expect(stats.config.enableDistributed).toBe(true);
      expect(stats.config.securityLevel).toBe('relaxed');
    });

    it('should provide a global VM instance', () => {
      const vm1 = getGlobalVM();
      const vm2 = getGlobalVM();
      expect(vm1).toBe(vm2);
    });
  });

  describe('Module Loading', () => {
    let vm: InfernoVM;

    beforeEach(() => {
      vm = createVM();
    });

    it('should load a module', async () => {
      const bytecode = new Uint8Array([0x01, 0x02, 0x03]);
      const module = await vm.loadModule('test-module', bytecode);
      
      expect(module.name).toBe('test-module');
      expect(module.bytecode).toEqual(bytecode);
      expect(module.version).toBe('1.0.0');
    });

    it('should list loaded modules', async () => {
      const bytecode = new Uint8Array([0x01, 0x02, 0x03]);
      await vm.loadModule('module1', bytecode);
      await vm.loadModule('module2', bytecode);
      
      const modules = vm.getModules();
      expect(modules).toContain('module1');
      expect(modules).toContain('module2');
      expect(modules.length).toBe(2);
    });

    it('should unload a module', async () => {
      const bytecode = new Uint8Array([0x01, 0x02, 0x03]);
      await vm.loadModule('test-module', bytecode);
      
      const result = vm.unloadModule('test-module');
      expect(result).toBe(true);
      expect(vm.getModules()).not.toContain('test-module');
    });
  });

  describe('VM Statistics', () => {
    it('should provide stats', () => {
      const vm = createVM();
      const stats = vm.getStats();
      
      expect(stats).toHaveProperty('loadedModules');
      expect(stats).toHaveProperty('memoryUsage');
      expect(stats).toHaveProperty('config');
      expect(stats.loadedModules).toBe(0);
    });
  });
});

describe('BytecodeLoader', () => {
  describe('Module Creation', () => {
    it('should create a simple bytecode module', () => {
      const code = [0x00, 0x01, 0x02, 0xFF]; // NOP, PUSH, POP, HALT
      const bytecode = BytecodeLoader.createModule(code);
      
      expect(bytecode).toBeInstanceOf(Uint8Array);
      expect(bytecode.length).toBeGreaterThan(code.length);
    });

    it('should verify valid bytecode', () => {
      const code = [0x00, 0x01, 0xFF];
      const bytecode = BytecodeLoader.createModule(code);
      
      const isValid = BytecodeLoader.verify(bytecode);
      expect(isValid).toBe(true);
    });

    it('should reject invalid bytecode', () => {
      const invalidBytecode = new Uint8Array([0x00, 0x01]);
      const isValid = BytecodeLoader.verify(invalidBytecode);
      expect(isValid).toBe(false);
    });
  });

  describe('Metadata Extraction', () => {
    it('should extract bytecode metadata', () => {
      const code = [0x00, 0x01, 0xFF];
      const bytecode = BytecodeLoader.createModule(code);
      
      const metadata = BytecodeLoader.extractMetadata(bytecode);
      expect(metadata).toHaveProperty('magic');
      expect(metadata).toHaveProperty('version');
      expect(metadata).toHaveProperty('entryPoint');
      expect(metadata).toHaveProperty('codeSize');
      expect(metadata.codeSize).toBe(code.length);
    });
  });
});

describe('VMRuntime', () => {
  describe('Process Management', () => {
    it('should create a process', () => {
      const process = runtime.createProcess();
      
      expect(process).toHaveProperty('id');
      expect(process).toHaveProperty('state');
      expect(process.state).toBe('suspended');
    });

    it('should execute bytecode', () => {
      const process = runtime.createProcess();
      const bytecode = new Uint8Array([0x01, 0x05, 0xFF]); // PUSH 5, HALT
      
      const result = runtime.execute(process.id, bytecode);
      expect(result).toBeDefined();
    });

    it('should terminate a process', () => {
      const process = runtime.createProcess();
      const terminated = runtime.terminateProcess(process.id);
      
      expect(terminated).toBe(true);
    });

    it('should get process status', () => {
      const process = runtime.createProcess();
      const status = runtime.getProcessStatus(process.id);
      
      expect(status).toBeDefined();
      expect(status?.id).toBe(process.id);
    });
  });
});
