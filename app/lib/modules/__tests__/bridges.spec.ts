import { describe, expect, it, beforeEach } from 'vitest';
import { createVM, type InfernoVM } from '~/lib/modules/vm/diy.dis';
import { VMRuntime } from '~/lib/modules/vm/vm-runtime';
import { NNVMBridge, MLVMBridge } from '~/lib/modules/bridges';
import { nn } from '~/lib/modules/nn/nn.b';
import { Linear, ReLU, Tanh, Sigmoid } from '~/lib/modules/nn/nn-modules';
import { buildModel, ModelBuilder } from '~/lib/modules/nn/model-builder';
import type { Tensor } from '~/lib/modules/ml/ml.m';

// ── NNVMBridge ──────────────────────────────────────────────────────────────

describe('NNVMBridge', () => {
  let vm: InfernoVM;
  let bridge: NNVMBridge;

  beforeEach(() => {
    vm = createVM();
    bridge = new NNVMBridge(vm);
  });

  describe('compileModel', () => {
    it('should compile a Sequential model and return one module name per layer', async () => {
      const model = nn()
        .add(new Linear({ inputSize: 3, outputSize: 2, bias: true }))
        .add(new ReLU());
      const names = await bridge.compileModel('test', model);

      expect(names).toHaveLength(2);
      expect(names[0]).toBe('test_layer_0');
      expect(names[1]).toBe('test_layer_1');
    });

    it('should load modules into the VM', async () => {
      const model = nn().add(new ReLU()).add(new Tanh());
      await bridge.compileModel('mymodel', model);

      const loaded = vm.getModules();
      expect(loaded).toContain('mymodel_layer_0');
      expect(loaded).toContain('mymodel_layer_1');
    });

    it('should produce different module names for different model names', async () => {
      const modelA = nn().add(new ReLU());
      const modelB = nn().add(new Sigmoid());

      await bridge.compileModel('a', modelA);
      await bridge.compileModel('b', modelB);

      expect(vm.getModules()).toContain('a_layer_0');
      expect(vm.getModules()).toContain('b_layer_0');
    });
  });

  describe('executeModel – ReLU bytecode', () => {
    it('should return 0 for a negative input', async () => {
      const model = nn().add(new ReLU());
      const names = await bridge.compileModel('relu-test', model);

      // Pass -3 as the scalar input to layer 0
      const result = await vm.execute(names[0], 'forward', [-3]);
      expect(result).toBe(0);
    });

    it('should return the input unchanged for a positive value', async () => {
      const model = nn().add(new ReLU());
      const names = await bridge.compileModel('relu-pos', model);

      const result = await vm.execute(names[0], 'forward', [5]);
      expect(result).toBe(5);
    });

    it('should return 0 for input exactly 0', async () => {
      const model = nn().add(new ReLU());
      const names = await bridge.compileModel('relu-zero', model);

      const result = await vm.execute(names[0], 'forward', [0]);
      expect(result).toBe(0);
    });
  });

  describe('executeModel – hard-Tanh bytecode', () => {
    it('should clamp positive values above 1 to 1', async () => {
      const model = nn().add(new Tanh());
      const names = await bridge.compileModel('tanh-test', model);

      const result = await vm.execute(names[0], 'forward', [3]);
      expect(result).toBeCloseTo(1.0, 5);
    });

    it('should clamp negative values below -1 to -1', async () => {
      const model = nn().add(new Tanh());
      const names = await bridge.compileModel('tanh-neg', model);

      const result = await vm.execute(names[0], 'forward', [-5]);
      expect(result).toBeCloseTo(-1.0, 5);
    });

    it('should pass through values within [-1, 1] unchanged', async () => {
      const model = nn().add(new Tanh());
      const names = await bridge.compileModel('tanh-pass', model);

      const result = await vm.execute(names[0], 'forward', [0.5]);
      expect(result).toBeCloseTo(0.5, 5);
    });
  });

  describe('executeModel – hard-Sigmoid bytecode', () => {
    it('should return 0 for large negative inputs', async () => {
      const model = nn().add(new Sigmoid());
      const names = await bridge.compileModel('sig-test', model);

      const result = await vm.execute(names[0], 'forward', [-10]);
      expect(result).toBe(0);
    });

    it('should return 1 for large positive inputs', async () => {
      const model = nn().add(new Sigmoid());
      const names = await bridge.compileModel('sig-pos', model);

      const result = await vm.execute(names[0], 'forward', [10]);
      expect(result).toBeCloseTo(1.0, 5);
    });

    it('should return 0.5 for input 0', async () => {
      const model = nn().add(new Sigmoid());
      const names = await bridge.compileModel('sig-zero', model);

      // hard-sigmoid(0) = 0/4 + 0.5 = 0.5
      const result = await vm.execute(names[0], 'forward', [0]);
      expect(result).toBeCloseTo(0.5, 5);
    });
  });

  describe('executeModel – pass-through for parametric layers', () => {
    it('should pass the scalar input through a Linear layer module', async () => {
      const model = nn().add(new Linear({ inputSize: 1, outputSize: 1, bias: false }));
      const names = await bridge.compileModel('linear-pt', model);

      const result = await vm.execute(names[0], 'forward', [7]);
      expect(result).toBe(7);
    });
  });
});

// ── MLVMBridge ───────────────────────────────────────────────────────────────

describe('MLVMBridge', () => {
  let vmRuntime: VMRuntime;
  let bridge: MLVMBridge;

  beforeEach(() => {
    vmRuntime = new VMRuntime();
    bridge = new MLVMBridge(vmRuntime);
  });

  it('should write and read back a 1-D tensor without data loss', () => {
    const proc = vmRuntime.createProcess(1024);
    const tensor: Tensor = {
      shape: [4],
      data: new Float32Array([1.5, 2.5, -3.14, 0.0]),
      dtype: 'float32',
    };

    const segment = bridge.writeTensor(proc, tensor, 0);
    const recovered = bridge.readTensor(proc, segment);

    expect(recovered.shape).toEqual(tensor.shape);
    expect(recovered.dtype).toBe('float32');

    const orig = tensor.data as Float32Array;
    const res = recovered.data as Float32Array;

    for (let i = 0; i < orig.length; i++) {
      expect(res[i]).toBeCloseTo(orig[i], 5);
    }
  });

  it('should write and read back a 2-D tensor', () => {
    const proc = vmRuntime.createProcess(4096);
    const tensor: Tensor = {
      shape: [2, 3],
      data: new Float32Array([1, 2, 3, 4, 5, 6]),
      dtype: 'float32',
    };

    const segment = bridge.writeTensor(proc, tensor, 0);
    const recovered = bridge.readTensor(proc, segment);

    expect(recovered.shape).toEqual([2, 3]);
    expect(Array.from(recovered.data as Float32Array)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('should return a segment descriptor with correct address and length', () => {
    const proc = vmRuntime.createProcess(1024);
    const tensor: Tensor = {
      shape: [2],
      data: new Float32Array([1.0, 2.0]),
      dtype: 'float32',
    };

    const segment = bridge.writeTensor(proc, tensor, 16);

    expect(segment.address).toBe(16);

    // 4 bytes ndim + 1*4 bytes shape + 2*4 bytes data
    expect(segment.length).toBe(4 + 4 + 8);
    expect(segment.dtype).toBe('float32');
  });
});

// ── Adam / AdamW optimiser ───────────────────────────────────────────────────

describe('ModelBuilder – optimisers', () => {
  const mse = (output: Tensor, target: Tensor): Tensor => {
    const o = output.data as Float32Array;
    const t = target.data as Float32Array;
    let sum = 0;

    for (let i = 0; i < o.length; i++) {
      sum += (o[i] - t[i]) ** 2;
    }

    return { shape: [1], data: new Float32Array([sum / o.length]), dtype: 'float32' };
  };

  const makeData = (): { inputs: Tensor[]; targets: Tensor[] } => {
    const inputs: Tensor[] = [];
    const targets: Tensor[] = [];

    for (let i = 0; i < 4; i++) {
      inputs.push({ shape: [1, 2], data: new Float32Array([i, i + 1]), dtype: 'float32' });
      targets.push({ shape: [1, 1], data: new Float32Array([i * 2]), dtype: 'float32' });
    }

    return { inputs, targets };
  };

  it('should train with SGD and reduce loss over epochs', () => {
    const builder = buildModel('sgd-test', [2]).linear(2, 1);
    const { inputs, targets } = makeData();

    const losses = builder.train(inputs, targets, mse, 0.01, 3, 'sgd');

    expect(losses).toHaveLength(3);

    // Loss should be finite numbers
    for (const l of losses) {
      expect(isFinite(l)).toBe(true);
    }
  });

  it('should train with Adam and return one loss per epoch', () => {
    const builder = buildModel('adam-test', [2]).linear(2, 1);
    const { inputs, targets } = makeData();

    const losses = builder.train(inputs, targets, mse, 0.001, 3, 'adam');

    expect(losses).toHaveLength(3);

    for (const l of losses) {
      expect(isFinite(l)).toBe(true);
    }
  });

  it('should train with AdamW and return one loss per epoch', () => {
    const builder = buildModel('adamw-test', [2]).linear(2, 1);
    const { inputs, targets } = makeData();

    const losses = builder.train(inputs, targets, mse, 0.001, 3, 'adamw', { weightDecay: 0.01 });

    expect(losses).toHaveLength(3);

    for (const l of losses) {
      expect(isFinite(l)).toBe(true);
    }
  });

  it('Adam and AdamW produce different parameter updates', () => {
    const { inputs, targets } = makeData();

    const builderAdam = buildModel('adam-cmp', [2]).linear(2, 1);
    const builderAdamW = buildModel('adamw-cmp', [2]).linear(2, 1);

    // Copy identical initial weights so both start from the same point
    const paramsAdam = builderAdam.parameters();
    const paramsAdamW = builderAdamW.parameters();

    for (let p = 0; p < paramsAdam.length; p++) {
      const src = paramsAdam[p].data as Float32Array;
      const dst = paramsAdamW[p].data as Float32Array;

      for (let k = 0; k < src.length; k++) {
        dst[k] = src[k];
      }
    }

    builderAdam.train(inputs, targets, mse, 0.001, 2, 'adam');
    builderAdamW.train(inputs, targets, mse, 0.001, 2, 'adamw', { weightDecay: 0.1 });

    // With non-trivial weight decay the AdamW parameters should differ
    const afterAdam = builderAdam.parameters();
    const afterAdamW = builderAdamW.parameters();
    let differ = false;

    outer: for (let p = 0; p < afterAdam.length; p++) {
      const a = afterAdam[p].data as Float32Array;
      const w = afterAdamW[p].data as Float32Array;

      for (let k = 0; k < a.length; k++) {
        if (Math.abs(a[k] - w[k]) > 1e-9) {
          differ = true;
          break outer;
        }
      }
    }

    expect(differ).toBe(true);
  });
});

// ── Model save / load ────────────────────────────────────────────────────────

describe('ModelBuilder – save / load', () => {
  it('should round-trip architecture and weights via save / load', () => {
    const builder = new ModelBuilder('save-test', [3]);
    builder.linear(3, 2).relu().linear(2, 1);

    // Fix weights to known values for deterministic comparison
    const params = builder.parameters();
    params.forEach((p, i) => {
      (p.data as Float32Array).fill(i + 1);
    });

    const json = builder.save();
    const parsed = JSON.parse(json);

    expect(parsed).toHaveProperty('architecture');
    expect(parsed).toHaveProperty('weights');
    expect(parsed.architecture.name).toBe('save-test');

    const restored = ModelBuilder.load(json);
    const restoredParams = restored.parameters();

    expect(restoredParams.length).toBe(params.length);

    for (let p = 0; p < params.length; p++) {
      const orig = params[p].data as Float32Array;
      const rest = restoredParams[p].data as Float32Array;

      for (let k = 0; k < orig.length; k++) {
        expect(rest[k]).toBeCloseTo(orig[k], 6);
      }
    }
  });

  it('toJSON / fromJSON restores architecture but not weights', () => {
    const builder = new ModelBuilder('arch-only', [2]);
    builder.linear(2, 1);

    // Set known weights
    (builder.parameters()[0].data as Float32Array).fill(99.0);

    const json = builder.toJSON();
    const restored = ModelBuilder.fromJSON(json);

    // Architecture restored
    expect(restored.getArchitecture().name).toBe('arch-only');

    // Weights are freshly initialised (not 99)
    const w = restored.parameters()[0].data as Float32Array;
    const allNinetynine = Array.from(w).every((v: number) => v === 99.0);
    expect(allNinetynine).toBe(false);
  });

  it('load should restore weights to match saved values', () => {
    const builder = new ModelBuilder('weight-test', [2]);
    builder.linear(2, 1, false); // no bias so only one parameter tensor

    const params = builder.parameters();
    (params[0].data as Float32Array).set([3.14, 2.71]);

    const saved = builder.save();
    const loaded = ModelBuilder.load(saved);
    const loadedParams = loaded.parameters();

    expect((loadedParams[0].data as Float32Array)[0]).toBeCloseTo(3.14, 5);
    expect((loadedParams[0].data as Float32Array)[1]).toBeCloseTo(2.71, 5);
  });

  it('save output is valid JSON', () => {
    const builder = new ModelBuilder('json-test', [4]);
    builder.linear(4, 2).relu();

    const json = builder.save();
    expect(() => JSON.parse(json)).not.toThrow();
  });
});
