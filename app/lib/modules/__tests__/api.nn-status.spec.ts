import { describe, expect, it } from 'vitest';
import { loader } from '~/routes/api.nn-status';
import { BytecodeLoader } from '~/lib/modules/vm/bytecode-loader';
import { getGlobalMLModule } from '~/lib/modules/ml/ml.m';
import { getGlobalVM } from '~/lib/modules/vm/diy.dis';

describe('/api/nn-status loader', () => {
  it('should return current VM and ML status details', async () => {
    const vm = getGlobalVM();
    const beforeVmStats = vm.getStats();
    const ml = getGlobalMLModule();
    const beforeMlStats = ml.getStats();
    const moduleName = `status-module-${Date.now()}`;
    const modelName = `status-model-${Date.now()}`;

    await vm.loadModule(moduleName, BytecodeLoader.createModule([0xff]));
    ml.loadModel(modelName, {
      vocabSize: 16,
      hiddenSize: 8,
      numLayers: 1,
      numHeads: 1,
      maxSequenceLength: 32,
    });

    const response = await loader({
      request: new Request('http://localhost/api/nn-status'),
      context: {} as never,
      params: {},
    });
    const payload = await response.json();

    expect(payload.system.name).toBe('nn.llmbol.dis.vm');
    expect(payload.vm.loadedModules).toBeGreaterThanOrEqual(beforeVmStats.loadedModules + 1);
    expect(payload.vm.processes).toBeGreaterThanOrEqual(beforeVmStats.processes + 1);
    expect(payload.vm.processList).toEqual(
      expect.arrayContaining([expect.objectContaining({ state: expect.any(String) })]),
    );
    expect(payload.ml.loadedModels).toBeGreaterThanOrEqual(beforeMlStats.loadedModels + 1);
    expect(payload.ml.modelNames).toContain(modelName);
    expect(payload.ml.status).toBe('active');
    expect(typeof payload.timestamp).toBe('string');
  });
});
