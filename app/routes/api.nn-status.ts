import type { LoaderFunctionArgs } from '@remix-run/cloudflare';
import { json } from '@remix-run/cloudflare';
import { SYSTEM_INFO } from '~/lib/modules/index';
import { getGlobalMLModule } from '~/lib/modules/ml/ml.m';
import { getGlobalVM } from '~/lib/modules/vm/diy.dis';

/**
 * Remix loader that returns JSON status for NN/VM/ML subsystems.
 */
export async function loader({ request: _request }: LoaderFunctionArgs) {
  const vm = getGlobalVM();
  const vmStats = vm.getStats();
  const ml = getGlobalMLModule();
  const mlStats = ml.getStats();

  return json({
    system: SYSTEM_INFO,
    vm: {
      loadedModules: vmStats.loadedModules,
      processes: vmStats.processes,
      processList: vmStats.processList,
    },
    ml: {
      loadedModels: mlStats.loadedModels,
      modelNames: mlStats.modelNames,
      status: mlStats.loadedModels > 0 ? 'active' : 'ready',
    },
    timestamp: new Date().toISOString(),
  });
}
