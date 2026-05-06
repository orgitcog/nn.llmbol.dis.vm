import type { LoaderFunctionArgs } from '@remix-run/cloudflare';
import { json } from '@remix-run/cloudflare';
import { SYSTEM_INFO } from '~/lib/modules/index';
import { runtime } from '~/lib/modules/vm/vm-runtime';
import { getGlobalMLModule } from '~/lib/modules/ml/ml.m';

/**
 * Remix loader that returns JSON status for NN/VM/ML subsystems.
 */
export async function loader({ request: _request }: LoaderFunctionArgs) {
  getGlobalMLModule();

  const processes = runtime.getAllProcesses();

  return json({
    system: SYSTEM_INFO,
    vm: {
      processes: processes.length,
      processList: processes.map((p) => ({ id: p.id, state: p.state })),
    },
    ml: {
      status: 'loaded',
    },
    timestamp: new Date().toISOString(),
  });
}
