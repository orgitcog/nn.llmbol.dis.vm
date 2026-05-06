import React from 'react';
import { useNNStatus } from '~/lib/hooks/useNNStatus';

/**
 * NNStatusPanel displays real-time status of the nn.llmbol.dis.vm subsystems.
 * Wire into Workbench.client.tsx behind a feature flag.
 */
export function NnStatusPanel() {
  const { status, error, loading } = useNNStatus(15_000);

  if (loading && !status) {
    return <div className="nn-status-panel p-2 text-sm text-bolt-elements-textSecondary">Loading NN status...</div>;
  }

  if (error) {
    return <div className="nn-status-panel p-2 text-sm text-red-500">NN Status error: {error}</div>;
  }

  if (!status) {
    return null;
  }

  return (
    <div className="nn-status-panel p-3 bg-bolt-elements-background-depth-2 rounded-lg text-xs">
      <h3 className="font-semibold text-bolt-elements-textPrimary mb-2">
        {status.system.name} v{status.system.version}
      </h3>
      <div className="space-y-1 text-bolt-elements-textSecondary">
        <div>
          VM Processes: <span className="text-bolt-elements-textPrimary">{status.vm.processes}</span>
        </div>
        <div>
          ML Module: <span className="text-bolt-elements-textPrimary">{status.ml.status}</span>
        </div>
        <div className="text-bolt-elements-textTertiary text-[10px]">
          Updated: {new Date(status.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}
