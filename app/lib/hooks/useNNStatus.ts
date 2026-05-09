import { useState, useEffect, useCallback } from 'react';

export interface NNStatus {
  system: { name: string; version: string; components: Record<string, string> };
  vm: { loadedModules: number; processes: number; processList: Array<{ id: string; state: string }> };
  ml: { status: string; loadedModels: number; modelNames: string[] };
  timestamp: string;
}

/**
 * React hook that polls the /api/nn-status endpoint at a configurable interval.
 */
export function useNNStatus(pollIntervalMs: number = 10_000) {
  const [status, setStatus] = useState<NNStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    setLoading(true);

    try {
      const res = await fetch('/api/nn-status');

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = (await res.json()) as NNStatus;
      setStatus(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();

    const interval = setInterval(fetchStatus, pollIntervalMs);

    return () => clearInterval(interval);
  }, [fetchStatus, pollIntervalMs]);

  return { status, error, loading, refresh: fetchStatus };
}
