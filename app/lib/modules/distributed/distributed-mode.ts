import {
  createCommunicationWithTransport,
  type Communication,
  type CommunicationConfig,
  type Message,
  type TransportAdapter,
} from '~/lib/modules/distributed/communication';
import { createComputeNode, type ComputeNode, type NodeInfo, type Task } from '~/lib/modules/distributed/compute-node';
import { createScheduler, type SchedulerConfig, type TaskScheduler } from '~/lib/modules/distributed/task-scheduler';

export interface DistributedModeConfig {
  nodeId: string;
  address?: string;
  port?: number;
  capabilities?: string[];
  maxConcurrentTasks?: number;
  scheduler?: Partial<SchedulerConfig>;
  communication?: Partial<CommunicationConfig>;
  remoteExecutionTimeout?: number;
}

export interface DistributedExecuteOptions {
  preferRemote?: boolean;
}

interface TaskExecutionRequestPayload {
  kind: 'execute-task';
  task: Task;
}

interface TaskExecutionResponsePayload {
  kind: 'task-result';
  taskId: string;
  result?: unknown;
  error?: string;
}

interface PendingRemoteTask {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class DistributedModeCoordinator {
  private readonly _config: DistributedModeConfig;
  private readonly _localNode: ComputeNode;
  private readonly _scheduler: TaskScheduler;
  private readonly _communication: Communication;
  private readonly _pendingRemoteTasks = new Map<string, PendingRemoteTask>();
  private _started = false;

  constructor(config: DistributedModeConfig, transport?: TransportAdapter) {
    this._config = {
      ...config,
      remoteExecutionTimeout: config.remoteExecutionTimeout ?? 10_000,
    };

    this._localNode = createComputeNode(
      this._config.nodeId,
      this._config.address ?? 'localhost',
      this._config.port ?? 8080,
      this._config.capabilities ?? ['inference', 'training', 'data_processing'],
    );

    this._scheduler = createScheduler(this._config.scheduler);
    this._scheduler.registerNode(this._localNode);

    this._communication = createCommunicationWithTransport(
      this._config.nodeId,
      transport ?? {
        send: async () => undefined,
        onReceive: () => undefined,
        isConnected: () => true,
      },
      this._config.communication,
    );

    this._communication.onMessage('request', (message) => {
      void this._handleRequestMessage(message);
    });

    this._communication.onMessage('response', (message) => {
      this._handleResponseMessage(message);
    });
  }

  start(): void {
    if (this._started) {
      return;
    }

    this._communication.start();
    this._started = true;
  }

  stop(): void {
    if (!this._started) {
      return;
    }

    this._communication.stop();
    this._started = false;

    for (const [, pending] of this._pendingRemoteTasks.entries()) {
      clearTimeout(pending.timer);
    }

    this._pendingRemoteTasks.clear();
  }

  registerPeer(peerId: string): void {
    this._communication.registerPeer(peerId);
  }

  unregisterPeer(peerId: string): boolean {
    return this._communication.unregisterPeer(peerId);
  }

  getLocalNodeInfo(): NodeInfo {
    return this._localNode.getInfo();
  }

  async executeTask(task: Task, options: DistributedExecuteOptions = {}): Promise<unknown> {
    if (options.preferRemote) {
      try {
        return await this._executeTaskRemote(task);
      } catch {
        return this._executeTaskLocal(task);
      }
    }

    return this._executeTaskLocal(task);
  }

  getStats() {
    return {
      started: this._started,
      localNode: this._localNode.getInfo(),
      scheduler: this._scheduler.getStats(),
      communication: this._communication.getStats(),
      pendingRemoteTasks: this._pendingRemoteTasks.size,
    };
  }

  private _executeTaskLocal(task: Task): Promise<unknown> {
    this._scheduler.submitTask(task);
    return this._scheduler.waitForResult(task.id);
  }

  private async _executeTaskRemote(task: Task): Promise<unknown> {
    const remotePeerId = this._communication.getOnlinePeers().find((peerId) => peerId !== this._config.nodeId);

    if (!remotePeerId) {
      throw new Error('No remote peers available');
    }

    const payload: TaskExecutionRequestPayload = {
      kind: 'execute-task',
      task,
    };

    const resultPromise = new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this._pendingRemoteTasks.delete(task.id);
        reject(new Error(`Remote task ${task.id} timed out`));
      }, this._config.remoteExecutionTimeout);

      this._pendingRemoteTasks.set(task.id, { resolve, reject, timer });
    });

    try {
      await this._communication.send(remotePeerId, payload, 'request');
    } catch (error) {
      const pending = this._pendingRemoteTasks.get(task.id);

      if (pending) {
        clearTimeout(pending.timer);
        this._pendingRemoteTasks.delete(task.id);
      }

      throw error;
    }

    return resultPromise;
  }

  private async _handleRequestMessage(message: Message): Promise<void> {
    if (!this._isAddressedToCurrentNode(message.to)) {
      return;
    }

    const payload = message.payload as TaskExecutionRequestPayload | undefined;

    if (!payload || payload.kind !== 'execute-task' || !payload.task) {
      return;
    }

    try {
      const result = await this._executeTaskLocal(payload.task);

      const response: TaskExecutionResponsePayload = {
        kind: 'task-result',
        taskId: payload.task.id,
        result,
      };

      await this._communication.send(message.from, response, 'response');
    } catch (error) {
      const response: TaskExecutionResponsePayload = {
        kind: 'task-result',
        taskId: payload.task.id,
        error: error instanceof Error ? error.message : String(error),
      };

      await this._communication.send(message.from, response, 'response');
    }
  }

  private _handleResponseMessage(message: Message): void {
    if (!this._isAddressedToCurrentNode(message.to)) {
      return;
    }

    const payload = message.payload as TaskExecutionResponsePayload | undefined;

    if (!payload || payload.kind !== 'task-result') {
      return;
    }

    const pending = this._pendingRemoteTasks.get(payload.taskId);

    if (!pending) {
      return;
    }

    clearTimeout(pending.timer);
    this._pendingRemoteTasks.delete(payload.taskId);

    if (payload.error) {
      pending.reject(new Error(payload.error));
    } else {
      pending.resolve(payload.result);
    }
  }

  private _isAddressedToCurrentNode(to: string | string[]): boolean {
    if (Array.isArray(to)) {
      return to.includes(this._config.nodeId);
    }

    return to === this._config.nodeId;
  }
}

export function createDistributedModeCoordinator(
  config: DistributedModeConfig,
  transport?: TransportAdapter,
): DistributedModeCoordinator {
  return new DistributedModeCoordinator(config, transport);
}
