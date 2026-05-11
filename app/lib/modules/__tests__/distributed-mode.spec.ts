import { beforeEach, describe, expect, it } from 'vitest';
import type { Message, TransportAdapter } from '~/lib/modules/distributed/communication';
import { createTask } from '~/lib/modules/distributed/task-scheduler';
import { createDistributedModeCoordinator } from '~/lib/modules/distributed/distributed-mode';

class InMemoryTransport implements TransportAdapter {
  private static _handlers = new Map<string, (message: Message) => void>();
  private readonly _nodeId: string;

  constructor(nodeId: string) {
    this._nodeId = nodeId;
  }

  static reset(): void {
    this._handlers.clear();
  }

  async send(to: string | string[], message: Message): Promise<void> {
    const recipients = Array.isArray(to) ? to : [to];

    for (const peerId of recipients) {
      const handler = InMemoryTransport._handlers.get(peerId);

      if (handler) {
        handler(message);
      }
    }
  }

  onReceive(handler: (message: Message) => void): void {
    InMemoryTransport._handlers.set(this._nodeId, handler);
  }

  isConnected(): boolean {
    return true;
  }
}

class FailingRequestTransport extends InMemoryTransport {
  override async send(to: string | string[], message: Message): Promise<void> {
    if (message.type === 'request') {
      throw new Error('Simulated request transport failure');
    }

    return super.send(to, message);
  }
}

describe('DistributedModeCoordinator', () => {
  beforeEach(() => {
    InMemoryTransport.reset();
  });

  it('should execute tasks locally by default', async () => {
    const coordinator = createDistributedModeCoordinator(
      {
        nodeId: 'local-node',
      },
      new InMemoryTransport('local-node'),
    );

    coordinator.start();

    const task = createTask('task-local', 'data_processing', { data: [1, 2, 3] }, 1);
    const result = (await coordinator.executeTask(task)) as {
      type: string;
      result: { mean: number; count: number };
    };

    expect(result.type).toBe('data_processing');
    expect(result.result.mean).toBe(2);
    expect(result.result.count).toBe(3);

    coordinator.stop();
  });

  it('should execute tasks on remote peers when preferred', async () => {
    const coordinatorA = createDistributedModeCoordinator(
      {
        nodeId: 'node-a',
      },
      new InMemoryTransport('node-a'),
    );

    const coordinatorB = createDistributedModeCoordinator(
      {
        nodeId: 'node-b',
      },
      new InMemoryTransport('node-b'),
    );

    coordinatorA.start();
    coordinatorB.start();

    coordinatorA.registerPeer('node-b');
    coordinatorB.registerPeer('node-a');

    const task = createTask('task-remote', 'data_processing', { data: [2, 4, 6] }, 1);
    const result = (await coordinatorA.executeTask(task, { preferRemote: true })) as {
      type: string;
      result: { mean: number };
    };

    const statsA = coordinatorA.getStats();
    const statsB = coordinatorB.getStats();

    expect(result.type).toBe('data_processing');
    expect(result.result.mean).toBe(4);
    expect(statsA.scheduler.completedTasks).toBe(0);
    expect(statsB.scheduler.completedTasks).toBe(1);

    coordinatorA.stop();
    coordinatorB.stop();
  });

  it('should fall back to local execution when remote request fails', async () => {
    const coordinator = createDistributedModeCoordinator(
      {
        nodeId: 'fallback-node',
      },
      new FailingRequestTransport('fallback-node'),
    );

    coordinator.start();
    coordinator.registerPeer('unreachable-peer');

    const task = createTask('task-fallback', 'data_processing', { data: [3, 3, 3] }, 1);
    const result = (await coordinator.executeTask(task, { preferRemote: true })) as {
      type: string;
      result: { mean: number };
    };

    const stats = coordinator.getStats();

    expect(result.type).toBe('data_processing');
    expect(result.result.mean).toBe(3);
    expect(stats.scheduler.completedTasks).toBe(1);

    coordinator.stop();
  });
});
