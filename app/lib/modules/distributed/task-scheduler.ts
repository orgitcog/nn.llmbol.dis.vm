/**
 * Task Scheduler
 *
 * Manages task distribution across compute nodes with load balancing
 * and fault tolerance.
 */

import type { ComputeNode, Task, NodeInfo } from '~/lib/modules/distributed/compute-node';

export type SchedulingStrategy = 'round-robin' | 'least-loaded' | 'random' | 'capability-based';

export interface SchedulerConfig {
  strategy: SchedulingStrategy;
  maxRetries: number;
  timeout: number;
}

export class TaskScheduler {
  private _nodes: Map<string, ComputeNode>;
  private _taskQueue: Task[];
  private _config: SchedulerConfig;
  private _nextNodeIndex: number;
  private _taskResults: Map<string, any>;
  private _cancelledTasks: Set<string>;
  private _pendingResolvers: Map<string, { resolve: (v: any) => void; reject: (e: Error) => void }>;
  private _isShuttingDown: boolean;

  constructor(config: Partial<SchedulerConfig> = {}) {
    this._nodes = new Map();
    this._taskQueue = [];
    this._config = {
      strategy: config.strategy ?? 'least-loaded',
      maxRetries: config.maxRetries ?? 3,
      timeout: config.timeout ?? 30000,
    };
    this._nextNodeIndex = 0;
    this._taskResults = new Map();
    this._cancelledTasks = new Set();
    this._pendingResolvers = new Map();
    this._isShuttingDown = false;
  }

  /**
   * Register a compute node with the scheduler.
   */
  registerNode(node: ComputeNode): void {
    const info = node.getInfo();
    this._nodes.set(info.id, node);
  }

  /**
   * Unregister a compute node. Returns true if it existed.
   */
  unregisterNode(nodeId: string): boolean {
    return this._nodes.delete(nodeId);
  }

  /**
   * Submit a single task for execution. Returns the task ID.
   */
  submitTask(task: Task): string {
    this._taskQueue.push(task);
    this._scheduleNext();

    return task.id;
  }

  /**
   * Submit multiple tasks. Returns their IDs.
   */
  submitTasks(tasks: Task[]): string[] {
    for (const task of tasks) {
      this._taskQueue.push(task);
    }

    this._scheduleNext();

    return tasks.map((t) => t.id);
  }

  /**
   * Returns a Promise that resolves with the task result, or rejects on error/cancellation.
   * If the result is already available it resolves immediately.
   */
  waitForResult(taskId: string): Promise<any> {
    if (this._taskResults.has(taskId)) {
      const stored = this._taskResults.get(taskId);

      if (stored && typeof stored === 'object' && 'error' in stored) {
        return Promise.reject(new Error(stored.error as string));
      }

      return Promise.resolve(stored);
    }

    return new Promise((resolve, reject) => {
      this._pendingResolvers.set(taskId, { resolve, reject });
    });
  }

  /**
   * Cancel a task. Removes it from the queue if pending, rejects its resolver.
   * Returns true if the task was found and cancelled.
   */
  cancelTask(taskId: string): boolean {
    if (this._cancelledTasks.has(taskId)) {
      return false;
    }

    this._cancelledTasks.add(taskId);

    const idx = this._taskQueue.findIndex((t) => t.id === taskId);

    if (idx > -1) {
      this._taskQueue.splice(idx, 1);
    }

    const resolver = this._pendingResolvers.get(taskId);

    if (resolver) {
      resolver.reject(new Error('Task cancelled'));
      this._pendingResolvers.delete(taskId);
    }

    return true;
  }

  /**
   * Shut down the scheduler: clears the queue and rejects all pending resolvers.
   */
  async shutdown(): Promise<void> {
    this._isShuttingDown = true;

    for (const task of this._taskQueue) {
      const resolver = this._pendingResolvers.get(task.id);

      if (resolver) {
        resolver.reject(new Error('Scheduler shut down'));
        this._pendingResolvers.delete(task.id);
      }
    }

    this._taskQueue = [];
  }

  /**
   * Get the stored result for a completed task, or undefined.
   */
  getResult(taskId: string): any {
    return this._taskResults.get(taskId);
  }

  /**
   * Returns true when a result (or error record) has been stored for the task.
   */
  isComplete(taskId: string): boolean {
    return this._taskResults.has(taskId);
  }

  /**
   * Get scheduler statistics.
   */
  getStats() {
    const nodeStats = Array.from(this._nodes.values()).map((node) => {
      const info = node.getInfo();
      return { id: info.id, status: info.status, load: info.load, tasks: node.getTaskCount() };
    });

    return {
      totalNodes: this._nodes.size,
      queuedTasks: this._taskQueue.length,
      completedTasks: this._taskResults.size,
      nodes: nodeStats,
      strategy: this._config.strategy,
    };
  }

  /**
   * Get info for all registered nodes.
   */
  getNodes(): NodeInfo[] {
    return Array.from(this._nodes.values()).map((node) => node.getInfo());
  }

  /**
   * Clear all stored task results.
   */
  clearResults(): void {
    this._taskResults.clear();
  }

  private _scheduleNext(): void {
    if (this._taskQueue.length === 0 || this._isShuttingDown) {
      return;
    }

    this._taskQueue.sort((a, b) => b.priority - a.priority);

    const tasksToSchedule = [...this._taskQueue];

    for (const task of tasksToSchedule) {
      if (this._cancelledTasks.has(task.id)) {
        const idx = this._taskQueue.indexOf(task);

        if (idx > -1) {
          this._taskQueue.splice(idx, 1);
        }

        continue;
      }

      const node = this._selectNode(task);

      if (node && node.assignTask(task)) {
        const idx = this._taskQueue.indexOf(task);

        if (idx > -1) {
          this._taskQueue.splice(idx, 1);
        }

        this._executeTask(node, task);
      }
    }
  }

  private _selectNode(task: Task): ComputeNode | null {
    const available = Array.from(this._nodes.values()).filter((n) => n.canAcceptTask());

    if (available.length === 0) {
      return null;
    }

    switch (this._config.strategy) {
      case 'round-robin':
        return this._selectRoundRobin(available);
      case 'least-loaded':
        return this._selectLeastLoaded(available);
      case 'random':
        return this._selectRandom(available);
      case 'capability-based':
        return this._selectByCapability(available, task);
      default:
        return available[0];
    }
  }

  private _selectRoundRobin(nodes: ComputeNode[]): ComputeNode {
    const node = nodes[this._nextNodeIndex % nodes.length];
    this._nextNodeIndex++;

    return node;
  }

  private _selectLeastLoaded(nodes: ComputeNode[]): ComputeNode {
    return nodes.reduce((least, current) => (current.getLoad() < least.getLoad() ? current : least));
  }

  private _selectRandom(nodes: ComputeNode[]): ComputeNode {
    return nodes[Math.floor(Math.random() * nodes.length)];
  }

  private _selectByCapability(nodes: ComputeNode[], task: Task): ComputeNode | null {
    const capable = nodes.filter((n) => n.hasCapability(task.type));
    const pool = capable.length > 0 ? capable : nodes;

    return pool.length > 0 ? this._selectLeastLoaded(pool) : null;
  }

  private async _executeTask(node: ComputeNode, task: Task): Promise<void> {
    if (this._cancelledTasks.has(task.id) || this._isShuttingDown) {
      return;
    }

    try {
      const result = await Promise.race([node.executeTask(task.id), this._timeout(this._config.timeout)]);

      this._taskResults.set(task.id, result);

      const resolver = this._pendingResolvers.get(task.id);

      if (resolver) {
        resolver.resolve(result);
        this._pendingResolvers.delete(task.id);
      }

      this._scheduleNext();
    } catch (err) {
      console.error(`Task ${task.id} failed:`, err);

      task.retryCount = (task.retryCount ?? 0) + 1;

      if (task.retryCount <= this._config.maxRetries && !this._cancelledTasks.has(task.id) && !this._isShuttingDown) {
        this._taskQueue.push(task);
        this._scheduleNext();
      } else {
        const errorMsg = err instanceof Error ? err.message : 'Max retries exceeded';
        this._taskResults.set(task.id, { error: errorMsg });

        const resolver = this._pendingResolvers.get(task.id);

        if (resolver) {
          resolver.reject(new Error(errorMsg));
          this._pendingResolvers.delete(task.id);
        }
      }
    }
  }

  private _timeout(ms: number): Promise<never> {
    return new Promise((_, reject) => setTimeout(() => reject(new Error('Task timeout')), ms));
  }
}

/**
 * Create a new task scheduler.
 */
export function createScheduler(config?: Partial<SchedulerConfig>): TaskScheduler {
  return new TaskScheduler(config);
}

/**
 * Create a task with the given parameters.
 */
export function createTask(id: string, type: string, payload: any, priority: number = 0): Task {
  return {
    id,
    type,
    payload,
    priority,
    createdAt: Date.now(),
  };
}
