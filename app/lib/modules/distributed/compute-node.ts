/**
 * Distributed Compute Node
 *
 * Represents a compute node in a distributed system that can execute
 * tasks and communicate with other nodes.
 */

import { getGlobalMLModule } from '~/lib/modules/ml/ml.m';

export interface NodeInfo {
  id: string;
  address: string;
  port: number;
  status: 'idle' | 'busy' | 'offline';
  capabilities: string[];
  load: number;
}

export interface Task {
  id: string;
  type: string;
  payload: any;
  priority: number;
  createdAt: number;
  assignedTo?: string;

  /** Number of times this task has been retried. Used by the scheduler for fault tolerance. */
  retryCount?: number;
}

export class ComputeNode {
  private _info: NodeInfo;
  private _tasks: Map<string, Task>;
  private _maxConcurrentTasks: number;
  private _completionCallbacks: Map<string, (result: any, error?: Error) => void>;

  constructor(id: string, address: string, port: number, capabilities: string[] = [], maxConcurrentTasks: number = 4) {
    this._info = {
      id,
      address,
      port,
      status: 'idle',
      capabilities,
      load: 0,
    };
    this._tasks = new Map();
    this._maxConcurrentTasks = maxConcurrentTasks;
    this._completionCallbacks = new Map();
  }

  /**
   * Get node information.
   */
  getInfo(): NodeInfo {
    return { ...this._info };
  }

  /**
   * Update node status.
   */
  setStatus(status: NodeInfo['status']): void {
    this._info.status = status;
  }

  /**
   * Check if node can accept more tasks.
   */
  canAcceptTask(): boolean {
    return this._info.status !== 'offline' && this._tasks.size < this._maxConcurrentTasks;
  }

  /**
   * Assign a task to this node. Returns false if the node cannot accept it.
   */
  assignTask(task: Task): boolean {
    if (!this.canAcceptTask()) {
      return false;
    }

    task.assignedTo = this._info.id;
    this._tasks.set(task.id, task);
    this._updateLoad();

    if (this._tasks.size > 0) {
      this._info.status = 'busy';
    }

    return true;
  }

  /**
   * Execute a previously assigned task. Invokes any registered completion callback.
   */
  async executeTask(taskId: string): Promise<any> {
    const task = this._tasks.get(taskId);

    if (!task) {
      throw new Error(`Task ${taskId} not found`);
    }

    try {
      const result = await this._processTask(task);
      this._tasks.delete(taskId);
      this._updateLoad();

      if (this._tasks.size === 0) {
        this._info.status = 'idle';
      }

      const cb = this._completionCallbacks.get(taskId);

      if (cb) {
        cb(result, undefined);
        this._completionCallbacks.delete(taskId);
      }

      return result;
    } catch (err) {
      this._tasks.delete(taskId);
      this._updateLoad();

      const error = err instanceof Error ? err : new Error(String(err));
      const cb = this._completionCallbacks.get(taskId);

      if (cb) {
        cb(undefined, error);
        this._completionCallbacks.delete(taskId);
      }

      throw error;
    }
  }

  /**
   * Register a callback that fires when the given task completes or fails.
   */
  onTaskComplete(taskId: string, callback: (result: any, error?: Error) => void): void {
    this._completionCallbacks.set(taskId, callback);
  }

  /**
   * Get all currently assigned tasks.
   */
  getTasks(): Task[] {
    return Array.from(this._tasks.values());
  }

  /**
   * Get the number of currently assigned tasks.
   */
  getTaskCount(): number {
    return this._tasks.size;
  }

  /**
   * Get the current load ratio (0–1).
   */
  getLoad(): number {
    return this._info.load;
  }

  /**
   * Check if this node has a specific capability.
   */
  hasCapability(capability: string): boolean {
    return this._info.capabilities.includes(capability);
  }

  /**
   * Add a capability to this node.
   */
  addCapability(capability: string): void {
    if (!this.hasCapability(capability)) {
      this._info.capabilities.push(capability);
    }
  }

  /**
   * Remove a capability from this node.
   */
  removeCapability(capability: string): void {
    const index = this._info.capabilities.indexOf(capability);

    if (index > -1) {
      this._info.capabilities.splice(index, 1);
    }
  }

  private async _processTask(task: Task): Promise<any> {
    switch (task.type) {
      case 'inference':
        return this._processInference(task.payload);
      case 'training':
        return this._processTraining(task.payload);
      case 'data_processing':
        return this._processData(task.payload);
      default:
        return { success: true, taskId: task.id };
    }
  }

  private async _processInference(payload: any): Promise<any> {
    const ml = getGlobalMLModule();

    if (Array.isArray(payload?.data) && (payload.data as unknown[]).length > 0) {
      const nums = (payload.data as unknown[]).map(Number);
      const tensor = ml.createTensor([nums.length], 'float32');
      const floats = tensor.data as Float32Array;

      for (let i = 0; i < nums.length; i++) {
        floats[i] = nums[i];
      }

      const activated = ml.relu(tensor);
      const probabilities = ml.softmax(activated);

      return {
        type: 'inference',
        result: Array.from(probabilities.data as Float32Array),
        shape: probabilities.shape,
      };
    }

    await this._delay(100);

    return { type: 'inference', result: payload };
  }

  private async _processTraining(payload: any): Promise<any> {
    await this._delay(500);
    return { type: 'training', result: payload };
  }

  private async _processData(payload: any): Promise<any> {
    if (Array.isArray(payload?.data) && (payload.data as unknown[]).length > 0) {
      const nums = (payload.data as unknown[]).map(Number);
      const sum = nums.reduce((a, b) => a + b, 0);
      const mean = sum / nums.length;
      const min = Math.min(...nums);
      const max = Math.max(...nums);

      return { type: 'data_processing', result: { mean, min, max, count: nums.length } };
    }

    await this._delay(50);

    return { type: 'data_processing', result: payload };
  }

  private _updateLoad(): void {
    this._info.load = this._tasks.size / this._maxConcurrentTasks;
  }

  private _delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

/**
 * Create a new compute node.
 */
export function createComputeNode(
  id: string,
  address: string = 'localhost',
  port: number = 8080,
  capabilities: string[] = [],
): ComputeNode {
  return new ComputeNode(id, address, port, capabilities);
}
