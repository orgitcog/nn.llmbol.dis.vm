/**
 * Task Scheduler
 * 
 * Manages task distribution across compute nodes with load balancing
 * and fault tolerance
 */

import type { ComputeNode, Task, NodeInfo } from './compute-node';

export type SchedulingStrategy = 'round-robin' | 'least-loaded' | 'random' | 'capability-based';

export interface SchedulerConfig {
  strategy: SchedulingStrategy;
  maxRetries: number;
  timeout: number;
}

export class TaskScheduler {
  private nodes: Map<string, ComputeNode>;
  private taskQueue: Task[];
  private config: SchedulerConfig;
  private nextNodeIndex: number;
  private taskResults: Map<string, any>;

  constructor(config: Partial<SchedulerConfig> = {}) {
    this.nodes = new Map();
    this.taskQueue = [];
    this.config = {
      strategy: config.strategy || 'least-loaded',
      maxRetries: config.maxRetries || 3,
      timeout: config.timeout || 30000,
    };
    this.nextNodeIndex = 0;
    this.taskResults = new Map();
  }

  /**
   * Register a compute node
   */
  registerNode(node: ComputeNode): void {
    const info = node.getInfo();
    this.nodes.set(info.id, node);
  }

  /**
   * Unregister a compute node
   */
  unregisterNode(nodeId: string): boolean {
    return this.nodes.delete(nodeId);
  }

  /**
   * Submit a task for execution
   */
  submitTask(task: Task): string {
    this.taskQueue.push(task);
    this.scheduleNext();
    return task.id;
  }

  /**
   * Submit multiple tasks
   */
  submitTasks(tasks: Task[]): string[] {
    for (const task of tasks) {
      this.taskQueue.push(task);
    }
    this.scheduleNext();
    return tasks.map(t => t.id);
  }

  /**
   * Schedule next task from queue
   */
  private scheduleNext(): void {
    if (this.taskQueue.length === 0) {
      return;
    }

    // Sort queue by priority
    this.taskQueue.sort((a, b) => b.priority - a.priority);

    // Try to schedule tasks
    const tasksToSchedule = [...this.taskQueue];
    for (const task of tasksToSchedule) {
      const node = this.selectNode(task);
      if (node) {
        if (node.assignTask(task)) {
          // Remove from queue
          const index = this.taskQueue.indexOf(task);
          if (index > -1) {
            this.taskQueue.splice(index, 1);
          }

          // Execute task
          this.executeTask(node, task);
        }
      }
    }
  }

  /**
   * Select a node for task execution based on strategy
   */
  private selectNode(task: Task): ComputeNode | null {
    const availableNodes = Array.from(this.nodes.values()).filter(node => 
      node.canAcceptTask()
    );

    if (availableNodes.length === 0) {
      return null;
    }

    switch (this.config.strategy) {
      case 'round-robin':
        return this.selectRoundRobin(availableNodes);
      case 'least-loaded':
        return this.selectLeastLoaded(availableNodes);
      case 'random':
        return this.selectRandom(availableNodes);
      case 'capability-based':
        return this.selectByCapability(availableNodes, task);
      default:
        return availableNodes[0];
    }
  }

  /**
   * Round-robin selection
   */
  private selectRoundRobin(nodes: ComputeNode[]): ComputeNode {
    const node = nodes[this.nextNodeIndex % nodes.length];
    this.nextNodeIndex++;
    return node;
  }

  /**
   * Select least loaded node
   */
  private selectLeastLoaded(nodes: ComputeNode[]): ComputeNode {
    return nodes.reduce((least, current) => 
      current.getLoad() < least.getLoad() ? current : least
    );
  }

  /**
   * Random selection
   */
  private selectRandom(nodes: ComputeNode[]): ComputeNode {
    const index = Math.floor(Math.random() * nodes.length);
    return nodes[index];
  }

  /**
   * Select by capability
   */
  private selectByCapability(nodes: ComputeNode[], task: Task): ComputeNode | null {
    // Filter nodes that have required capability
    const capableNodes = nodes.filter(node => 
      node.hasCapability(task.type)
    );

    if (capableNodes.length === 0) {
      // Fallback to any available node
      return nodes.length > 0 ? nodes[0] : null;
    }

    // Select least loaded among capable nodes
    return this.selectLeastLoaded(capableNodes);
  }

  /**
   * Execute task on node
   */
  private async executeTask(node: ComputeNode, task: Task): Promise<void> {
    try {
      const result = await Promise.race([
        node.executeTask(task.id),
        this.timeout(this.config.timeout),
      ]);

      this.taskResults.set(task.id, result);
      
      // Schedule next task
      this.scheduleNext();
    } catch (error) {
      console.error(`Task ${task.id} failed:`, error);
      
      // Re-queue task if retries remain
      if (task.priority > -this.config.maxRetries) {
        task.priority--;
        this.taskQueue.push(task);
        this.scheduleNext();
      } else {
        this.taskResults.set(task.id, { error: 'Max retries exceeded' });
      }
    }
  }

  /**
   * Timeout helper
   */
  private timeout(ms: number): Promise<never> {
    return new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Task timeout')), ms)
    );
  }

  /**
   * Get task result
   */
  getResult(taskId: string): any {
    return this.taskResults.get(taskId);
  }

  /**
   * Check if task is complete
   */
  isComplete(taskId: string): boolean {
    return this.taskResults.has(taskId);
  }

  /**
   * Get scheduler statistics
   */
  getStats() {
    const nodeStats = Array.from(this.nodes.values()).map(node => {
      const info = node.getInfo();
      return {
        id: info.id,
        status: info.status,
        load: info.load,
        tasks: node.getTaskCount(),
      };
    });

    return {
      totalNodes: this.nodes.size,
      queuedTasks: this.taskQueue.length,
      completedTasks: this.taskResults.size,
      nodes: nodeStats,
      strategy: this.config.strategy,
    };
  }

  /**
   * Get all nodes
   */
  getNodes(): NodeInfo[] {
    return Array.from(this.nodes.values()).map(node => node.getInfo());
  }

  /**
   * Clear completed results
   */
  clearResults(): void {
    this.taskResults.clear();
  }
}

/**
 * Create a new task scheduler
 */
export function createScheduler(config?: Partial<SchedulerConfig>): TaskScheduler {
  return new TaskScheduler(config);
}

/**
 * Create a task
 */
export function createTask(
  id: string,
  type: string,
  payload: any,
  priority: number = 0
): Task {
  return {
    id,
    type,
    payload,
    priority,
    createdAt: Date.now(),
  };
}
